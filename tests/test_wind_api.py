import io
import struct
import zlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn
from torchvision import transforms as T

import app.main as api
from ml.wind_inference import PRODUCTION_CHECKPOINT, preprocess_wind_image


class FixedWindModel(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, images):
        return torch.full((len(images), 1), self.value, device=images.device)


@pytest.fixture(autouse=True)
def clear_wind_model_cache():
    api._wind_model = None
    api._wind_device = None
    yield
    api._wind_model = None
    api._wind_device = None


@pytest.fixture
def png_bytes():
    image = Image.new("RGB", (64, 32), (180, 90, 30))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_preprocessing_matches_evaluation_rgb_resize_and_normalization(png_bytes):
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    transform = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    actual = preprocess_wind_image(png_bytes)
    assert actual.shape == (1, 3, 224, 448)
    assert torch.equal(actual[0], transform(image))


def test_wind_endpoint_keeps_raw_and_clipped_display_values_separate(png_bytes, monkeypatch):
    loader = Mock(return_value=(FixedWindModel(-0.05), torch.device("cpu")))
    monkeypatch.setattr(api, "load_wind_model", loader)
    client = TestClient(api.app)
    response = client.post("/predict/wind", files={"file": ("storm.png", png_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Model A ResNet-18 IR wind regression"
    assert body["checkpoint"] == "multiyear45_resnet18_ir_weighted.pt"
    assert body["unit"] == "kt"
    assert body["normalized_model_output"] == pytest.approx(-0.05)
    assert body["raw_prediction_kt"] == pytest.approx(-5.0)
    assert body["application_prediction_kt"] == 0
    assert body["predicted_wind_kt"] == 0
    assert body["category"] == "TD"


def test_model_is_loaded_once_for_multiple_requests(png_bytes, monkeypatch):
    loader = Mock(return_value=(FixedWindModel(0.42), torch.device("cpu")))
    monkeypatch.setattr(api, "load_wind_model", loader)
    client = TestClient(api.app)
    for _ in range(2):
        response = client.post("/predict/wind", files={"file": ("storm.png", png_bytes, "image/png")})
        assert response.status_code == 200
        assert response.json()["raw_prediction_kt"] == pytest.approx(42.0)
    loader.assert_called_once_with()


def test_invalid_mime_and_corrupt_image_are_rejected(png_bytes):
    client = TestClient(api.app)
    wrong_type = client.post("/predict/wind", files={"file": ("storm.txt", png_bytes, "text/plain")})
    assert wrong_type.status_code == 415
    invalid_image = client.post("/predict/wind", files={"file": ("storm.png", b"not an image", "image/png")})
    assert invalid_image.status_code == 400
    assert "valid readable image" in invalid_image.json()["detail"]


def test_upload_byte_and_decoded_pixel_limits(png_bytes):
    client = TestClient(api.app)
    too_large = client.post("/predict/wind", files={"file": ("storm.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")})
    assert too_large.status_code == 413
    huge_header = bytearray(png_bytes)
    struct.pack_into(">II", huge_header, 16, 6_500, 6_500)
    struct.pack_into(">I", huge_header, 29, zlib.crc32(huge_header[12:29]) & 0xFFFFFFFF)
    too_many_pixels = client.post("/predict/wind", files={"file": ("storm.png", bytes(huge_header), "image/png")})
    assert too_many_pixels.status_code == 400
    assert "40 megapixel" in too_many_pixels.json()["detail"]


def test_nonfinite_prediction_is_a_service_error_not_a_zero_estimate(png_bytes, monkeypatch):
    monkeypatch.setattr(api, "load_wind_model", lambda: (FixedWindModel(float("nan")), torch.device("cpu")))
    response = TestClient(api.app).post("/predict/wind", files={"file": ("storm.png", png_bytes, "image/png")})
    assert response.status_code == 503
    assert "No estimate was produced" in response.json()["detail"]


def test_checkpoint_loading_failure_is_a_clear_service_error(png_bytes, monkeypatch):
    def fail():
        raise FileNotFoundError("missing checkpoint")
    monkeypatch.setattr(api, "load_wind_model", fail)
    response = TestClient(api.app).post("/predict/wind", files={"file": ("storm.png", png_bytes, "image/png")})
    assert response.status_code == 503
    assert "Model A checkpoint" in response.json()["detail"]
    assert "missing checkpoint" not in response.json()["detail"]


def test_existing_pattern_route_contract_is_unchanged(png_bytes):
    response = TestClient(api.app).post("/predict/pattern", files={"file": ("storm.png", png_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert {"status", "pattern_predicted", "dvorak_taxonomy", "confidence", "probabilities", "grad_cam_grid"} <= body.keys()


@pytest.mark.skipif(not PRODUCTION_CHECKPOINT.is_file(), reason="Model A checkpoint is not present")
def test_real_model_a_checkpoint_predicts_from_a_synthetic_image(png_bytes):
    response = TestClient(api.app).post("/predict/wind", files={"file": ("storm.png", png_bytes, "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert np.isfinite(body["raw_prediction_kt"])
    assert body["checkpoint"] == PRODUCTION_CHECKPOINT.name
    assert body["application_prediction_kt"] == max(body["raw_prediction_kt"], 0)


# --- optional central pressure alongside the image (multi-metric inference) ---


def _wind_response(png_bytes, monkeypatch, data=None):
    monkeypatch.setattr(api, "load_wind_model", lambda: (FixedWindModel(0.9), torch.device("cpu")))
    client = TestClient(api.app)
    return client.post(
        "/predict/wind",
        files={"file": ("chip.png", png_bytes, "image/png")},
        data=data or {},
    )


def test_pressure_input_adds_a_second_independent_intensity_estimate(png_bytes, monkeypatch):
    body = _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "950"}).json()
    multi = body["multi_metric"]
    assert multi["observed_pressure_hpa"] == pytest.approx(950.0)
    # Atkinson-Holliday puts 950 hPa near 95 kt; the image model here is pinned to 90 kt.
    assert 80.0 < multi["pressure_derived_wind_kt"] < 110.0
    assert multi["image_minus_pressure_kt"] == pytest.approx(
        body["raw_prediction_kt"] - multi["pressure_derived_wind_kt"], abs=0.11
    )


def test_the_image_estimate_is_not_altered_by_the_pressure_the_user_typed(png_bytes, monkeypatch):
    """Model A takes pixels only. Letting a typed pressure move its output would be a lie."""
    without = _wind_response(png_bytes, monkeypatch).json()
    with_pressure = _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "905"}).json()
    assert without["raw_prediction_kt"] == pytest.approx(with_pressure["raw_prediction_kt"])


def test_without_pressure_no_second_estimate_is_reported(png_bytes, monkeypatch):
    assert _wind_response(png_bytes, monkeypatch).json()["multi_metric"] is None


def test_the_image_estimate_also_reports_its_own_derived_pressure(png_bytes, monkeypatch):
    body = _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "950"}).json()
    assert 880.0 <= body["multi_metric"]["image_derived_pressure_hpa"] <= 1012.0


def test_an_impossible_pressure_is_refused(png_bytes, monkeypatch):
    assert _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "1500"}).status_code == 422
    assert _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "not a number"}).status_code == 422


def test_agreement_is_described_in_words_for_the_dashboard(png_bytes, monkeypatch):
    body = _wind_response(png_bytes, monkeypatch, {"pressure_hpa": "950"}).json()
    assert body["multi_metric"]["agreement"] in {"close", "moderate", "poor"}
