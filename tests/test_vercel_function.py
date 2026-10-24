"""Tests for the Vercel serverless trajectory function.

The function is a plain BaseHTTPRequestHandler rather than a FastAPI app, so none of the
framework's validation applies and the handler has to do it itself. These tests run the
real handler over a real socket, because that is the only way to exercise the request
parsing, the status codes and the body limits together.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUNCTION_PATH = PROJECT_ROOT / "api" / "predict" / "trajectory.py"

AMPHAN = [
    {"lat": 12.5, "lon": 86.4, "wind_kt": 120, "pressure_hpa": 932, "timestamp": "2020-05-17T18:00:00Z"},
    {"lat": 13.2, "lon": 86.4, "wind_kt": 130, "pressure_hpa": 919, "timestamp": "2020-05-18T00:00:00Z"},
    {"lat": 13.4, "lon": 86.2, "wind_kt": 140, "pressure_hpa": 911, "timestamp": "2020-05-18T06:00:00Z"},
    {"lat": 14.1, "lon": 86.4, "wind_kt": 145, "pressure_hpa": 901, "timestamp": "2020-05-18T12:00:00Z"},
]


def _load_handler():
    spec = importlib.util.spec_from_file_location("vercel_trajectory", FUNCTION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server():
    module = _load_handler()
    httpd = HTTPServer(("127.0.0.1", 0), module.handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


def post(base: str, payload, raw: bytes | None = None, content_type: str = "application/json"):
    body = raw if raw is not None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base, data=body, headers={"content-type": content_type}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw_body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(raw_body)
        except json.JSONDecodeError:
            return error.code, {"raw": raw_body}


class TestSuccess:
    def test_a_valid_request_returns_the_documented_shape(self, server):
        status, body = post(server, {"cyclone_id": "AMPHAN", "observations": AMPHAN})
        assert status == 200
        for key in (
            "cyclone_id", "region", "forecast_generated_at", "source", "runtime",
            "current_metrics", "waypoints", "landfall_estimate", "summary", "disclaimer",
        ):
            assert key in body
        assert body["cyclone_id"] == "AMPHAN"
        assert body["region"] == "Bay of Bengal"
        assert len(body["waypoints"]) == 8

    def test_it_serves_the_trained_model_through_the_numpy_runtime(self, server):
        _, body = post(server, {"cyclone_id": "AMPHAN", "observations": AMPHAN})
        if body["source"] != "trained-sequence-model":
            pytest.skip("No exported weights present; the fallback path is covered elsewhere.")
        assert body["runtime"] == "numpy"

    def test_the_cone_grows_with_lead_time(self, server):
        _, body = post(server, {"cyclone_id": "AMPHAN", "observations": AMPHAN})
        radii = [waypoint["uncertainty_radius_km"] for waypoint in body["waypoints"]]
        assert radii == sorted(radii)

    def test_a_single_observation_is_accepted_and_declared(self, server):
        status, body = post(server, {"cyclone_id": "SOLO", "observations": [AMPHAN[-1]]})
        assert status == 200
        assert "padded_input_steps" in body["inputs_imputed"]


class TestValidation:
    def test_an_empty_observation_list_is_refused(self, server):
        status, body = post(server, {"cyclone_id": "X", "observations": []})
        assert status == 422
        assert "observations" in body["detail"]

    def test_a_latitude_out_of_range_is_refused(self, server):
        status, body = post(server, {"cyclone_id": "X", "observations": [{"lat": 991.0, "lon": 88.0}]})
        assert status == 422
        assert "lat" in body["detail"]

    def test_a_longitude_out_of_range_is_refused(self, server):
        status, _ = post(server, {"cyclone_id": "X", "observations": [{"lat": 15.0, "lon": 400.0}]})
        assert status == 422

    def test_an_impossible_pressure_is_refused(self, server):
        status, _ = post(
            server, {"cyclone_id": "X", "observations": [{"lat": 15.0, "lon": 88.0, "pressure_hpa": 1500}]}
        )
        assert status == 422

    def test_a_missing_cyclone_id_is_refused(self, server):
        status, body = post(server, {"observations": AMPHAN})
        assert status == 422
        assert "cyclone_id" in body["detail"]

    def test_a_position_with_no_intensity_is_refused_by_the_engine(self, server):
        status, body = post(server, {"cyclone_id": "X", "observations": [{"lat": 15.0, "lon": 88.0}]})
        assert status == 422
        assert "wind_kt" in body["detail"] or "pressure" in body["detail"]

    def test_a_non_numeric_latitude_is_refused(self, server):
        status, _ = post(server, {"cyclone_id": "X", "observations": [{"lat": "north", "lon": 88.0}]})
        assert status == 422

    def test_a_json_array_instead_of_an_object_is_refused(self, server):
        status, _ = post(server, [1, 2, 3])
        assert status == 422


class TestRequestHandling:
    def test_malformed_json_is_a_400_not_a_crash(self, server):
        status, body = post(server, None, raw=b"{not json")
        assert status == 400
        assert "JSON" in body["detail"]

    def test_an_empty_body_is_refused(self, server):
        status, _ = post(server, None, raw=b"")
        assert status == 400

    def test_an_oversized_body_is_refused_before_parsing(self, server):
        huge = json.dumps(
            {"cyclone_id": "HUGE", "observations": [{"lat": 12.5, "lon": 86.4, "wind_kt": 120}] * 20_000}
        ).encode("utf-8")
        status, body = post(server, None, raw=huge)
        assert status == 413
        assert "too large" in body["detail"]

    def test_too_many_observations_is_refused(self, server):
        many = [{"lat": 12.5, "lon": 86.4, "wind_kt": 120} for _ in range(500)]
        status, body = post(server, {"cyclone_id": "MANY", "observations": many})
        assert status == 422
        assert "at most" in body["detail"]

    def test_a_get_request_is_told_to_use_post(self, server):
        request = urllib.request.Request(server, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        assert status == 405
