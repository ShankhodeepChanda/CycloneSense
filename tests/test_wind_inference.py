from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from data.build_dataset import category
from ml.wind_inference import (
    NonFiniteModelOutputError,
    load_wind_model,
    predict_wind,
    separate_wind_predictions,
)


class FixedOutputModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def forward(self, images):
        return self.outputs[: len(images)].reshape(-1, 1)


def test_positive_negative_and_zero_keep_raw_and_separate_display_predictions():
    model = FixedOutputModel([0.37, -0.05, 0.0])
    model.train()
    result = predict_wind(model, torch.zeros((3, 1)))

    assert not model.training
    assert np.allclose(result.normalized_model_output, [0.37, -0.05, 0.0])
    assert np.allclose(result.raw_prediction_kt, [37.0, -5.0, 0.0])
    assert np.allclose(result.application_prediction_kt, [37.0, 0.0, 0.0])
    assert result.raw_prediction_kt[1] == pytest.approx(-5.0)
    assert result.application_prediction_kt[1] == 0.0
    assert not result.raw_prediction_kt.flags.writeable
    assert category(result.raw_prediction_kt[1]) == category(result.application_prediction_kt[1]) == "TD"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outputs_raise_without_zero_fallback(value):
    with pytest.raises(NonFiniteModelOutputError, match="non-finite"):
        separate_wind_predictions([value])


def test_checkpoint_loading_is_strict_and_output_still_scales_by_100():
    class TinyWindModel(nn.Module):
        def __init__(self, num_classes=1, pretrained=False):
            super().__init__()
            assert num_classes == 1
            assert pretrained is False
            self.linear = nn.Linear(1, 1, bias=False)

        def forward(self, images):
            return self.linear(images)

    checkpoint = {
        "architecture": "resnet18",
        "channel": "ir",
        "target": "wind_kt/100",
        "model_state_dict": {"linear.weight": torch.tensor([[0.25]])},
    }
    with patch("ml.wind_inference.CycloneImageClassifier", TinyWindModel):
        with patch("ml.wind_inference.torch.load", return_value=checkpoint) as load:
            model, device = load_wind_model("mock-model-a.pt", device="cpu")
    load.assert_called_once()
    assert load.call_args.kwargs["weights_only"] is True
    assert model.training is False
    assert device == torch.device("cpu")

    result = predict_wind(model, torch.tensor([[1.0]]), device=device)
    assert result.normalized_model_output.tolist() == pytest.approx([0.25])
    assert result.raw_prediction_kt.tolist() == pytest.approx([25.0])
    assert result.application_prediction_kt.tolist() == pytest.approx([25.0])
