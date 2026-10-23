"""Safe inference boundary for the wind-speed regression checkpoint.

Raw knots remain available for scientific evaluation. Only the separately named
application prediction is constrained to be nonnegative.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.model import CycloneImageClassifier

PRODUCTION_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "multiyear45_resnet18_ir_weighted.pt"
)
TARGET_SCALE_KT = 100.0


class NonFiniteModelOutputError(ValueError):
    """Raised when a model emits NaN or infinity; callers must surface the failure."""


@dataclass(frozen=True)
class WindPredictionBatch:
    normalized_model_output: np.ndarray
    raw_prediction_kt: np.ndarray
    application_prediction_kt: np.ndarray


def separate_wind_predictions(normalized_output: Any) -> WindPredictionBatch:
    """Convert wind_kt/100 outputs into raw knots and a nonnegative display value.

    Non-finite output rejects the entire batch. It is never replaced with zero.
    """
    normalized = np.array(normalized_output, dtype=np.float64, copy=True)
    if not np.isfinite(normalized).all():
        raise NonFiniteModelOutputError(
            "Wind model returned a non-finite normalized output; no prediction was produced."
        )
    raw = normalized * TARGET_SCALE_KT
    if not np.isfinite(raw).all():
        raise NonFiniteModelOutputError(
            "Wind model output became non-finite during conversion to knots."
        )
    application = np.maximum(raw, 0.0)
    for values in (normalized, raw, application):
        values.setflags(write=False)
    return WindPredictionBatch(
        normalized_model_output=normalized,
        raw_prediction_kt=raw,
        application_prediction_kt=application,
    )


def predict_wind(model: nn.Module, images: torch.Tensor, device: str | torch.device | None = None) -> WindPredictionBatch:
    """Run deterministic inference and return raw plus application-facing values."""
    model.eval()
    inputs = images.to(device) if device is not None else images
    with torch.inference_mode():
        output = model(inputs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("Wind model must return a torch.Tensor.")
    if output.ndim < 1 or output.shape[-1] != 1:
        raise ValueError(
            f"Wind model must return one normalized value per image; got shape {tuple(output.shape)}."
        )
    normalized = output.squeeze(-1).detach().cpu().numpy()
    return separate_wind_predictions(normalized)


def load_wind_model(
    checkpoint_path: str | Path = PRODUCTION_CHECKPOINT,
    device: str | torch.device | None = None,
) -> tuple[CycloneImageClassifier, torch.device]:
    """Load a compatible IR wind checkpoint strictly, defaulting to production Model A."""
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location=selected, weights_only=True)
    expected = {
        "architecture": "resnet18",
        "channel": "ir",
        "target": "wind_kt/100",
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unexpected wind checkpoint metadata in {path}: {mismatches}")
    model = CycloneImageClassifier(num_classes=1, pretrained=False).to(selected)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, selected
