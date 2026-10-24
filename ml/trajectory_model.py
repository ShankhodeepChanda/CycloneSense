"""Multi-modal spatio-temporal forecaster for Bay of Bengal cyclones.

Three branches feed one recurrent core:

* a ResNet-18 vision trunk turns each timestep's satellite chip into a 128-d vector,
  with early layers frozen and a learned embedding standing in for steps that have no
  chip (most North Indian Ocean steps do not);
* a small MLP embeds the seven per-step metrics (position, recent motion, wind, central
  pressure, distance to land) into 64 d;
* the fused sequence runs through a GRU, and two heads read its final state: cumulative
  track offsets and intensity at each of the eight forecast leads.

Outputs are in the scaled frame defined by ``ml.sequence_dataset``; callers denormalise
them. Keeping the model in that frame is what lets training, the checkpoint and the API
agree on units without passing scalers around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ml.sequence_dataset import (
    DELTA_SCALE_DEG,
    FEATURES,
    INPUT_STEPS,
    NORM,
    TARGET_LEADS,
    TARGET_NORM,
)

VISION_DIM = 128
TABULAR_DIM = 64
HIDDEN_DIM = 128

#: Vision trunk blocks kept frozen. Fine-tuning only layer4 keeps the trainable model
#: small enough for the data this basin actually has.
FROZEN_VISION_BLOCKS = ("conv1", "bn1", "layer1", "layer2", "layer3")


class VisionEncoder(nn.Module):
    """Per-timestep ResNet-18 embedding with a learned stand-in for absent chips."""

    def __init__(self, out_dim: int = VISION_DIM, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.DEFAULT
            except Exception:  # pragma: no cover - offline machines simply train from scratch
                weights = None
        trunk = resnet18(weights=weights)
        self.feature_dim = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.project = nn.Sequential(nn.Linear(self.feature_dim, out_dim), nn.GELU())
        self.missing = nn.Parameter(torch.zeros(out_dim))
        for name in FROZEN_VISION_BLOCKS:
            for parameter in getattr(self.trunk, name).parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor | None, mask: torch.Tensor | None, batch: int, steps: int) -> torch.Tensor:
        missing = self.missing.expand(batch, steps, -1)
        if image is None:
            return missing
        if mask is None:
            mask = image.new_ones(batch, steps)
        flat = image.reshape(batch * steps, *image.shape[2:])
        present = mask.reshape(-1) > 0.5
        embedded = missing.reshape(batch * steps, -1).clone()
        if present.any():
            # Only the steps that really have a chip go through the trunk, so a fully
            # masked batch costs nothing and cannot pick up gradients from zero pixels.
            encoded = self.project(self.trunk(flat[present]))
            embedded = embedded.index_copy(0, present.nonzero(as_tuple=True)[0], encoded)
        return embedded.reshape(batch, steps, -1)


class BayOfBengalCycloneForecaster(nn.Module):
    """Fuses satellite imagery with best-track metrics to forecast track and intensity."""

    def __init__(
        self,
        use_vision: bool = False,
        pretrained: bool = False,
        n_features: int = len(FEATURES),
        input_steps: int = INPUT_STEPS,
        horizon: int = len(TARGET_LEADS),
        hidden_dim: int = HIDDEN_DIM,
        vision_dim: int = VISION_DIM,
        tabular_dim: int = TABULAR_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_vision = bool(use_vision)
        self.n_features = n_features
        self.input_steps = input_steps
        self.horizon = horizon

        self.tabular = nn.Sequential(
            nn.Linear(n_features, tabular_dim),
            nn.GELU(),
            nn.Linear(tabular_dim, tabular_dim),
            nn.LayerNorm(tabular_dim),
        )
        self.vision = VisionEncoder(vision_dim, pretrained=pretrained) if self.use_vision else None

        fused_dim = tabular_dim + (vision_dim if self.use_vision else 0)
        self.fuse = nn.Sequential(nn.Linear(fused_dim, hidden_dim), nn.GELU())
        self.sequence = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.trajectory_head = self._head(hidden_dim, horizon * 2)
        self.intensity_head = self._head(hidden_dim, horizon * 2)

    @staticmethod
    def _head(hidden_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        tabular: torch.Tensor,
        image: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps, _ = tabular.shape
        embedded = self.tabular(tabular)
        if self.use_vision:
            assert self.vision is not None
            visual = self.vision(image, image_mask, batch, steps)
            embedded = torch.cat([embedded, visual], dim=-1)
        _, last = self.sequence(self.fuse(embedded))
        state = self.dropout(last[-1])
        return {
            "delta": self.trajectory_head(state).view(batch, self.horizon, 2),
            "intensity": self.intensity_head(state).view(batch, self.horizon, 2),
        }


def forecaster_metadata(model: BayOfBengalCycloneForecaster) -> dict[str, Any]:
    """The frame a checkpoint must carry for inference to reproduce training."""
    return {
        "architecture": "bob-resnet18-gru" if model.use_vision else "bob-tabular-gru",
        "use_vision": model.use_vision,
        "feature_names": list(FEATURES),
        "target_leads": list(TARGET_LEADS),
        "input_steps": model.input_steps,
        "horizon": model.horizon,
        "norm": {name: list(values) for name, values in NORM.items()},
        "target_norm": {name: list(values) for name, values in TARGET_NORM.items()},
        "delta_scale_deg": DELTA_SCALE_DEG,
    }


def save_forecaster_checkpoint(
    model: BayOfBengalCycloneForecaster,
    path: str | Path,
    lead_track_rmse_km: list[float],
    metrics: dict[str, Any],
    **extra: Any,
) -> Path:
    """Write weights plus the full frame needed to run them, including per-lead error.

    ``lead_track_rmse_km`` is what the API turns into cone radii, so it travels with the
    weights rather than being guessed at serving time.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "lead_track_rmse_km": [float(value) for value in lead_track_rmse_km],
        "metrics": metrics,
        **forecaster_metadata(model),
        **extra,
    }
    torch.save(payload, target)
    return target


def load_forecaster(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[BayOfBengalCycloneForecaster, dict[str, Any]]:
    """Load a forecaster and its metadata, rejecting a checkpoint from a different frame."""
    selected = torch.device(device or "cpu")
    checkpoint = torch.load(Path(path), map_location=selected, weights_only=False)
    for key in ("model_state_dict", "use_vision", "feature_names", "target_leads", "input_steps"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint {path} is missing the required key {key!r}.")
    if checkpoint["feature_names"] != list(FEATURES):
        raise ValueError(
            f"Checkpoint {path} was trained on features {checkpoint['feature_names']}, "
            f"but this build expects {list(FEATURES)}."
        )
    model = BayOfBengalCycloneForecaster(
        use_vision=bool(checkpoint["use_vision"]),
        pretrained=False,
        n_features=len(checkpoint["feature_names"]),
        input_steps=int(checkpoint["input_steps"]),
        horizon=int(checkpoint.get("horizon", len(checkpoint["target_leads"]))),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(selected).eval()
    meta = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    meta.setdefault("metrics", {})
    meta.setdefault("lead_track_rmse_km", [])
    return model, meta
