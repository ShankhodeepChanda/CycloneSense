"""Torch-free inference for the trained trajectory forecaster.

PyTorch is 534 MB installed, which does not fit a 250 MB serverless function, so the
deployed API cannot import it. The trained model is small and entirely standard -- an
MLP, a LayerNorm, one GRU layer and two linear heads -- so serving it needs nothing but
numpy once the weights are out of the checkpoint.

The split of duties is deliberate:

* :func:`export_numpy_weights` runs **with** torch, on a developer machine or in CI, and
  writes a ``.npz`` holding the weights plus the metadata the API depends on.
* :class:`NumpyForecaster` runs **without** torch and reproduces the forward pass.

``tests/test_numpy_forecaster.py`` pins this path against PyTorch on random, real and
extreme inputs, including the checkpoint actually being served. That test is the only
reason to trust a hand-written forward pass: serving is where the model meets users, so
a divergence here would be a wrong forecast rather than a failing test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: Must match torch.nn.LayerNorm's default.
LAYER_NORM_EPS = 1e-5

_METADATA_KEY = "__metadata__"

# Vectorised exact erf. torch.nn.functional.gelu's default is the erf form, not the tanh
# approximation, and the two differ by about 1e-3 around |x| = 2 -- enough to show up in
# a forecast. scipy would provide erf but costs over 100 MB, so this uses math.erf.
_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    values = np.asarray(x, dtype=np.float64)
    return 0.5 * values * (1.0 + _erf(values / math.sqrt(2.0)))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """LayerNorm over the last axis, using the biased variance torch uses."""
    values = np.asarray(x, dtype=np.float64)
    mean = values.mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)  # biased, matching torch
    return (values - mean) / np.sqrt(variance + eps) * weight + bias


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    values = np.asarray(x, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """torch.nn.Linear: x @ weight.T + bias."""
    out = np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    return out if bias is None else out + bias


def export_numpy_weights(checkpoint_path: str | Path, out_path: str | Path) -> Path:
    """Convert a trained checkpoint into a torch-free ``.npz``. Requires torch."""
    from ml.trajectory_model import load_forecaster  # imported here: export-time only

    model, meta = load_forecaster(checkpoint_path)
    if meta.get("use_vision"):
        raise ValueError(
            f"{checkpoint_path} was trained with the vision branch. The numpy serving path "
            "implements the tabular model only, because a ResNet-18 forward pass in numpy "
            "would be both slow and a second thing to keep in step with torch."
        )

    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    arrays[_METADATA_KEY] = np.array(json.dumps(meta, default=str))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    return target


class NumpyForecaster:
    """The trained forecaster's forward pass, in numpy.

    Mirrors :class:`ml.trajectory_model.BayOfBengalCycloneForecaster` with
    ``use_vision=False``: tabular MLP -> fuse -> GRU -> trajectory and intensity heads.
    """

    def __init__(self, weights: dict[str, np.ndarray], metadata: dict[str, Any]):
        self.weights = {name: np.asarray(value, dtype=np.float64) for name, value in weights.items()}
        self.metadata = metadata
        self.horizon = int(metadata.get("horizon") or len(metadata["target_leads"]))
        missing = [name for name in self._required_keys() if name not in self.weights]
        if missing:
            raise ValueError(f"Exported weights are missing {missing}.")

    @staticmethod
    def _required_keys() -> tuple[str, ...]:
        return (
            "tabular.0.weight", "tabular.0.bias",
            "tabular.2.weight", "tabular.2.bias",
            "tabular.3.weight", "tabular.3.bias",
            "fuse.0.weight", "fuse.0.bias",
            "sequence.weight_ih_l0", "sequence.weight_hh_l0",
            "sequence.bias_ih_l0", "sequence.bias_hh_l0",
            "trajectory_head.0.weight", "trajectory_head.0.bias",
            "trajectory_head.2.weight", "trajectory_head.2.bias",
            "intensity_head.0.weight", "intensity_head.0.bias",
            "intensity_head.2.weight", "intensity_head.2.bias",
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple["NumpyForecaster", dict[str, Any]]:
        """Load exported weights and their metadata. Never imports torch."""
        with np.load(Path(path), allow_pickle=False) as handle:
            if _METADATA_KEY not in handle.files:
                raise ValueError(f"{path} carries no metadata; re-export it with export_numpy_weights.")
            metadata = json.loads(str(handle[_METADATA_KEY]))
            weights = {name: handle[name] for name in handle.files if name != _METADATA_KEY}
        return cls(weights, metadata), metadata

    def _gru_step(self, x: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        """One PyTorch GRU step. Gate order in the packed weights is r, z, n."""
        weight_ih = self.weights["sequence.weight_ih_l0"]
        weight_hh = self.weights["sequence.weight_hh_l0"]
        bias_ih = self.weights["sequence.bias_ih_l0"]
        bias_hh = self.weights["sequence.bias_hh_l0"]
        size = weight_hh.shape[1]

        gates_x = x @ weight_ih.T + bias_ih
        gates_h = hidden @ weight_hh.T + bias_hh

        reset = sigmoid(gates_x[:, :size] + gates_h[:, :size])
        update = sigmoid(gates_x[:, size : 2 * size] + gates_h[:, size : 2 * size])
        # The reset gate multiplies the hidden contribution only, after its bias.
        candidate = np.tanh(gates_x[:, 2 * size :] + reset * gates_h[:, 2 * size :])
        return (1.0 - update) * candidate + update * hidden

    def forward(self, tabular: np.ndarray) -> dict[str, np.ndarray]:
        """Normalised features (batch, steps, features) in; scaled predictions out.

        Returns ``delta`` and ``intensity``, both (batch, horizon, 2), in the same scaled
        frame the torch model emits; callers denormalise exactly as before.
        """
        values = np.asarray(tabular, dtype=np.float64)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3:
            raise ValueError(f"Expected (batch, steps, features); got shape {values.shape}.")
        batch, steps, _ = values.shape

        embedded = linear(values, self.weights["tabular.0.weight"], self.weights["tabular.0.bias"])
        embedded = gelu(embedded)
        embedded = linear(embedded, self.weights["tabular.2.weight"], self.weights["tabular.2.bias"])
        embedded = layer_norm(embedded, self.weights["tabular.3.weight"], self.weights["tabular.3.bias"])

        fused = gelu(linear(embedded, self.weights["fuse.0.weight"], self.weights["fuse.0.bias"]))

        hidden_size = self.weights["sequence.weight_hh_l0"].shape[1]
        hidden = np.zeros((batch, hidden_size), dtype=np.float64)
        for step in range(steps):
            hidden = self._gru_step(fused[:, step, :], hidden)
        # Dropout is identity in eval mode, so the final hidden state passes straight on.

        def head(prefix: str) -> np.ndarray:
            out = gelu(linear(hidden, self.weights[f"{prefix}.0.weight"], self.weights[f"{prefix}.0.bias"]))
            out = linear(out, self.weights[f"{prefix}.2.weight"], self.weights[f"{prefix}.2.bias"])
            return out.reshape(batch, self.horizon, 2)

        return {"delta": head("trajectory_head"), "intensity": head("intensity_head")}


def main(argv: list[str] | None = None) -> int:
    """CLI: export a trained checkpoint for torch-free serving."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default="models/bob_trajectory_gru.pt")
    parser.add_argument("--out", default=None, help="Defaults to the checkpoint path with .npz")
    args = parser.parse_args(argv)

    source = Path(args.checkpoint)
    target = Path(args.out) if args.out else source.with_suffix(".npz")
    written = export_numpy_weights(source, target)
    forecaster, meta = NumpyForecaster.load(written)
    print(f"exported {written} ({written.stat().st_size / 1024:.0f} KB)")
    print(f"  leads: {meta['target_leads']}")
    print(f"  cone radii (km): {[round(v, 1) for v in meta.get('lead_track_rmse_km', [])]}")
    print(f"  parameters: {sum(v.size for v in forecaster.weights.values()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
