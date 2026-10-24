"""The torch-free serving path must reproduce PyTorch exactly, not approximately.

Serving is where the model meets users, so a divergence here would be a wrong forecast
rather than a wrong test. These tests pin the numpy forward pass against the real one.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch

from ml.numpy_forecaster import (
    NumpyForecaster,
    export_numpy_weights,
    gelu,
    layer_norm,
)
from ml.sequence_dataset import FEATURES, INPUT_STEPS, TARGET_LEADS, normalise
from ml.trajectory_model import BayOfBengalCycloneForecaster, save_forecaster_checkpoint

HORIZON = len(TARGET_LEADS)


@pytest.fixture
def torch_checkpoint(tmp_path):
    """A small real checkpoint, saved the way training saves one."""
    torch.manual_seed(7)
    model = BayOfBengalCycloneForecaster(use_vision=False).eval()
    path = tmp_path / "tabular.pt"
    save_forecaster_checkpoint(
        model,
        path,
        lead_track_rmse_km=[32.2, 65.5, 106.5, 153.7, 197.5, 237.7, 285.9, 337.2],
        metrics={"val_track_mae_km": 155.0},
        train_storms=["A", "B"],
        val_storms=["C"],
    )
    return path, model


class TestActivations:
    def test_gelu_matches_torch(self):
        values = np.linspace(-6.0, 6.0, 41)
        expected = torch.nn.functional.gelu(torch.tensor(values)).numpy()
        np.testing.assert_allclose(gelu(values), expected, atol=1e-12)

    def test_layer_norm_matches_torch(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(3, 5, 8))
        weight = rng.normal(size=8)
        bias = rng.normal(size=8)
        expected = torch.nn.functional.layer_norm(
            torch.tensor(values), (8,), torch.tensor(weight), torch.tensor(bias), eps=1e-5
        ).numpy()
        np.testing.assert_allclose(layer_norm(values, weight, bias), expected, atol=1e-12)

    def test_layer_norm_uses_the_biased_variance_torch_uses(self):
        """An unbiased variance would be subtly wrong and hard to notice."""
        values = np.array([[1.0, 2.0, 3.0, 10.0]])
        weight, bias = np.ones(4), np.zeros(4)
        expected = torch.nn.functional.layer_norm(
            torch.tensor(values), (4,), torch.ones(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
        ).numpy()
        np.testing.assert_allclose(layer_norm(values, weight, bias), expected, atol=1e-12)


class TestExportAndLoad:
    def test_export_writes_an_npz_beside_the_checkpoint(self, torch_checkpoint, tmp_path):
        path, _ = torch_checkpoint
        exported = export_numpy_weights(path, tmp_path / "tabular.npz")
        assert exported.exists()
        assert exported.stat().st_size > 0

    def test_the_exported_metadata_carries_what_serving_needs(self, torch_checkpoint, tmp_path):
        path, _ = torch_checkpoint
        export_numpy_weights(path, tmp_path / "m.npz")
        _, meta = NumpyForecaster.load(tmp_path / "m.npz")
        assert meta["feature_names"] == FEATURES
        assert meta["target_leads"] == TARGET_LEADS
        assert meta["input_steps"] == INPUT_STEPS
        assert meta["use_vision"] is False
        assert meta["lead_track_rmse_km"][3] == pytest.approx(153.7)
        assert meta["metrics"]["val_track_mae_km"] == pytest.approx(155.0)
        assert meta["train_storms"] == ["A", "B"]

    def test_a_vision_checkpoint_is_refused_with_a_clear_reason(self, tmp_path):
        model = BayOfBengalCycloneForecaster(use_vision=True, pretrained=False).eval()
        path = tmp_path / "vision.pt"
        save_forecaster_checkpoint(model, path, lead_track_rmse_km=[1.0] * HORIZON, metrics={})
        with pytest.raises(ValueError, match="vision"):
            export_numpy_weights(path, tmp_path / "vision.npz")


class TestNumericalAgreementWithTorch:
    def _both(self, torch_checkpoint, tmp_path, tabular):
        path, model = torch_checkpoint
        export_numpy_weights(path, tmp_path / "x.npz")
        forecaster, _ = NumpyForecaster.load(tmp_path / "x.npz")
        with torch.inference_mode():
            expected = model(torch.tensor(tabular, dtype=torch.float32))
        actual = forecaster.forward(tabular)
        return actual, expected

    def test_random_input_agrees_to_float32_precision(self, torch_checkpoint, tmp_path):
        rng = np.random.default_rng(3)
        tabular = rng.normal(size=(5, INPUT_STEPS, len(FEATURES))).astype(np.float32)
        actual, expected = self._both(torch_checkpoint, tmp_path, tabular)
        np.testing.assert_allclose(actual["delta"], expected["delta"].numpy(), atol=2e-5)
        np.testing.assert_allclose(actual["intensity"], expected["intensity"].numpy(), atol=2e-5)

    def test_real_amphan_input_agrees(self, torch_checkpoint, tmp_path):
        raw = np.array(
            [
                [11.2, 86.1, 0.3, 0.0, 60.0, 982.0, 603.0],
                [11.5, 86.2, 0.3, 0.1, 70.0, 978.0, 630.0],
                [11.9, 86.2, 0.4, 0.0, 80.0, 970.0, 654.0],
                [12.5, 86.4, 0.6, 0.2, 120.0, 932.0, 636.0],
            ]
        )
        tabular = normalise(raw).astype(np.float32)[None, ...]
        actual, expected = self._both(torch_checkpoint, tmp_path, tabular)
        np.testing.assert_allclose(actual["delta"], expected["delta"].numpy(), atol=2e-5)

    def test_extreme_input_does_not_diverge(self, torch_checkpoint, tmp_path):
        """A dateline storm once produced a feature of -599; the paths must still agree."""
        tabular = np.full((1, INPUT_STEPS, len(FEATURES)), -30.0, dtype=np.float32)
        actual, expected = self._both(torch_checkpoint, tmp_path, tabular)
        np.testing.assert_allclose(actual["delta"], expected["delta"].numpy(), atol=2e-5)

    def test_output_shapes_match_the_documented_contract(self, torch_checkpoint, tmp_path):
        rng = np.random.default_rng(5)
        tabular = rng.normal(size=(2, INPUT_STEPS, len(FEATURES))).astype(np.float32)
        actual, _ = self._both(torch_checkpoint, tmp_path, tabular)
        assert actual["delta"].shape == (2, HORIZON, 2)
        assert actual["intensity"].shape == (2, HORIZON, 2)

    def test_the_shipped_checkpoint_agrees_too(self, tmp_path):
        """Not just a fresh random model: the weights actually being served."""
        from pathlib import Path

        shipped = Path("models/bob_trajectory_gru.pt")
        if not shipped.is_file():
            pytest.skip("The shipped checkpoint is not present on this machine.")
        from ml.trajectory_model import load_forecaster

        model, _ = load_forecaster(shipped)
        export_numpy_weights(shipped, tmp_path / "shipped.npz")
        forecaster, _ = NumpyForecaster.load(tmp_path / "shipped.npz")
        rng = np.random.default_rng(11)
        tabular = rng.normal(size=(4, INPUT_STEPS, len(FEATURES))).astype(np.float32)
        with torch.inference_mode():
            expected = model(torch.tensor(tabular))
        actual = forecaster.forward(tabular)
        np.testing.assert_allclose(actual["delta"], expected["delta"].numpy(), atol=2e-5)
        np.testing.assert_allclose(actual["intensity"], expected["intensity"].numpy(), atol=2e-5)


class TestServingNeedsNoTorch:
    def test_importing_the_serving_module_does_not_pull_in_torch(self):
        """Vercel's 250 MB function limit is the whole reason this module exists."""
        code = (
            "import sys; import ml.numpy_forecaster; "
            "assert 'torch' not in sys.modules, 'numpy_forecaster imported torch'; "
            "print('clean')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout

    def test_running_a_forecast_does_not_pull_in_torch(self, tmp_path):
        from pathlib import Path

        shipped = Path("models/bob_trajectory_gru.npz")
        if not shipped.is_file():
            pytest.skip("Export the shipped checkpoint first.")
        code = (
            "import sys, numpy as np\n"
            "from ml.numpy_forecaster import NumpyForecaster\n"
            f"f, meta = NumpyForecaster.load({str(shipped)!r})\n"
            "out = f.forward(np.zeros((1, 4, 7), dtype=np.float32))\n"
            "assert out['delta'].shape == (1, 8, 2)\n"
            "assert 'torch' not in sys.modules, 'forecast imported torch'\n"
            "print('clean')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout


HEAVY_MODULES = ("torch", "torchvision", "pandas", "scipy", "sklearn")

# Built as a list of lines and joined at use, so no escape survives two layers of quoting.
_HEAVY_CHECK = [
    "import sys",
    f"heavy = [n for n in {HEAVY_MODULES!r} if n in sys.modules]",
    "print('HEAVY:' + ','.join(heavy) if heavy else 'clean')",
]


def _run(lines: list[str]) -> str:
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)], capture_output=True, text=True, timeout=240
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class TestServingBundleStaysSmall:
    """Vercel's function limit is 250 MB unzipped.

    torch is 534 MB and pandas 67 MB, so the serving path must import neither. numpy
    alone is 34 MB, which is what makes a single-platform deployment possible at all.
    """

    def test_the_trajectory_engine_imports_without_torch_or_pandas(self):
        assert _run(["import app.trajectory"] + _HEAVY_CHECK) == "clean"

    def test_the_numpy_forecaster_imports_without_torch_or_pandas(self):
        assert _run(["import ml.numpy_forecaster"] + _HEAVY_CHECK) == "clean"

    def test_a_full_forecast_runs_without_torch_or_pandas(self):
        lines = [
            "from app.trajectory import TrajectoryEngine",
            "fixes = [",
            "    {'lat': 12.5, 'lon': 86.4, 'wind_kt': 120, 'pressure_hpa': 932},",
            "    {'lat': 13.2, 'lon': 86.4, 'wind_kt': 130, 'pressure_hpa': 919},",
            "    {'lat': 13.4, 'lon': 86.2, 'wind_kt': 140, 'pressure_hpa': 911},",
            "    {'lat': 14.1, 'lon': 86.4, 'wind_kt': 145, 'pressure_hpa': 901},",
            "]",
            "result = TrajectoryEngine().forecast('AMPHAN', fixes)",
            "assert len(result['waypoints']) == 8, result",
            "assert result['source'] == 'trained-sequence-model', result['source']",
        ] + _HEAVY_CHECK
        assert _run(lines) == "clean"
