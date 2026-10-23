"""Tests for trajectory training: distance maths, the loss path, and the checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ml.sequence_dataset import (
    FEATURES,
    INPUT_STEPS,
    TARGET_LEADS,
    CycloneSequenceDataset,
    build_sequence_windows,
)
from ml.trajectory_model import BayOfBengalCycloneForecaster, load_forecaster
from ml.train_trajectory import (
    TrainConfig,
    compute_losses,
    evaluate_per_lead,
    haversine_km,
    haversine_km_torch,
    linear_baseline_per_lead,
    origin_from_tabular,
    run_epoch,
    train,
)

EARTH_DEGREE_KM = 111.19  # one degree of latitude on a 6371 km sphere


def synthetic_windows(rows: int = 40, sid: str = "TESTSTORM"):
    times = pd.date_range("2020-05-10", periods=rows, freq="6h")
    frame = pd.DataFrame(
        {
            "sid": sid,
            "name": "TESTSTORM",
            "basin": "NI",
            "time": times,
            "lat": [11.0 + 0.4 * i for i in range(rows)],
            "lon": [87.0 + 0.2 * i for i in range(rows)],
            "wind_kt": [35.0 + 3.0 * i for i in range(rows)],
            "pres_hpa": [1002.0 - 2.5 * i for i in range(rows)],
            "dist2land_km": [700.0 - 15.0 * i for i in range(rows)],
            "dlat": [np.nan] + [0.4] * (rows - 1),
            "dlon": [np.nan] + [0.2] * (rows - 1),
        }
    )
    return build_sequence_windows(frame)


class TestHaversine:
    def test_one_degree_of_latitude_is_about_111_km(self):
        assert abs(haversine_km(0.0, 0.0, 1.0, 0.0) - EARTH_DEGREE_KM) < 1.0

    def test_one_degree_of_longitude_shrinks_towards_the_pole(self):
        at_equator = haversine_km(0.0, 0.0, 0.0, 1.0)
        at_sixty = haversine_km(60.0, 0.0, 60.0, 1.0)
        assert abs(at_equator - EARTH_DEGREE_KM) < 1.0
        assert abs(at_sixty - EARTH_DEGREE_KM * 0.5) < 1.0

    def test_identical_points_are_zero_km_apart(self):
        assert haversine_km(17.5, 86.2, 17.5, 86.2) == pytest.approx(0.0, abs=1e-9)

    def test_known_bay_of_bengal_leg(self):
        """Visakhapatnam to Kolkata, about 740 km great-circle."""
        distance = haversine_km(17.69, 83.22, 22.57, 88.36)
        assert 700.0 < distance < 780.0

    def test_vectorised_over_arrays(self):
        lat = np.array([0.0, 0.0]), np.array([1.0, 2.0])
        distance = haversine_km(lat[0], np.zeros(2), lat[1], np.zeros(2))
        assert distance.shape == (2,)
        assert distance[1] == pytest.approx(2 * distance[0], rel=1e-3)

    def test_torch_haversine_matches_the_numpy_one(self):
        numpy_km = haversine_km(12.0, 85.0, 15.5, 88.5)
        torch_km = haversine_km_torch(
            torch.tensor([12.0]), torch.tensor([85.0]), torch.tensor([15.5]), torch.tensor([88.5])
        )
        assert float(torch_km[0]) == pytest.approx(numpy_km, rel=1e-5)

    def test_torch_haversine_is_differentiable(self):
        lat = torch.tensor([15.0], requires_grad=True)
        haversine_km_torch(lat, torch.tensor([85.0]), torch.tensor([17.0]), torch.tensor([86.0])).backward()
        assert lat.grad is not None and torch.isfinite(lat.grad).all()


class TestOrigin:
    def test_origin_is_the_raw_position_at_t0(self):
        windows = synthetic_windows()
        item = CycloneSequenceDataset(windows, with_images=False)[0]
        lat0, lon0 = origin_from_tabular(item["tabular"].unsqueeze(0))
        expected_lat = windows[0].features[INPUT_STEPS - 1, FEATURES.index("lat")]
        expected_lon = windows[0].features[INPUT_STEPS - 1, FEATURES.index("lon")]
        assert float(lat0[0]) == pytest.approx(expected_lat, abs=1e-3)
        assert float(lon0[0]) == pytest.approx(expected_lon, abs=1e-3)


class TestLossPath:
    def _batch(self, batch_size: int = 8):
        dataset = CycloneSequenceDataset(synthetic_windows(), with_images=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return next(iter(loader))

    def test_losses_are_finite_scalars_with_a_track_error_in_km(self):
        model = BayOfBengalCycloneForecaster(use_vision=False)
        losses = compute_losses(model(self._batch()["tabular"]), self._batch())
        for key in ("total", "track_km", "intensity"):
            assert losses[key].ndim == 0
            assert torch.isfinite(losses[key])
        assert losses["track_km"] > 0

    def test_a_perfect_prediction_has_near_zero_track_error(self):
        batch = self._batch()
        from ml.sequence_dataset import DELTA_SCALE_DEG, normalise_intensity

        perfect = {
            "delta": batch["target_delta"] / DELTA_SCALE_DEG,
            "intensity": torch.tensor(
                normalise_intensity(batch["target_intensity"].numpy()), dtype=torch.float32
            ),
        }
        losses = compute_losses(perfect, batch)
        assert float(losses["track_km"]) < 1.0
        assert float(losses["intensity"]) < 1e-6

    def test_training_reduces_the_loss_on_a_single_batch(self):
        torch.manual_seed(0)
        model = BayOfBengalCycloneForecaster(use_vision=False)
        dataset = CycloneSequenceDataset(synthetic_windows(), with_images=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = run_epoch(model, loader, optimiser)["total"]
        for _ in range(40):
            last = run_epoch(model, loader, optimiser)["total"]
        assert last < 0.5 * first

    def test_eval_epoch_leaves_the_weights_untouched(self):
        model = BayOfBengalCycloneForecaster(use_vision=False)
        before = [p.detach().clone() for p in model.parameters()]
        dataset = CycloneSequenceDataset(synthetic_windows(), with_images=False)
        run_epoch(model, torch.utils.data.DataLoader(dataset, batch_size=4), optimiser=None)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


class TestEvaluation:
    def test_per_lead_errors_cover_every_forecast_hour(self):
        model = BayOfBengalCycloneForecaster(use_vision=False).eval()
        report = evaluate_per_lead(model, synthetic_windows())
        assert report["leads"] == TARGET_LEADS
        assert len(report["track_rmse_km"]) == len(TARGET_LEADS)
        assert len(report["track_mae_km"]) == len(TARGET_LEADS)
        assert len(report["wind_mae_kt"]) == len(TARGET_LEADS)
        assert len(report["pressure_mae_hpa"]) == len(TARGET_LEADS)
        assert all(np.isfinite(report["track_rmse_km"]))
        assert all(value >= 0 for value in report["track_rmse_km"])

    def test_the_linear_baseline_grows_with_lead_time_on_a_curving_track(self):
        rows = 40
        times = pd.date_range("2020-05-10", periods=rows, freq="6h")
        angle = np.linspace(0, 1.4, rows)
        frame = pd.DataFrame(
            {
                "sid": "CURVED",
                "name": "CURVED",
                "basin": "NI",
                "time": times,
                "lat": 12.0 + 6.0 * np.sin(angle),
                "lon": 88.0 + 6.0 * (1 - np.cos(angle)),
                "wind_kt": 60.0,
                "pres_hpa": 980.0,
                "dist2land_km": 400.0,
                "dlat": np.nan,
                "dlon": np.nan,
            }
        )
        baseline = linear_baseline_per_lead(build_sequence_windows(frame))
        assert baseline["track_rmse_km"][-1] > baseline["track_rmse_km"][0]

    def test_the_linear_baseline_is_exact_on_a_straight_track(self):
        baseline = linear_baseline_per_lead(synthetic_windows())
        assert max(baseline["track_rmse_km"]) < 1.0


class TestTrainEntryPoint:
    def test_training_writes_a_checkpoint_carrying_the_frame_inference_needs(self, tmp_path):
        checkpoint = tmp_path / "bob.pt"
        metrics = tmp_path / "bob_metrics.json"
        result = train(
            TrainConfig(
                tracks="kaggle_trial/tracks.csv",
                basins=("NI",),
                epochs=2,
                batch_size=32,
                checkpoint=checkpoint,
                metrics_path=metrics,
                seed=0,
            )
        )
        assert checkpoint.exists()
        model, meta = load_forecaster(checkpoint)
        assert meta["use_vision"] is False
        assert meta["feature_names"] == FEATURES
        assert meta["target_leads"] == TARGET_LEADS
        assert meta["input_steps"] == INPUT_STEPS
        assert len(meta["lead_track_rmse_km"]) == len(TARGET_LEADS)
        assert all(value > 0 for value in meta["lead_track_rmse_km"])
        assert "norm" in meta and "val_track_mae_km" in meta["metrics"]
        assert result["val_windows"] > 0
        assert json.loads(metrics.read_text())["metrics"]["val_track_mae_km"] > 0

    def test_training_reports_the_baseline_alongside_the_model(self, tmp_path):
        result = train(
            TrainConfig(
                tracks="kaggle_trial/tracks.csv",
                basins=("NI",),
                epochs=1,
                batch_size=32,
                checkpoint=tmp_path / "b.pt",
                metrics_path=tmp_path / "b.json",
                seed=0,
            )
        )
        assert len(result["baseline"]["track_rmse_km"]) == len(TARGET_LEADS)
        assert len(result["validation"]["track_rmse_km"]) == len(TARGET_LEADS)

    def test_fine_tuning_starts_from_an_existing_checkpoint(self, tmp_path):
        pretrained = tmp_path / "global.pt"
        train(
            TrainConfig(
                tracks="kaggle_trial/tracks.csv",
                basins=None,
                epochs=1,
                batch_size=64,
                checkpoint=pretrained,
                metrics_path=tmp_path / "global.json",
                seed=0,
                max_windows=400,
            )
        )
        tuned = tmp_path / "tuned.pt"
        result = train(
            TrainConfig(
                tracks="kaggle_trial/tracks.csv",
                basins=("NI",),
                epochs=1,
                batch_size=32,
                checkpoint=tuned,
                metrics_path=tmp_path / "tuned.json",
                seed=0,
                init_from=pretrained,
            )
        )
        assert tuned.exists()
        assert result["initialised_from"] == str(pretrained)

    def test_a_basin_with_no_windows_is_reported_not_crashed(self, tmp_path):
        with pytest.raises(ValueError, match="no sequence windows"):
            train(
                TrainConfig(
                    tracks="kaggle_trial/tracks.csv",
                    basins=("ZZ",),
                    epochs=1,
                    batch_size=8,
                    checkpoint=tmp_path / "none.pt",
                    metrics_path=tmp_path / "none.json",
                )
            )
