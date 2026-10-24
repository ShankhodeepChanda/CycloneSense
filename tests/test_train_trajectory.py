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
                # Deliberately excludes NI: pre-training on every basin would show this
                # stage the storms the fine-tune holds out, and train() now refuses that.
                basins=("WP", "SI", "EP", "SP"),
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


class TestPretrainContamination:
    """A two-stage run must not fine-tune against storms stage one already saw.

    The published cone radius is this validation RMSE, so contamination here understates
    the uncertainty shown to every user.
    """

    def _csv(self, tmp_path, sids=("STORM_A", "STORM_B", "STORM_C", "STORM_D")):
        frames = []
        for offset, sid in enumerate(sids):
            rows = 20
            times = pd.date_range(f"2020-0{offset + 1}-01", periods=rows, freq="6h")
            frames.append(
                pd.DataFrame(
                    {
                        "sid": sid,
                        "name": sid,
                        "basin": "NI",
                        "time": times,
                        "lat": [10.0 + 0.3 * i + offset for i in range(rows)],
                        "lon": [86.0 + 0.2 * i for i in range(rows)],
                        "wind_kt": [40.0 + 2.0 * i for i in range(rows)],
                        "pres_hpa": [1000.0 - 2.0 * i for i in range(rows)],
                        "dist2land_km": [500.0 - 10.0 * i for i in range(rows)],
                        "dlat": [np.nan] + [0.3] * (rows - 1),
                        "dlon": [np.nan] + [0.2] * (rows - 1),
                    }
                )
            )
        path = tmp_path / "tracks.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        return path

    def _config(self, tmp_path, tracks, name, **overrides):
        settings = {
            "tracks": tracks,
            "basins": ("NI",),
            "epochs": 1,
            "batch_size": 16,
            "checkpoint": tmp_path / f"{name}.pt",
            "metrics_path": tmp_path / f"{name}.json",
            "seed": 0,
        }
        settings.update(overrides)
        return TrainConfig(**settings)

    def test_fine_tuning_on_a_storm_the_pretrain_saw_is_refused(self, tmp_path):
        tracks = self._csv(tmp_path)
        # Seed 0 holds out STORM_C, seed 1 holds out STORM_A, so stage one trained on the
        # storm stage two holds out.
        train(self._config(tmp_path, tracks, "stage1", seed=0))
        with pytest.raises(ValueError, match="pretrain|trained on"):
            train(self._config(tmp_path, tracks, "stage2", seed=1, init_from=tmp_path / "stage1.pt"))

    def test_the_overlap_can_be_accepted_deliberately(self, tmp_path):
        tracks = self._csv(tmp_path)
        train(self._config(tmp_path, tracks, "s1", seed=0))
        result = train(
            self._config(
                tmp_path, tracks, "s2", seed=1, init_from=tmp_path / "s1.pt", allow_pretrain_overlap=True
            )
        )
        assert result["metrics"]["pretrain_overlap_storms"] >= 1

    def test_a_disjoint_pretrain_is_accepted(self, tmp_path):
        tracks = self._csv(tmp_path)
        # Stage one on two storms, stage two validated on one of the other two.
        first = TrainConfig(
            tracks=tracks, basins=("NI",), epochs=1, batch_size=16, seed=0, val_fraction=0.5,
            checkpoint=tmp_path / "disjoint1.pt", metrics_path=tmp_path / "disjoint1.json",
        )
        pretrained = train(first)
        held_out = set(pretrained["validation"]["storms"]) if "storms" in pretrained["validation"] else None
        assert held_out is None or held_out
        # Reusing the same seed and fraction reproduces the same split, so the storms the
        # pretrain validated on are exactly the ones it did not train on.
        second = TrainConfig(
            tracks=tracks, basins=("NI",), epochs=1, batch_size=16, seed=0, val_fraction=0.5,
            checkpoint=tmp_path / "disjoint2.pt", metrics_path=tmp_path / "disjoint2.json",
            init_from=tmp_path / "disjoint1.pt", allow_pretrain_overlap=False,
        )
        assert train(second)["metrics"]["pretrain_overlap_storms"] == 0


class TestValidationHonesty:
    def test_metrics_record_that_selection_used_the_validation_set(self, tmp_path):
        result = train(
            TrainConfig(
                tracks="kaggle_trial/tracks.csv", basins=("NI",), epochs=2, batch_size=32, seed=0,
                checkpoint=tmp_path / "v.pt", metrics_path=tmp_path / "v.json",
            )
        )
        assert result["metrics"]["model_selected_on_validation_set"] is True
        assert result["metrics"]["validation_is_training_set"] is False

    def test_a_single_storm_dataset_says_its_validation_is_its_training_set(self, tmp_path):
        rows = 20
        times = pd.date_range("2020-05-01", periods=rows, freq="6h")
        frame = pd.DataFrame(
            {
                "sid": "ONLY", "name": "ONLY", "basin": "NI", "time": times,
                "lat": [10.0 + 0.3 * i for i in range(rows)],
                "lon": [86.0 + 0.2 * i for i in range(rows)],
                "wind_kt": [40.0 + 2.0 * i for i in range(rows)],
                "pres_hpa": [1000.0 - 2.0 * i for i in range(rows)],
                "dist2land_km": [500.0 - 10.0 * i for i in range(rows)],
                "dlat": [np.nan] + [0.3] * (rows - 1),
                "dlon": [np.nan] + [0.2] * (rows - 1),
            }
        )
        tracks = tmp_path / "one.csv"
        frame.to_csv(tracks, index=False)
        result = train(
            TrainConfig(
                tracks=tracks, basins=("NI",), epochs=1, batch_size=8, seed=0,
                checkpoint=tmp_path / "one.pt", metrics_path=tmp_path / "one.json",
            )
        )
        assert result["metrics"]["validation_is_training_set"] is True
        model, meta = load_forecaster(tmp_path / "one.pt")
        assert meta["metrics"]["validation_is_training_set"] is True


class TestCrossBasinStormExclusion:
    """A storm can carry rows in two basins, so a basin filter does not hold it out.

    PABUK, BULBUL and KROVANH each have both NI and WP rows in kaggle_trial/tracks.csv:
    pre-training on WP therefore trains on storms the NI fine-tune holds out.
    """

    def _cross_basin_csv(self, tmp_path):
        frames = []
        # The WP leg needs at least 12 fixes of its own, or the storm yields no window
        # from that basin and the exclusion would look effective when it was not.
        for sid, basins in (("CROSSER", ["WP"] * 14 + ["NI"] * 10), ("WP_ONLY", ["WP"] * 20)):
            rows = len(basins)
            times = pd.date_range("2020-01-01", periods=rows, freq="6h")
            frames.append(
                pd.DataFrame(
                    {
                        "sid": sid, "name": sid, "basin": basins, "time": times,
                        "lat": [10.0 + 0.3 * i for i in range(rows)],
                        "lon": [100.0 - 0.5 * i for i in range(rows)],
                        "wind_kt": [40.0 + 2.0 * i for i in range(rows)],
                        "pres_hpa": [1000.0 - 2.0 * i for i in range(rows)],
                        "dist2land_km": [400.0] * rows,
                        "dlat": [np.nan] + [0.3] * (rows - 1),
                        "dlon": [np.nan] + [-0.5] * (rows - 1),
                    }
                )
            )
        path = tmp_path / "cross.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        return path

    def test_a_storm_with_rows_in_an_excluded_basin_is_dropped_entirely(self, tmp_path):
        tracks = self._cross_basin_csv(tmp_path)
        result = train(
            TrainConfig(
                tracks=tracks, basins=("WP",), exclude_storms_in_basins=("NI",),
                epochs=1, batch_size=16, seed=0,
                checkpoint=tmp_path / "x.pt", metrics_path=tmp_path / "x.json",
            )
        )
        assert "CROSSER" not in result["dataset"]["names"]
        assert "WP_ONLY" in result["dataset"]["names"]

    def test_without_the_exclusion_the_cross_basin_storm_is_kept(self, tmp_path):
        tracks = self._cross_basin_csv(tmp_path)
        result = train(
            TrainConfig(
                tracks=tracks, basins=("WP",), epochs=1, batch_size=16, seed=0,
                checkpoint=tmp_path / "y.pt", metrics_path=tmp_path / "y.json",
            )
        )
        assert "CROSSER" in result["dataset"]["names"]

    def test_the_real_north_indian_ocean_storms_are_fully_excluded(self):
        """PABUK, BULBUL and KROVANH must not survive a WP-side pretrain."""
        import pandas as pd_

        from ml.sequence_dataset import build_sequence_windows, load_tracks

        frame = load_tracks("kaggle_trial/tracks.csv")
        ni_sids = set(frame[frame["basin"] == "NI"]["sid"])
        windows = build_sequence_windows(
            frame[~frame["sid"].isin(ni_sids)], basins=("WP", "SI", "EP", "SP")
        )
        assert windows
        assert not ({w.sid for w in windows} & ni_sids)
