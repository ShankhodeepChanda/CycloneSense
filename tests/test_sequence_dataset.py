"""Tests for the North Indian Ocean sliding-window sequence pipeline.

The pipeline turns IBTrACS best-track rows into fixed-length temporal windows:
four 6-hourly input steps (t-18h .. t0) and eight forecast targets (+6h .. +48h).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.sequence_dataset import (
    FEATURES,
    INPUT_STEPS,
    NORM,
    STEP_HOURS,
    TARGET_LEADS,
    CycloneSequenceDataset,
    build_sequence_windows,
    denormalise,
    load_tracks,
    normalise,
    split_by_storm,
)

TRACKS = "kaggle_trial/tracks.csv"
WINDOW_ROWS = INPUT_STEPS + len(TARGET_LEADS)


def synthetic_track(
    sid: str = "2020136N10088",
    name: str = "AMPHAN",
    basin: str = "NI",
    rows: int = WINDOW_ROWS,
    start: str = "2020-05-16 00:00:00",
    step_hours: int = STEP_HOURS,
) -> pd.DataFrame:
    """A perfectly regular straight-line storm: easy to assert exact target values on."""
    times = pd.date_range(start=start, periods=rows, freq=f"{step_hours}h")
    return pd.DataFrame(
        {
            "sid": sid,
            "name": name,
            "basin": basin,
            "season": 2020,
            "time": times,
            "lat": [10.0 + 0.5 * i for i in range(rows)],
            "lon": [86.0 + 0.25 * i for i in range(rows)],
            "wind_kt": [40.0 + 5.0 * i for i in range(rows)],
            "pres_hpa": [1000.0 - 4.0 * i for i in range(rows)],
            "dist2land_km": [600.0 - 20.0 * i for i in range(rows)],
            "dlat": [np.nan] + [0.5] * (rows - 1),
            "dlon": [np.nan] + [0.25] * (rows - 1),
        }
    )


class TestWindowConstruction:
    def test_window_arrays_have_the_documented_shapes(self):
        windows = build_sequence_windows(synthetic_track())
        assert len(windows) == 1
        window = windows[0]
        assert window.features.shape == (INPUT_STEPS, len(FEATURES))
        assert window.target_coords.shape == (len(TARGET_LEADS), 2)
        assert window.target_intensity.shape == (len(TARGET_LEADS), 2)

    def test_target_coords_are_cumulative_offsets_from_t0(self):
        window = build_sequence_windows(synthetic_track())[0]
        # t0 is the last input step: row index INPUT_STEPS - 1.
        expected = np.array([[0.5 * (k + 1), 0.25 * (k + 1)] for k in range(len(TARGET_LEADS))])
        np.testing.assert_allclose(window.target_coords, expected, atol=1e-9)

    def test_target_intensity_is_absolute_wind_and_pressure(self):
        window = build_sequence_windows(synthetic_track())[0]
        wind0, pres0 = 40.0 + 5.0 * (INPUT_STEPS - 1), 1000.0 - 4.0 * (INPUT_STEPS - 1)
        expected = np.array(
            [[wind0 + 5.0 * (k + 1), pres0 - 4.0 * (k + 1)] for k in range(len(TARGET_LEADS))]
        )
        np.testing.assert_allclose(window.target_intensity, expected, atol=1e-9)

    def test_t0_is_the_last_input_step(self):
        window = build_sequence_windows(synthetic_track())[0]
        assert window.t0 == pd.Timestamp("2020-05-16 00:00:00") + pd.Timedelta(
            hours=STEP_HOURS * (INPUT_STEPS - 1)
        )

    def test_a_track_one_row_short_yields_no_window(self):
        assert build_sequence_windows(synthetic_track(rows=WINDOW_ROWS - 1)) == []

    def test_longer_track_slides_one_step_at_a_time(self):
        windows = build_sequence_windows(synthetic_track(rows=WINDOW_ROWS + 3))
        assert len(windows) == 4


class TestContiguity:
    def test_a_time_gap_splits_the_sequence(self):
        """A single missing 6-hourly fix must not be bridged into a fake window."""
        df = synthetic_track(rows=WINDOW_ROWS + 1)
        df = df.drop(index=4).reset_index(drop=True)  # punch a 12 h hole mid-track
        assert build_sequence_windows(df) == []

    def test_irregular_spacing_is_rejected(self):
        df = synthetic_track(rows=WINDOW_ROWS, step_hours=3)
        assert build_sequence_windows(df) == []

    def test_windows_never_span_two_storms(self):
        """Two storms whose fixes are contiguous in time must still not be joined."""
        first = synthetic_track(sid="STORM_A", name="A", rows=WINDOW_ROWS - 2)
        follow_on = pd.Timestamp("2020-05-16 00:00:00") + pd.Timedelta(
            hours=STEP_HOURS * (WINDOW_ROWS - 2)
        )
        second = synthetic_track(
            sid="STORM_B", name="B", rows=WINDOW_ROWS - 2, start=str(follow_on)
        )
        windows = build_sequence_windows(pd.concat([first, second], ignore_index=True))
        assert windows == []


class TestMissingValues:
    def test_missing_pressure_is_imputed_and_declared(self):
        df = synthetic_track()
        df.loc[1, "pres_hpa"] = np.nan
        window = build_sequence_windows(df)[0]
        pressure_column = window.features[:, FEATURES.index("pres_hpa")]
        assert np.isfinite(pressure_column).all()
        assert window.was_imputed["pres_hpa"] is True
        assert window.was_imputed["lat"] is False

    def test_first_fix_motion_deltas_default_to_zero(self):
        window = build_sequence_windows(synthetic_track())[0]
        assert window.features[0, FEATURES.index("dlat")] == 0.0
        assert window.features[0, FEATURES.index("dlon")] == 0.0
        assert window.was_imputed["dlat"] is True

    def test_a_window_with_a_missing_target_is_dropped(self):
        df = synthetic_track()
        df.loc[WINDOW_ROWS - 1, "lat"] = np.nan
        assert build_sequence_windows(df) == []

    def test_an_all_nan_pressure_storm_still_produces_windows(self):
        df = synthetic_track()
        df["pres_hpa"] = np.nan
        window = build_sequence_windows(df)[0]
        assert np.isfinite(window.features[:, FEATURES.index("pres_hpa")]).all()
        assert np.isfinite(window.target_intensity).all()
        assert window.was_imputed["pres_hpa"] is True


class TestBasinFiltering:
    def test_only_the_requested_basin_is_kept(self):
        ni = synthetic_track(sid="NI_1", basin="NI")
        wp = synthetic_track(sid="WP_1", basin="WP")
        windows = build_sequence_windows(pd.concat([ni, wp], ignore_index=True), basins=("NI",))
        assert {w.basin for w in windows} == {"NI"}

    def test_basins_none_keeps_every_basin(self):
        ni = synthetic_track(sid="NI_1", basin="NI")
        wp = synthetic_track(sid="WP_1", basin="WP")
        windows = build_sequence_windows(pd.concat([ni, wp], ignore_index=True), basins=None)
        assert {w.basin for w in windows} == {"NI", "WP"}

    def test_real_north_indian_ocean_data_yields_a_trainable_window_count(self):
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        assert len(windows) >= 250
        assert {w.basin for w in windows} == {"NI"}
        names = {w.name for w in windows}
        assert "AMPHAN" in names and "FANI" in names

    def test_real_windows_are_finite_everywhere(self):
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        assert all(np.isfinite(w.features).all() for w in windows)
        assert all(np.isfinite(w.target_coords).all() for w in windows)
        assert all(np.isfinite(w.target_intensity).all() for w in windows)


class TestStormGroupedSplit:
    def test_no_storm_appears_in_both_sides_of_the_split(self):
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        train, val = split_by_storm(windows, val_fraction=0.25, seed=0)
        assert train and val
        assert not ({w.sid for w in train} & {w.sid for w in val})
        assert len(train) + len(val) == len(windows)

    def test_the_split_is_deterministic_for_a_seed(self):
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        first = split_by_storm(windows, val_fraction=0.25, seed=7)
        second = split_by_storm(windows, val_fraction=0.25, seed=7)
        assert [w.t0 for w in first[1]] == [w.t0 for w in second[1]]


class TestNormalisation:
    def test_normalising_the_centre_gives_zero(self):
        centre = np.array([[NORM[name][0] for name in FEATURES]], dtype=np.float64)
        np.testing.assert_allclose(normalise(centre), 0.0, atol=1e-9)

    def test_denormalise_inverts_normalise(self):
        raw = np.array([[12.0, 88.0, 0.4, -0.2, 75.0, 970.0, 180.0]])
        np.testing.assert_allclose(denormalise(normalise(raw)), raw, atol=1e-9)

    def test_real_features_normalise_into_a_small_range(self):
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        stacked = normalise(np.concatenate([w.features for w in windows], axis=0))
        assert np.abs(stacked).max() < 10.0


class TestTorchDataset:
    def test_dataset_item_exposes_normalised_tensors_for_the_model(self):
        torch = pytest.importorskip("torch")
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        dataset = CycloneSequenceDataset(windows)
        assert len(dataset) == len(windows)
        item = dataset[0]
        assert item["tabular"].shape == (INPUT_STEPS, len(FEATURES))
        assert item["target_delta"].shape == (len(TARGET_LEADS), 2)
        assert item["target_intensity"].shape == (len(TARGET_LEADS), 2)
        assert item["image_mask"].shape == (INPUT_STEPS,)
        assert item["tabular"].dtype == torch.float32
        assert torch.isfinite(item["tabular"]).all()

    def test_dataset_without_images_marks_every_step_as_missing(self):
        pytest.importorskip("torch")
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        item = CycloneSequenceDataset(windows)[0]
        assert item["image_mask"].sum().item() == 0.0
        assert item["image"].shape == (INPUT_STEPS, 3, 224, 224)

    def test_dataset_batches_with_the_default_collate(self):
        torch = pytest.importorskip("torch")
        windows = build_sequence_windows(load_tracks(TRACKS), basins=("NI",))
        loader = torch.utils.data.DataLoader(CycloneSequenceDataset(windows), batch_size=4)
        batch = next(iter(loader))
        assert batch["tabular"].shape == (4, INPUT_STEPS, len(FEATURES))
        assert batch["target_delta"].shape == (4, len(TARGET_LEADS), 2)


class TestImageAlignment:
    def test_image_paths_align_one_per_input_step(self):
        window = build_sequence_windows(synthetic_track())[0]
        assert len(window.image_paths) == INPUT_STEPS
        assert all(path is None for path in window.image_paths)

    def test_an_images_index_attaches_chips_to_matching_steps(self):
        df = synthetic_track()
        index = pd.DataFrame(
            {
                "image_path": ["images/ir/terra/amphan_0.jpg"],
                "channel": ["ir"],
                "sid": [df.loc[1, "sid"]],
                "time": [df.loc[1, "time"]],
            }
        )
        window = build_sequence_windows(df, images_index=index)[0]
        assert window.image_paths[1] == "images/ir/terra/amphan_0.jpg"
        assert window.image_paths[0] is None
