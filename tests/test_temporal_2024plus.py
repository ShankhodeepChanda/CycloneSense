import pandas as pd
from types import SimpleNamespace

from scripts import evaluate_temporal_2024plus as temporal


def test_evaluator_overlap_scan_does_not_compare_candidate_dataset_to_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, "ROOT", tmp_path)
    training = tmp_path / "training"
    candidate = tmp_path / "data" / "temporal_2024plus"
    training.mkdir()
    candidate.mkdir(parents=True)
    pd.DataFrame([{"sid": "  TRAIN-SID ", "split": "train"}]).to_csv(training / "images_index.csv", index=False)
    pd.DataFrame([{"sid": "CANDIDATE-SID", "split": "temporal_2024plus"}]).to_csv(candidate / "images_index.csv", index=False)
    pd.DataFrame([{"sid": "TRACK-SID", "split": "val"}]).to_csv(training / "tracks.csv", index=False)

    sids, sources = temporal.known_local_sids(candidate.resolve())

    assert sids == {"TRAIN-SID", "TRACK-SID"}
    assert all("temporal_2024plus" not in source for source in sources)


def test_track_pairing_requires_exact_time_wind_coordinates_and_temporal_split():
    key = ("SID-A", pd.Timestamp("2024-01-01T00:00:00Z"))
    candidate = {"sid_key": key[0], "time_utc": key[1], "actual_wind_kt": 50.0,
                 "latitude_num": 10.0, "longitude_num": 120.0}
    matched = SimpleNamespace(track_wind_kt=50.0, track_lat=10.0, track_lon=120.0,
                              split="temporal_2024plus")
    wrong_wind = SimpleNamespace(**{**vars(matched), "track_wind_kt": 51.0})
    wrong_split = SimpleNamespace(**{**vars(matched), "split": "train"})

    assert temporal.track_alignment_issue(candidate, {key: matched}, set()) is None
    assert temporal.track_alignment_issue(candidate, {}, set()) == "no_exact_sid_time_track_match"
    assert temporal.track_alignment_issue(candidate, {key: matched}, {key}) == "duplicate_sid_time_track_rows"
    assert temporal.track_alignment_issue(candidate, {key: wrong_wind}, set()) == "track_wind_label_mismatch"
    assert temporal.track_alignment_issue(candidate, {key: wrong_split}, set()) == "track_not_marked_temporal_holdout"


def test_metrics_include_per_year_and_per_storm_results(tmp_path, monkeypatch):
    monkeypatch.setattr(temporal, "ROOT", tmp_path)
    monkeypatch.setattr(temporal, "PRODUCTION_CHECKPOINT", tmp_path / "models" / "model.pt")
    baseline_dir = tmp_path / "kaggle_multiyear45"
    baseline_dir.mkdir()
    pd.DataFrame([{"channel": "ir", "split": "train", "wind_kt": 60.0}]).to_csv(
        baseline_dir / "images_index.csv", index=False)
    predictions = pd.DataFrame([
        {"sid": "SID-A", "year": 2024, "satellite": "terra", "actual_wind_kt": 50.0,
         "actual_category": "TS", "raw_prediction_kt": 45.0, "application_prediction_kt": 45.0,
         "raw_category": "TS", "application_category": "TS"},
        {"sid": "SID-A", "year": 2024, "satellite": "aqua", "actual_wind_kt": 70.0,
         "actual_category": "C1", "raw_prediction_kt": 80.0, "application_prediction_kt": 80.0,
         "raw_category": "C1", "application_category": "C1"},
    ])

    report = temporal.compute_report(predictions, "a" * 64, "cpu", [])

    assert report["per_year"]["2024"]["images"] == 2
    assert report["per_storm"]["SID-A"]["images"] == 2
    assert report["mean_storm_mae_kt"] == 7.5
