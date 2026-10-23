"""Evaluate the frozen Model A on validated, storm-disjoint 2024+ IR chips."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_dataset import category
from ml.wind_inference import PRODUCTION_CHECKPOINT, load_wind_model, predict_wind, preprocess_wind_image

EXPECTED_MODEL_A_SHA256 = "3ac72bfbd2675e2aa79ce131aa6eef138a4867dfb5ca0063bb4ac5b19068d6d8"
PREDICTION_COLUMNS = ["sid", "time", "satellite", "image_path", "actual_wind_kt", "actual_category",
    "normalized_model_output", "raw_prediction_kt", "application_prediction_kt", "raw_error_kt",
    "application_error_kt", "raw_category", "application_category", "year"]
PROVENANCE_COLUMNS = {"track_source_url", "track_source_file", "track_source_file_sha256",
    "USA_AGENCY", "USA_STATUS", "IFLAG", "TRACK_TYPE", "image_sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sid_key(value: object) -> str:
    return str(value).strip().upper() if pd.notna(value) else ""


def known_local_sids(exclude_root: Path) -> tuple[set[str], list[str]]:
    found: set[str] = set()
    sources: list[str] = []
    skip = {".venv", "node_modules", ".git", "reports"}
    repository_root = ROOT.resolve()
    for path in sorted(ROOT.rglob("images_index.csv")):
        relative = path.resolve().relative_to(repository_root)
        if skip.intersection(relative.parts) or exclude_root == path.resolve().parent or exclude_root in path.resolve().parents:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda col: col in {"sid", "split"})
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if "sid" in frame:
            found.update(frame.sid.map(sid_key).loc[lambda s: s.ne("")])
            sources.append(str(path.relative_to(ROOT)))
    for path in sorted(ROOT.rglob("tracks.csv")):
        relative = path.resolve().relative_to(repository_root)
        if skip.intersection(relative.parts) or exclude_root == path.resolve().parent or exclude_root in path.resolve().parents:
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda col: col in {"sid", "SID", "split"})
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        column = "sid" if "sid" in frame else "SID" if "SID" in frame else None
        if column:
            found.update(frame[column].map(sid_key).loc[lambda s: s.ne("")])
            sources.append(str(path.relative_to(ROOT)))
    return found, sorted(set(sources))


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {"images": int(len(actual)), "mae_kt": float(np.mean(np.abs(error))),
        "rmse_kt": float(np.sqrt(np.mean(error ** 2))), "mean_bias_kt": float(np.mean(error))}


def category_metrics(frame: pd.DataFrame, prediction: str) -> dict:
    truth = frame.actual_category.to_numpy()
    guessed = frame[prediction].to_numpy()
    labels = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
    index = {label: i for i, label in enumerate(labels)}
    distance = np.array([abs(index[a] - index[b]) for a, b in zip(truth, guessed)])
    return {"category_accuracy": float(np.mean(truth == guessed)),
            "within_one_category_accuracy": float(np.mean(distance <= 1))}


def track_alignment_issue(row: dict, track_lookup: dict, ambiguous_keys: set) -> str | None:
    key = (row["sid_key"], row["time_utc"])
    lat, lon = row["latitude_num"], row["longitude_num"]
    if not np.isfinite(lat) or not np.isfinite(lon) or abs(lat) > 90 or abs(lon) > 180:
        return "invalid_or_missing_coordinates"
    if key in ambiguous_keys:
        return "duplicate_sid_time_track_rows"
    track = track_lookup.get(key)
    if track is None:
        return "no_exact_sid_time_track_match"
    target = row["actual_wind_kt"]
    if not np.isfinite(track.track_wind_kt) or not np.isclose(float(track.track_wind_kt), float(target), rtol=0, atol=1e-6):
        return "track_wind_label_mismatch"
    if not np.isfinite(track.track_lat) or not np.isfinite(track.track_lon):
        return "track_coordinates_missing"
    if not np.isclose(float(track.track_lat), float(row["latitude_num"]), rtol=0, atol=1e-6) or not np.isclose(float(track.track_lon), float(row["longitude_num"]), rtol=0, atol=1e-6):
        return "track_coordinates_mismatch"
    if str(track.split).strip().lower() != "temporal_2024plus":
        return "track_not_marked_temporal_holdout"
    return None


def compute_report(frame: pd.DataFrame, checkpoint_hash: str, device: str, source_files: list[str]) -> dict:
    actual = frame.actual_wind_kt.to_numpy(dtype=float)
    raw = frame.raw_prediction_kt.to_numpy(dtype=float)
    app = frame.application_prediction_kt.to_numpy(dtype=float)
    baseline_frame = pd.read_csv(ROOT / "kaggle_multiyear45" / "images_index.csv")
    training = baseline_frame[(baseline_frame.channel == "ir") & (baseline_frame.split == "train")]
    training_targets = pd.to_numeric(training.wind_kt, errors="coerce").dropna()
    baseline = float(training_targets.mean())
    baseline_categories = np.array([category(baseline)] * len(frame))
    actual_categories = frame.actual_category.to_numpy()
    labels = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
    label_index = {label: i for i, label in enumerate(labels)}
    baseline_distance = np.array([abs(label_index[a] - label_index[b]) for a, b in zip(actual_categories, baseline_categories)])
    out = {
        "status": "evaluated", "evaluation": "independent temporal holdout; storm IDs disjoint from all repository indexes/tracks",
        "checkpoint": str(PRODUCTION_CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip() or None,
        "device": device,
        "preprocessing": {"decode": "Pillow RGB", "resize": "torchvision Resize(224), preserve aspect", "tensor": "ToTensor",
            "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225], "target_scale": "wind_kt / 100"},
        "data": {"images": int(len(frame)), "independent_storms": int(frame.sid.nunique()),
            "categories": frame.actual_category.value_counts().to_dict(), "source_file_hashes": source_files},
        "raw_metrics": {**score(actual, raw), **category_metrics(frame, "raw_category"),
            "negative_raw_predictions": int(np.sum(raw < 0))},
        "application_metrics": {**score(actual, app), **category_metrics(frame, "application_category"),
            "clipped_predictions": int(np.sum(raw < 0))},
        "baseline": {"name": "constant training-set mean; frozen from original IR training split", "training_mean_kt": baseline,
            **score(actual, np.full_like(actual, baseline)),
            "category_accuracy": float(np.mean(actual_categories == baseline_categories)),
            "within_one_category_accuracy": float(np.mean(baseline_distance <= 1))},
        "per_category": {}, "per_storm": {}, "per_year": {}, "per_satellite": {},
        "low_sample_guidance": "Flag categories with fewer than 10 images or fewer than 5 independent storms.",
        "holdout_tuning_performed": False,
    }
    for cat, part in frame.groupby("actual_category", sort=False):
        cat_metrics = {**score(part.actual_wind_kt.to_numpy(float), part.raw_prediction_kt.to_numpy(float)),
            "storms": int(part.sid.nunique()), "low_sample": bool(len(part) < 10 or part.sid.nunique() < 5)}
        out["per_category"][cat] = cat_metrics
    for sid, part in frame.groupby("sid", sort=True):
        out["per_storm"][sid] = {**score(part.actual_wind_kt.to_numpy(float), part.raw_prediction_kt.to_numpy(float)),
            "mean_bias_kt": float(np.mean(part.raw_prediction_kt - part.actual_wind_kt)),
            "actual_categories": sorted(part.actual_category.unique().tolist())}
    for group_col, key in (("year", "per_year"), ("satellite", "per_satellite")):
        for label, part in frame.groupby(group_col, dropna=False):
            out[key][str(label)] = score(part.actual_wind_kt.to_numpy(float), part.raw_prediction_kt.to_numpy(float))
    out["mean_storm_mae_kt"] = float(np.mean([entry["mae_kt"] for entry in out["per_storm"].values()]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared dataset root containing images_index.csv and images/")
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Pinned inference device for repeatable evaluation; defaults to CPU")
    args = parser.parse_args()
    data_dir, out_dir = args.data_dir.resolve(), args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {out_dir}")
    if args.batch_size < 1 or args.batch_size > 128:
        raise ValueError("--batch-size must be between 1 and 128")
    out_dir.mkdir(parents=True)
    checkpoint_hash = sha256(PRODUCTION_CHECKPOINT) if PRODUCTION_CHECKPOINT.is_file() else None
    sids, sid_sources = known_local_sids(data_dir)
    readiness = {"status": "blocked_no_valid_pairs", "data_dir": str(data_dir), "checkpoint": str(PRODUCTION_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_hash, "expected_checkpoint_sha256": EXPECTED_MODEL_A_SHA256,
        "checkpoint_frozen_digest_matches": checkpoint_hash == EXPECTED_MODEL_A_SHA256,
        "existing_storm_sources": sid_sources, "overlap_storms_excluded": [], "candidate_images": 0,
        "valid_images": 0, "valid_storms": 0, "invalid_image_rows": 0, "duplicate_rows_excluded": 0,
        "model_inference_performed": False, "metrics": None, "blockers": []}
    empty_predictions = pd.DataFrame(columns=PREDICTION_COLUMNS)
    index_path = data_dir / "images_index.csv"
    if not index_path.is_file():
        readiness["blockers"].append("No images_index.csv exists. Candidate track jobs are not paired with local images.")
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0

    frame = pd.read_csv(index_path)
    required = {"sid", "time", "channel", "satellite", "image_path", "wind_kt", "category", "valid_fraction", "offset_h", "lat", "lon"}
    missing = sorted((required | PROVENANCE_COLUMNS) - set(frame.columns))
    if missing:
        readiness["blockers"].append(f"Index is missing required image/label/provenance columns: {missing}")
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0
    tracks_path = data_dir / "tracks.csv"
    if not tracks_path.is_file():
        readiness["blockers"].append("Paired tracks.csv is missing; exact SID/time/USA_WIND label alignment cannot be independently checked.")
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0
    tracks = pd.read_csv(tracks_path)
    track_required = {"sid", "time", "wind_kt", "lat", "lon", "split"}
    missing_tracks = sorted(track_required - set(tracks.columns))
    if missing_tracks:
        readiness["blockers"].append(f"tracks.csv is missing alignment columns: {missing_tracks}")
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0
    tracks["sid_key"] = tracks.sid.map(sid_key)
    tracks["time_utc"] = pd.to_datetime(tracks.time, errors="coerce", utc=True)
    tracks["track_wind_kt"] = pd.to_numeric(tracks.wind_kt, errors="coerce")
    tracks["track_lat"] = pd.to_numeric(tracks.lat, errors="coerce")
    tracks["track_lon"] = pd.to_numeric(tracks.lon, errors="coerce")
    duplicate_track_keys = tracks.duplicated(["sid_key", "time_utc"], keep=False)
    ambiguous_track_keys = set(zip(tracks.loc[duplicate_track_keys, "sid_key"], tracks.loc[duplicate_track_keys, "time_utc"]))
    track_lookup = {(row.sid_key, row.time_utc): row for row in tracks.loc[~duplicate_track_keys].itertuples()}
    frame["sid_key"] = frame.sid.map(sid_key)
    frame["time_utc"] = pd.to_datetime(frame.time, errors="coerce", utc=True)
    frame["actual_wind_kt"] = pd.to_numeric(frame.wind_kt, errors="coerce")
    frame["latitude_num"] = pd.to_numeric(frame.lat, errors="coerce")
    frame["longitude_num"] = pd.to_numeric(frame.lon, errors="coerce")
    frame["valid_fraction_num"] = pd.to_numeric(frame.valid_fraction, errors="coerce")
    frame["offset_h_num"] = pd.to_numeric(frame.offset_h, errors="coerce")
    frame = frame[frame.channel.eq("ir") & frame.time_utc.dt.year.ge(2024)].copy()
    readiness["candidate_images"] = int(len(frame))
    duplicates = frame.duplicated(["sid_key", "time_utc", "satellite"], keep=False) | frame.duplicated("image_path", keep=False)
    readiness["duplicate_rows_excluded"] = int(duplicates.sum())
    overlap = sorted(set(frame.sid_key) & sids)
    readiness["overlap_storms_excluded"] = overlap
    frame = frame[~frame.sid_key.isin(sids) & ~duplicates].copy()
    invalid_reasons: list[dict] = []
    valid_records: list[dict] = []
    aligned_track_rows = 0
    for row in frame.to_dict(orient="records"):
        reason = None
        stamp = row["time_utc"]
        target = row["actual_wind_kt"]
        key = (row["sid_key"], stamp)
        track = track_lookup.get(key)
        reason = track_alignment_issue(row, track_lookup, ambiguous_track_keys)
        if reason is None and str(row.get("TRACK_TYPE", "")).strip().lower() != "main":
            reason = "track_type_not_main"
        elif reason is None and str(row.get("USA_AGENCY", "")).strip().lower() in {"", "nan", "none"}:
            reason = "missing_usa_agency_provenance"
        elif reason is None and row.get("track_source_url") != builder.IBTRACS_URL:
            reason = "unrecognized_track_source_url"
        elif reason is None and (not isinstance(row.get("track_source_file"), str) or not str(row.get("track_source_file")).strip()):
            reason = "missing_track_source_file"
        elif reason is None and (not isinstance(row.get("track_source_file_sha256"), str) or len(str(row.get("track_source_file_sha256"))) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in str(row.get("track_source_file_sha256")))):
            reason = "invalid_track_source_sha256"
        elif reason is None and (not isinstance(row.get("image_sha256"), str) or len(str(row.get("image_sha256"))) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in str(row.get("image_sha256")))):
            reason = "invalid_image_sha256"
        elif reason is None and (pd.isna(stamp) or stamp.year < 2024):
            reason = "invalid_or_pre_2024_timestamp"
        elif reason is None and (not np.isfinite(target) or target <= 0):
            reason = "missing_or_nonpositive_wind_label"
        elif reason is None and category(float(target)) != str(row["category"]):
            reason = "category_does_not_match_wind_label"
        elif reason is None and (not np.isfinite(row["valid_fraction_num"]) or row["valid_fraction_num"] < 0.6):
            reason = "valid_fraction_below_0.6_or_missing"
        elif reason is None and (not np.isfinite(row["offset_h_num"]) or abs(row["offset_h_num"]) > 3):
            reason = "satellite_pass_offset_exceeds_3h_or_missing"
        elif reason is None and (str(row.get("label_status_review_required", "")).strip().lower() in {"true", "1", "yes"} or "PROV" in str(row.get("USA_STATUS", "")).upper()):
            reason = "provisional_label_requires_review"
        if reason is None:
            aligned_track_rows += 1
        image_path = (data_dir / str(row["image_path"])).resolve()
        try:
            image_path.relative_to(data_dir)
        except ValueError:
            reason = "image_path_escapes_dataset_root"
        if reason is None:
            try:
                with Image.open(image_path) as image:
                    if image.size != (256, 256):
                        reason = f"image_dimensions_not_256x256:{image.size}"
                    else:
                        image.verify()
                if reason is None and sha256(image_path) != str(row["image_sha256"]):
                    reason = "image_sha256_mismatch"
                if reason is None:
                    source_path = Path(str(row["track_source_file"]))
                    if not source_path.is_absolute():
                        source_path = (ROOT / source_path).resolve()
                    if source_path.is_file() and sha256(source_path) != str(row["track_source_file_sha256"]):
                        reason = "track_source_sha256_mismatch"
                if reason is None:
                    tensor = preprocess_wind_image(image_path.read_bytes())
                    valid_records.append({**row, "image_tensor": tensor, "image_path_absolute": image_path})
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                reason = f"missing_or_corrupt_image:{type(exc).__name__}"
        if reason:
            invalid_reasons.append({"sid": row.get("sid"), "time": row.get("time"), "image_path": row.get("image_path"), "reason": reason})
    readiness["invalid_image_rows"] = len(invalid_reasons)
    readiness["track_alignment_valid_rows"] = aligned_track_rows
    readiness["valid_images"] = len(valid_records)
    readiness["valid_storms"] = len({record["sid_key"] for record in valid_records})
    pd.DataFrame(invalid_reasons, columns=["sid", "time", "image_path", "reason"]).to_csv(out_dir / "excluded_rows.csv", index=False)
    if checkpoint_hash != EXPECTED_MODEL_A_SHA256:
        readiness["blockers"].append("Production Model A checkpoint is absent or its SHA-256 differs from the frozen audit digest.")
    if not valid_records:
        readiness["blockers"].append("No valid 2024+ infrared image/track pairs remain after validation and storm-overlap exclusion.")
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        readiness["status"] = "blocked_no_valid_pairs"
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0
    if readiness["blockers"]:
        empty_predictions.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
        print(json.dumps(readiness, indent=2))
        return 0

    model, device = load_wind_model(PRODUCTION_CHECKPOINT, device=args.device)
    prediction_rows = []
    for start in range(0, len(valid_records), args.batch_size):
        batch = valid_records[start:start + args.batch_size]
        images = torch.cat([record["image_tensor"] for record in batch], dim=0)
        result = predict_wind(model, images, device=device)
        for i, record in enumerate(batch):
            raw = float(result.raw_prediction_kt[i])
            app = float(result.application_prediction_kt[i])
            actual = float(record["actual_wind_kt"])
            prediction_rows.append({"sid": record["sid"], "time": record["time"], "satellite": record["satellite"],
                "image_path": record["image_path"], "actual_wind_kt": actual, "actual_category": category(actual),
                "normalized_model_output": float(result.normalized_model_output[i]), "raw_prediction_kt": raw,
                "application_prediction_kt": app, "raw_error_kt": raw - actual, "application_error_kt": app - actual,
                "raw_category": category(raw), "application_category": category(app),
                "year": int(pd.Timestamp(record["time"]).year)})
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    source_hashes = sorted(set(str(r.get("track_source_file_sha256", "unknown")) for r in valid_records))
    metrics = compute_report(predictions, checkpoint_hash, str(device), source_hashes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    readiness.update({"status": "evaluated", "model_inference_performed": True, "metrics": "metrics.json"})
    (out_dir / "temporal_evaluation_readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps({"status": readiness["status"], "images": len(predictions), "storms": predictions.sid.nunique(),
        "raw_mae_kt": metrics["raw_metrics"]["mae_kt"], "negative_raw_predictions": metrics["raw_metrics"]["negative_raw_predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
