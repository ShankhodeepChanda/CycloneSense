"""Plan or explicitly fetch a reproducible, storm-disjoint 2024+ IR holdout.

Default behaviour only reads local files and writes a manifest. Network imagery
retrieval requires both --fetch and an explicit --max-requests limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import build_dataset as builder


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sid(value: object) -> str:
    return str(value).strip().upper() if pd.notna(value) else ""


def local_storm_ids() -> tuple[set[str], list[str]]:
    storm_ids: set[str] = set()
    inputs: list[str] = []
    excluded_dirs = {".venv", "node_modules", ".git", "reports"}
    for index_path in sorted(ROOT.rglob("images_index.csv")):
        if excluded_dirs.intersection(index_path.parts):
            continue
        try:
            frame = pd.read_csv(index_path, usecols=lambda name: name in {"sid", "split"})
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if "sid" not in frame:
            continue
        storm_ids.update(frame.sid.map(normalize_sid).loc[lambda col: col.ne("")])
        inputs.append(str(index_path.relative_to(ROOT)))
    for tracks_path in sorted(ROOT.rglob("tracks.csv")):
        if excluded_dirs.intersection(tracks_path.parts):
            continue
        try:
            frame = pd.read_csv(tracks_path, usecols=lambda name: name in {"sid", "SID", "split"})
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        sid_col = "sid" if "sid" in frame else "SID" if "SID" in frame else None
        if sid_col:
            storm_ids.update(frame[sid_col].map(normalize_sid).loc[lambda col: col.ne("")])
            inputs.append(str(tracks_path.relative_to(ROOT)))
    return storm_ids, sorted(set(inputs))


def read_source_rows(path: Path) -> pd.DataFrame:
    wanted = {
        "SID", "ISO_TIME", "USA_WIND", "USA_PRES", "USA_AGENCY", "USA_STATUS",
        "WMO_WIND", "WMO_AGENCY", "IFLAG", "TRACK_TYPE", "LAT", "LON",
    }
    raw = pd.read_csv(path, skiprows=[1], usecols=lambda name: name in wanted, dtype=str, low_memory=False)
    raw["sid_key"] = raw["SID"].map(normalize_sid)
    raw["time_key"] = pd.to_datetime(raw["ISO_TIME"], errors="coerce", utc=True)
    return raw.drop_duplicates(["sid_key", "time_key"])


def enrich_job(job: dict, labels: pd.DataFrame, source_path: Path, source_hash: str) -> dict:
    stamp = pd.Timestamp(job["time"])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    key = (normalize_sid(job["sid"]), stamp)
    matches = labels[(labels.sid_key == key[0]) & (labels.time_key == key[1])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one raw IBTrACS label for {key}, found {len(matches)}")
    source = matches.iloc[0]
    return {
        **job,
        "channel": "ir",
        "year": stamp.year,
        "time": stamp.isoformat(),
        "split": "temporal_2024plus",
        "track_source_url": builder.IBTRACS_URL,
        "track_source_file": str(source_path.relative_to(ROOT)) if source_path.is_relative_to(ROOT) else str(source_path),
        "track_source_file_sha256": source_hash,
        "USA_AGENCY": source.get("USA_AGENCY"),
        "USA_STATUS": source.get("USA_STATUS"),
        "WMO_WIND": source.get("WMO_WIND"),
        "WMO_AGENCY": source.get("WMO_AGENCY"),
        "IFLAG": source.get("IFLAG"),
        "TRACK_TYPE": source.get("TRACK_TYPE"),
        "label_status_review_required": "PROV" in str(source.get("USA_STATUS", "")).upper(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/raw/ibtracs.since1980.list.v04r01.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="New, empty output path; existing paths are refused")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year - 1,
                        help="Defaults to last completed UTC year; excludes current provisional season")
    parser.add_argument("--fetch", action="store_true", help="Explicitly retrieve IR chips from NASA GIBS")
    parser.add_argument("--max-requests", type=int, default=0,
                        help="Required with --fetch. Hard limit for GIBS chip requests (max 500 per run).")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"Refusing to use existing output path: {out_dir}")
    if args.start_year < 2024 or args.end_year < args.start_year:
        raise ValueError("Temporal holdout must start in 2024 or later and end no earlier than it starts.")
    if args.fetch and not 1 <= args.max_requests <= 500:
        raise ValueError("--fetch requires --max-requests between 1 and 500.")
    if not args.fetch and args.max_requests:
        raise ValueError("--max-requests requires --fetch.")
    args.cache = args.cache.resolve()
    if not args.cache.is_file():
        raise FileNotFoundError(f"Local IBTrACS cache not found: {args.cache}. This script does not auto-download tracks.")

    out_dir.mkdir(parents=True)
    raw_hash = sha256(args.cache)
    source_rows = read_source_rows(args.cache)
    tracks = builder.load_tracks(args.cache, args.start_year, args.end_year)
    tracks = tracks[tracks.time.dt.year.between(args.start_year, args.end_year)].copy()
    plans = builder.plan_chips(tracks, chip_deg=10.0)
    planned = pd.DataFrame(plans)
    if len(planned):
        planned["sid_key"] = planned.sid.map(normalize_sid)
        planned["year"] = pd.to_datetime(planned.time, utc=True).dt.year
        planned = planned[planned.year.between(args.start_year, args.end_year)]
    existing_storms, existing_inputs = local_storm_ids()
    candidate_storms = set(planned.sid.map(normalize_sid)) if len(planned) else set()
    overlap = sorted(candidate_storms & existing_storms)
    eligible = planned[~planned.sid_key.isin(existing_storms)].copy() if len(planned) else planned
    job_dicts = eligible.drop(columns=["sid_key"]).to_dict(orient="records") if len(eligible) else []
    manifest = [enrich_job(job, source_rows, args.cache, raw_hash) for job in job_dicts]
    manifest_frame = pd.DataFrame(manifest)
    manifest_path = out_dir / "candidate_manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)

    local_pairs = []
    excluded_dirs = {".venv", "node_modules", ".git", "reports"}
    for index_path in sorted(ROOT.rglob("images_index.csv")):
        if excluded_dirs.intersection(index_path.parts):
            continue
        frame = pd.read_csv(index_path, usecols=lambda name: name in {"sid", "channel", "time"})
        stamps = pd.to_datetime(frame.time, errors="coerce", utc=True)
        pairs = frame[frame.channel.eq("ir") & stamps.dt.year.ge(2024)]
        local_pairs.append({"index": str(index_path.relative_to(ROOT)), "rows": int(len(pairs)),
                            "storms": int(pairs.sid.map(normalize_sid).nunique())})

    fetched = 0
    failed = 0
    if args.fetch:
        limited = manifest[:args.max_requests]
        session = builder.make_session()
        index_rows = []
        track_ids = {normalize_sid(row["sid"]) for row in limited}
        future_tracks = tracks[tracks.sid.map(normalize_sid).isin(track_ids)].copy()
        future_tracks["split"] = "temporal_2024plus"
        future_tracks.to_csv(out_dir / "tracks.csv", index=False)
        for row in limited:
            job = {key: row[key] for key in ("sat", "date", "sid", "name", "basin", "time", "lat", "lon",
                    "wind_kt", "pres_hpa", "category", "ri_24h", "offset_h") if key in row}
            job["split"] = "temporal_2024plus"
            result = builder.do_job(session, job, "ir", out_dir, 10.0, 256, 0.6)
            if result is None:
                failed += 1
                continue
            image_path = out_dir / result["image_path"]
            result.update({key: row.get(key) for key in ("track_source_url", "track_source_file",
                "track_source_file_sha256", "USA_AGENCY", "USA_STATUS", "WMO_WIND", "WMO_AGENCY",
                "IFLAG", "TRACK_TYPE", "year", "label_status_review_required")})
            result["image_sha256"] = sha256(image_path)
            result["image_width"] = 256
            result["image_height"] = 256
            result["split"] = "temporal_2024plus"
            index_rows.append(result)
        fetched = len(index_rows)
        pd.DataFrame(index_rows).to_csv(out_dir / "images_index.csv", index=False)

    result_manifest_hash = sha256(manifest_path)
    readiness = {
        "status": "candidate_manifest_ready" if len(manifest_frame) else "blocked_no_eligible_track_jobs",
        "source": {"url": builder.IBTRACS_URL, "local_file": str(args.cache), "sha256": raw_hash,
                   "version": "IBTrACS v04r01", "original_retrieval_date": "not recorded in the local cache metadata",
                   "pipeline_run_utc": datetime.now(timezone.utc).isoformat()},
        "years": {"start": args.start_year, "end": args.end_year},
        "track_rows": int(len(tracks)), "candidate_storms": int(len(candidate_storms)),
        "overlap_storms_excluded": overlap, "eligible_storms_after_overlap": int(eligible.sid.nunique()) if len(eligible) else 0,
        "planned_ir_jobs": int(len(manifest_frame)), "manifest_sha256": result_manifest_hash,
        "local_2024plus_ir_inventory": local_pairs,
        "local_2024plus_ir_rows": int(sum(item["rows"] for item in local_pairs)),
        "fetch_requested": bool(args.fetch), "max_requests": args.max_requests if args.fetch else 0,
        "retrieved_images": fetched, "failed_or_no_coverage_requests": failed,
        "imagery_paired_and_validated": bool(fetched),
        "existing_train_val_test_splits_modified": False,
        "network_requests_performed": fetched + failed,
        "local_split_sources_checked": existing_inputs,
        "validation_contract": ["6-hourly main-track IBTrACS records", "USA_WIND positive knots retained as wind_kt",
            "closest daytime Terra/Aqua pass within 3 h", "IR only", "256 x 256 RGB palette-rendered chips",
            "valid pixel fraction >= 0.6", "storm IDs compared against all local indexes/tracks",
            "raw source agency/status fields retained", "image SHA-256 recorded when fetched"],
        "limitations": ["A planned GIBS request is not evidence that an image exists until fetched and decoded.",
            "USA/WMO best-track fields have agency-dependent practices; inspect USA_AGENCY, USA_STATUS, IFLAG and WMO fields.",
            "No training or evaluation is performed by this preparation script."],
    }
    (out_dir / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps({"status": readiness["status"], "output": str(out_dir), "track_rows": len(tracks),
        "eligible_storms": readiness["eligible_storms_after_overlap"], "planned_ir_jobs": len(manifest_frame),
        "local_2024plus_ir_rows": readiness["local_2024plus_ir_rows"], "network_requests": readiness["network_requests_performed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
