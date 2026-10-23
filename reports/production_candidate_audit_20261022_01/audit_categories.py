from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
DATASETS = {
    "original_2018_2020": (ROOT / "kaggle_multiyear45", ["train", "val", "test"]),
    "temporal_2021_2023": (ROOT / "kaggle_temporal_test", ["temporal_test"]),
}


def cat(kt: float) -> str:
    for limit, name in ((34, "TD"), (64, "TS"), (83, "C1"), (96, "C2"), (113, "C3"), (137, "C4")):
        if kt < limit:
            return name
    return "C5"


def files_for_index(frame: pd.DataFrame, root: Path) -> tuple[int, list[str]]:
    paths = frame["image_path"].dropna().astype(str).tolist()
    missing = [p for p in paths if not (root / p).is_file()]
    return len(missing), missing[:5]


def audit_index(name: str, root: Path, allowed: list[str]):
    raw = pd.read_csv(root / "images_index.csv")
    raw = raw[raw["channel"].eq("ir") & raw["split"].isin(allowed)].copy()
    raw["wind_kt"] = pd.to_numeric(raw["wind_kt"], errors="coerce")
    raw["derived_category"] = raw["wind_kt"].map(lambda x: cat(float(x)) if pd.notna(x) else None)
    raw["time_parsed"] = pd.to_datetime(raw["time"], errors="coerce")
    raw["offset_h"] = pd.to_numeric(raw["offset_h"], errors="coerce")
    raw["valid_fraction"] = pd.to_numeric(raw["valid_fraction"], errors="coerce")
    dist = []
    split_summary = {}
    for split in allowed:
        part = raw[raw["split"].eq(split)]
        split_summary[split] = {"images": int(len(part)), "storms": int(part["sid"].nunique())}
        for category in CATS:
            subset = part[part["derived_category"].eq(category)]
            dist.append({
                "dataset": name,
                "split": split,
                "category": category,
                "images": int(len(subset)),
                "storms": int(subset["sid"].nunique()),
                "image_share_of_split": float(len(subset) / len(part)) if len(part) else 0.0,
                "image_count_warning": bool(len(subset) < 10),
                "storm_count_warning": bool(subset["sid"].nunique() < 5),
            })
    overlaps = {}
    for i, a in enumerate(allowed):
        a_sids = set(raw[raw["split"].eq(a)]["sid"].dropna().astype(str))
        for b in allowed[i + 1:]:
            b_sids = set(raw[raw["split"].eq(b)]["sid"].dropna().astype(str))
            overlaps[f"{a}__{b}"] = sorted(a_sids & b_sids)
    missing_count, missing_sample = files_for_index(raw, root)
    category_mismatches = raw[raw["category"].astype(str) != raw["derived_category"].astype(str)]
    quality = {
        "index_path": str((root / "images_index.csv").relative_to(ROOT)),
        "channel": "ir",
        "year_range": [int(raw["time_parsed"].dt.year.min()), int(raw["time_parsed"].dt.year.max())],
        "rows": int(len(raw)),
        "split_summary": split_summary,
        "storm_overlap_between_splits": overlaps,
        "missing_targets": int(raw["wind_kt"].isna().sum()),
        "nonfinite_targets": int((~np.isfinite(raw["wind_kt"].to_numpy(dtype=float))).sum()),
        "nonpositive_targets": int((raw["wind_kt"] <= 0).sum()),
        "stored_vs_recomputed_category_mismatches": int(len(category_mismatches)),
        "duplicate_image_paths": int(raw["image_path"].duplicated().sum()),
        "missing_image_files": missing_count,
        "missing_image_path_examples": missing_sample,
        "valid_fraction_min": float(raw["valid_fraction"].min()),
        "valid_fraction_median": float(raw["valid_fraction"].median()),
        "valid_fraction_below_dataset_threshold_0_6": int((raw["valid_fraction"] < 0.6).sum()),
        "max_abs_satellite_track_offset_hours": float(raw["offset_h"].abs().max()),
        "offset_over_3_hours": int((raw["offset_h"].abs() > 3.0).sum()),
    }
    return raw, dist, quality


def raw_ibtracs_summary(path: Path) -> dict:
    years = Counter()
    main_positive = Counter()
    agencies = Counter()
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "present": False}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        next(reader, None)  # IBTrACS second row contains units
        for row in reader:
            time = row.get("ISO_TIME", "")
            if len(time) < 4 or not time[:4].isdigit():
                continue
            year = int(time[:4])
            if year < 2024:
                continue
            years[year] += 1
            wind = row.get("USA_WIND", "")
            try:
                positive = float(wind) > 0
            except (TypeError, ValueError):
                positive = False
            if str(row.get("TRACK_TYPE", "")).strip().lower() == "main" and positive:
                main_positive[year] += 1
                agencies[(year, row.get("USA_AGENCY", ""))] += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "present": True,
        "records_by_year_2024_plus": dict(sorted(years.items())),
        "main_track_records_with_positive_USA_WIND_by_year_2024_plus": dict(sorted(main_positive.items())),
        "USA_AGENCY_values_for_main_positive_records_2024_plus": {
            str(year): dict(sorted({agency: n for (yr, agency), n in agencies.items() if yr == year}.items()))
            for year in sorted({yr for yr, _ in agencies})
        },
    }


def main():
    all_dist = []
    audit = {}
    primary = None
    temporal = None
    for name, (root, splits) in DATASETS.items():
        frame, dist, quality = audit_index(name, root, splits)
        all_dist.extend(dist)
        audit[name] = quality
        if name == "original_2018_2020":
            primary = frame
        else:
            temporal = frame
    pd.DataFrame(all_dist).to_csv(OUT / "category_distribution_by_split.csv", index=False)

    # Verify exact storm identity separation across the earlier training pool and later temporal holdout.
    primary_sids = set(primary["sid"].dropna().astype(str))
    temporal_sids = set(temporal["sid"].dropna().astype(str))
    audit["cross_dataset_storm_overlap_original_vs_temporal"] = sorted(primary_sids & temporal_sids)

    # Compare image-index labels to the corresponding source-track rows.
    tracks = pd.read_csv(ROOT / "kaggle_multiyear45" / "tracks.csv")
    tracks["wind_kt"] = pd.to_numeric(tracks["wind_kt"], errors="coerce")
    tracks["time"] = pd.to_datetime(tracks["time"], errors="coerce")
    primary_for_join = primary.copy()
    primary_for_join["time"] = pd.to_datetime(primary_for_join["time"], errors="coerce")
    join = primary_for_join.merge(tracks[[c for c in ["sid", "time", "wind_kt", "wmo_wind_kt"] if c in tracks.columns]], on=["sid", "time"], suffixes=("_image", "_track"), how="left")
    audit["original_index_to_tracks_alignment"] = {
        "rows": int(len(join)),
        "rows_without_exact_sid_time_match": int(join["wind_kt_track"].isna().sum()),
        "rows_where_image_and_track_USA_wind_differ": int((join["wind_kt_image"] != join["wind_kt_track"]).sum()),
        "wmo_wind_available_rows": int(pd.to_numeric(join.get("wmo_wind_kt"), errors="coerce").notna().sum()) if "wmo_wind_kt" in join else 0,
        "note": "Image targets match the saved processed track values; this is an internal consistency check, not independent meteorological label verification.",
    }

    # Count raw source track records locally; these are labels/tracks, not downloaded image chips.
    cache = ROOT / "data" / "raw" / "ibtracs.since1980.list.v04r01.csv"
    audit["local_raw_ibtracs_2024_plus"] = raw_ibtracs_summary(cache)

    # Existing intensity dataset index inventory is intentionally limited to labeled pairs.
    inventory = []
    for path in sorted(ROOT.rglob("images_index.csv")):
        rel = path.relative_to(ROOT)
        if any(part in {"node_modules", ".venv", ".git"} for part in rel.parts):
            continue
        frame = pd.read_csv(path, usecols=lambda c: c in {"channel", "time", "split", "sid"})
        frame = frame[frame["channel"].eq("ir")]
        times = pd.to_datetime(frame["time"], errors="coerce")
        inventory.append({
            "index_path": str(rel),
            "ir_images": int(len(frame)),
            "ir_storms": int(frame["sid"].nunique()),
            "year_min": int(times.dt.year.min()) if times.notna().any() else None,
            "year_max": int(times.dt.year.max()) if times.notna().any() else None,
        })
    audit["local_ir_chip_index_inventory"] = inventory
    audit["finding_2024_plus"] = (
        "Local raw IBTrACS track rows for 2024+ are present in the cache, but no indexed labeled IR image chips are present for 2024+; the newest paired IR index is the 2021-2023 temporal set."
    )

    # Image-count inverse-square-root weighting is based on images, not independent storms.
    train_counts = primary[primary["split"].eq("train")]["derived_category"].value_counts()
    n_train = int((primary["split"] == "train").sum())
    raw = {c: min(3.0, (n_train / (len(CATS) * max(int(train_counts.get(c, 0)), 1))) ** 0.5) for c in CATS}
    weighted_mean = sum(int(train_counts.get(c, 0)) * raw[c] for c in CATS) / n_train
    audit["training_category_weights_reconstructed"] = {
        "formula": "min(3, sqrt(n_train/(7*category_image_count))) then normalize to image-weighted mean 1 and cap at 3",
        "train_category_image_counts": {c: int(train_counts.get(c, 0)) for c in CATS},
        "weights": {c: float(min(3.0, raw[c] / weighted_mean)) for c in CATS},
        "note": "Loss weighting changes per-image influence; it does not add rare-category storms or independent examples.",
    }

    # Held-out validation errors by actual category from fresh predictions only.
    predictions = pd.read_csv(OUT / "raw_prediction_reproduction.csv")
    val = predictions[(predictions["partition"] == "2018_2020_val_storm_heldout") & predictions["model"].isin(["A", "C"])].copy()
    val["error"] = val["predicted_wind_kt"] - val["actual_wind_kt"]
    val_rows = []
    for c in CATS:
        part = val[val["actual_category"].eq(c)]
        if part.empty:
            continue
        for model_id, group in part.groupby("model"):
            val_rows.append({
                "category": c,
                "model": model_id,
                "images": int(len(group)),
                "storms": int(group["sid"].nunique()),
                "mae_kt": float(np.abs(group["error"]).mean()),
                "rmse_kt": float(np.sqrt(np.mean(group["error"] ** 2))),
                "mean_bias_kt": float(group["error"].mean()),
                "raw_negative_predictions": int((group["predicted_wind_kt"] < 0).sum()),
            })
    pd.DataFrame(val_rows).to_csv(OUT / "validation_error_by_category.csv", index=False)

    report = {
        "category_definition": "TD <34 kt; TS 34-63; C1 64-82; C2 83-95; C3 96-112; C4 113-136; C5 >=137; recomputed using data/build_dataset.py::category.",
        "primary_category_counts_by_split": {name: {row["category"]: {"images": row["images"], "storms": row["storms"]} for row in all_dist if row["dataset"] == "original_2018_2020" and row["split"] == name} for name in ["train", "val", "test"]},
        "quality_and_provenance": audit,
        "interpretation": [
            "C4/C5 are severely sparse by image and storm in this small 45-storm sample; C5 has only 2 training images from one storm.",
            "Internal category labels exactly agree with category thresholds applied to wind_kt, but this does not independently validate IBTrACS wind estimates.",
            "No indexed IR image or label pair from 2024+ was found; local raw IBTrACS track rows do extend into 2024+.",
        ],
    }
    (OUT / "category_data_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["primary_category_counts_by_split"], indent=2))
    print(json.dumps({k: v for k, v in audit.items() if "2024_plus" in k or k == "cross_dataset_storm_overlap_original_vs_temporal"}, indent=2))
    print(f"Saved category table: {OUT / 'category_distribution_by_split.csv'}")
    print(f"Saved quality audit: {OUT / 'category_data_audit.json'}")
    print(f"Saved validation category errors: {OUT / 'validation_error_by_category.csv'}")


if __name__ == "__main__":
    main()
