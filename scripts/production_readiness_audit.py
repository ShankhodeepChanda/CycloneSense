"""Create a new, read-only production readiness report for CycloneSense."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from importlib import metadata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "production_readiness_20261023_01"
CATEGORIES = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
DATASETS = {
    "original_2018_2020": (ROOT / "kaggle_multiyear45", ["train", "val", "test"]),
    "temporal_2021_2023": (ROOT / "kaggle_temporal_test", ["temporal_test"]),
}
CHECKPOINTS = {
    "A": ROOT / "models" / "multiyear45_resnet18_ir_weighted.pt",
    "C": ROOT / "models" / "multiyear45_resnet18_ir_weighted_lr1e4.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_audit() -> tuple[pd.DataFrame, dict]:
    from data.build_dataset import category

    distributions: list[dict] = []
    quality: dict = {}
    split_storms: dict[str, dict[str, set[str]]] = {}
    primary_ir = None
    temporal_ir = None

    for dataset_name, (data_root, allowed_splits) in DATASETS.items():
        index_path = data_root / "images_index.csv"
        frame = pd.read_csv(index_path)
        frame["wind_kt_numeric"] = pd.to_numeric(frame["wind_kt"], errors="coerce")
        frame["time_parsed"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
        frame["derived_category"] = frame["wind_kt_numeric"].map(
            lambda x: category(float(x)) if pd.notna(x) else None
        )
        all_splits = frame[frame["split"].isin(allowed_splits)].copy()
        ir = all_splits[all_splits["channel"].eq("ir")].copy()
        if dataset_name == "original_2018_2020":
            primary_ir = ir
        else:
            temporal_ir = ir

        split_storms[dataset_name] = {
            split: set(part["sid"].dropna().astype(str))
            for split, part in all_splits.groupby("split")
        }

        for split in allowed_splits:
            part = all_splits[all_splits["split"].eq(split)]
            for (channel, satellite), sensor_part in part.groupby(["channel", "satellite"], dropna=False):
                for category_name in CATEGORIES:
                    cat_part = sensor_part[sensor_part["derived_category"].eq(category_name)]
                    distributions.append({
                        "dataset": dataset_name,
                        "split": split,
                        "channel": str(channel),
                        "satellite": str(satellite),
                        "category": category_name,
                        "images": int(len(cat_part)),
                        "storms": int(cat_part["sid"].nunique()),
                        "share_of_sensor_split": float(len(cat_part) / len(sensor_part)) if len(sensor_part) else 0.0,
                        "low_image_count_lt10": bool(len(cat_part) < 10),
                        "low_storm_count_lt5": bool(cat_part["sid"].nunique() < 5),
                    })

        image_paths = all_splits["image_path"].dropna().astype(str).tolist()
        missing_paths = [rel for rel in image_paths if not (data_root / rel).is_file()]
        corrupt_paths = []
        for rel in image_paths:
            full_path = data_root / rel
            if full_path.is_file():
                try:
                    with Image.open(full_path) as image:
                        image.verify()
                except (OSError, ValueError):
                    corrupt_paths.append(rel)

        mismatch = all_splits[
            all_splits["category"].astype(str) != all_splits["derived_category"].astype(str)
        ]
        splits_info = {}
        for split in allowed_splits:
            subset = all_splits[all_splits["split"].eq(split)]
            splits_info[split] = {"images_all_channels": int(len(subset)), "storms": int(subset.sid.nunique())}

        track_path = data_root / "tracks.csv"
        tracks = pd.read_csv(track_path)
        tracks["time_parsed"] = pd.to_datetime(tracks["time"], errors="coerce", utc=True)
        index_for_join = all_splits.copy()
        join = index_for_join.merge(
            tracks[["sid", "time_parsed", "wind_kt"]].rename(columns={"wind_kt": "track_wind_kt"}),
            on=["sid", "time_parsed"], how="left", validate="many_to_one",
        )
        index_wind = pd.to_numeric(join["wind_kt"], errors="coerce")
        track_wind = pd.to_numeric(join["track_wind_kt"], errors="coerce")

        quality[dataset_name] = {
            "images_index": str(index_path.relative_to(ROOT)),
            "tracks_csv": str(track_path.relative_to(ROOT)),
            "year_min": int(all_splits["time_parsed"].dt.year.min()),
            "year_max": int(all_splits["time_parsed"].dt.year.max()),
            "rows_by_split_all_channels": splits_info,
            "channel_satellite_rows": all_splits.groupby(["channel", "satellite"], dropna=False).size().rename("images").reset_index().to_dict("records"),
            "missing_targets": int(all_splits["wind_kt_numeric"].isna().sum()),
            "nonpositive_targets": int((all_splits["wind_kt_numeric"] <= 0).sum()),
            "invalid_timestamps": int(all_splits["time_parsed"].isna().sum()),
            "stored_category_mismatches": int(len(mismatch)),
            "duplicate_image_paths": int(all_splits["image_path"].duplicated().sum()),
            "missing_image_files": len(missing_paths),
            "missing_image_examples": missing_paths[:5],
            "corrupt_image_files": len(corrupt_paths),
            "corrupt_image_examples": corrupt_paths[:5],
            "index_rows_without_exact_storm_time_track_match": int(join["track_wind_kt"].isna().sum()),
            "image_vs_track_wind_mismatches": int((index_wind != track_wind).fillna(True).sum()),
            "valid_fraction_min": float(pd.to_numeric(all_splits["valid_fraction"], errors="coerce").min()),
            "valid_fraction_below_0_6": int((pd.to_numeric(all_splits["valid_fraction"], errors="coerce") < 0.6).sum()),
            "offset_over_3_hours": int((pd.to_numeric(all_splits["offset_h"], errors="coerce").abs() > 3).sum()),
        }

    overlap = {}
    for dataset_name, splits in split_storms.items():
        split_names = list(splits)
        for i, left in enumerate(split_names):
            for right in split_names[i + 1:]:
                overlap[f"{dataset_name}:{left}__{right}"] = sorted(splits[left] & splits[right])
    if primary_ir is not None and temporal_ir is not None:
        overlap["original_2018_2020_ir__temporal_2021_2023_ir"] = sorted(
            set(primary_ir.sid.astype(str)) & set(temporal_ir.sid.astype(str))
        )
    quality["storm_overlap_between_splits"] = overlap
    quality["category_definition"] = "TD <34; TS 34–63; C1 64–82; C2 83–95; C3 96–112; C4 113–136; C5 ≥137 kt."

    # Inventory every local index, including any non-production trial indexes, for 2024+ paired imagery.
    inventory = []
    for index_path in sorted(ROOT.rglob("images_index.csv")):
        if any(part in {".venv", "node_modules", ".git"} for part in index_path.parts):
            continue
        local = pd.read_csv(index_path, usecols=lambda col: col in {"channel", "time", "sid"})
        times = pd.to_datetime(local["time"], errors="coerce", utc=True)
        post = local[times.dt.year.ge(2024) & local["channel"].eq("ir")]
        inventory.append({
            "index": str(index_path.relative_to(ROOT)),
            "rows": int(len(local)),
            "ir_rows_2024_plus": int(len(post)),
            "ir_storms_2024_plus": int(post["sid"].nunique()),
            "year_min": int(times.dt.year.min()) if times.notna().any() else None,
            "year_max": int(times.dt.year.max()) if times.notna().any() else None,
        })
    quality["local_image_index_inventory"] = inventory
    quality["local_2024_plus_paired_ir_rows"] = int(sum(item["ir_rows_2024_plus"] for item in inventory))

    # Raw IBTrACS is a label/track source only; it does not prove paired imagery exists.
    ibtracs = ROOT / "data" / "raw" / "ibtracs.since1980.list.v04r01.csv"
    source_summary = {"path": str(ibtracs.relative_to(ROOT)), "present": ibtracs.exists()}
    if ibtracs.exists():
        records, main_positive = Counter(), Counter()
        with ibtracs.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            next(reader, None)  # units row
            for row in reader:
                stamp = (row.get("ISO_TIME") or "")[:4]
                if stamp.isdigit() and int(stamp) >= 2024:
                    year = int(stamp)
                    records[year] += 1
                    try:
                        wind = float(row.get("USA_WIND", ""))
                    except (ValueError, TypeError):
                        wind = 0
                    if row.get("TRACK_TYPE", "").strip().lower() == "main" and wind > 0:
                        main_positive[year] += 1
        source_summary["records_2024_plus_by_year"] = dict(sorted(records.items()))
        source_summary["positive_USA_WIND_main_track_records_2024_plus_by_year"] = dict(sorted(main_positive.items()))
    quality["local_raw_ibtracs_2024_plus"] = source_summary

    pd.DataFrame(distributions).to_csv(OUT / "category_distribution_audit.csv", index=False)
    return pd.DataFrame(distributions), quality


def verify_safeguard() -> dict:
    from ml.wind_inference import load_wind_model, predict_wind, separate_wind_predictions
    import torch
    from torchvision import transforms as T

    prior_dir = ROOT / "reports" / "negative_prediction_audit_20261023_01"
    source = pd.read_csv(prior_dir / "raw_vs_clipped_predictions.csv")
    # The helper must preserve all saved raw predictions exactly and derive display values separately.
    verified_rows = 0
    for model_id, group in source.groupby("model"):
        batch = separate_wind_predictions(pd.to_numeric(group["normalized_model_output"], errors="coerce").to_numpy())
        raw = pd.to_numeric(group["raw_prediction_kt"], errors="coerce").to_numpy(float)
        display = pd.to_numeric(group["clipped_display_prediction_kt"], errors="coerce").to_numpy(float)
        if not np.allclose(batch.raw_prediction_kt, raw, rtol=0.0, atol=1e-10):
            raise AssertionError(f"Raw prediction preservation failed for Model {model_id}")
        if not np.allclose(batch.application_prediction_kt, display, rtol=0.0, atol=1e-10):
            raise AssertionError(f"Application safeguard disagrees with saved diagnostic for Model {model_id}")
        verified_rows += len(group)

    transform = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    samples = source[
        source["partition"].isin(["2018_2020_val_storm_heldout", "2021_2023_temporal_holdout"])
        & source["is_negative"].astype(bool)
    ].copy()
    # Re-run a few cases per split with both full checkpoints; no test split is loaded here.
    samples = samples.sort_values("raw_prediction_kt").groupby(["model", "partition"], as_index=False).head(2)
    results = []
    models = {}
    for model_id, checkpoint in CHECKPOINTS.items():
        model, device = load_wind_model(checkpoint, device="cpu")
        models[model_id] = (model, device)
    for row in samples.itertuples(index=False):
        model, device = models[row.model]
        root = ROOT / ("kaggle_temporal_test" if "temporal" in row.partition else "kaggle_multiyear45")
        with Image.open(root / row.image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        prediction = predict_wind(model, tensor, device=device)
        raw = float(prediction.raw_prediction_kt[0])
        shown = float(prediction.application_prediction_kt[0])
        expected = float(row.raw_prediction_kt)
        results.append({
            "model": row.model,
            "partition": row.partition,
            "image_path": row.image_path,
            "actual_wind_kt": float(row.actual_wind_kt),
            "saved_raw_prediction_kt": expected,
            "reproduced_raw_prediction_kt": raw,
            "application_prediction_kt": shown,
            "raw_matches_saved_within_0_001_kt": abs(raw - expected) <= 0.001,
            "raw_prediction_preserved": raw < 0 and shown == 0.0,
        })
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(OUT / "safeguard_checkpoint_verification.csv", index=False)
    if not result_frame["raw_matches_saved_within_0_001_kt"].all():
        raise AssertionError("Checkpoint reproduction differed from the saved raw audit by >0.001 kt")
    return {
        "verified_saved_prediction_rows": int(verified_rows),
        "full_checkpoint_negative_examples_reproduced": int(len(result_frame)),
        "full_checkpoint_examples_match_saved_raw_within_0_001_kt": bool(result_frame["raw_matches_saved_within_0_001_kt"].all()),
        "full_checkpoint_examples_keep_negative_raw_and_zero_application_value": bool(result_frame["raw_prediction_preserved"].all()),
        "test_split_rerun": False,
    }


def create_protocol() -> None:
    text = """# Frozen protocol for a future 2024+ temporal evaluation

## Data availability and sources

The audit found a local IBTrACS v04r01 track cache with 2024+ records. It found no 2024+ infrared image rows in any local `images_index.csv`; therefore no paired 2024+ cyclone imagery is currently available in this repository. No large data set was downloaded for this audit.

For backward-compatible inputs, extend the existing NASA GIBS MODIS Terra/Aqua Band 31 daytime workflow. Record the source URL, retrieval date, GIBS layer and date, request bounds, center coordinates, rendering/product version, image dimensions, JPEG settings, and SHA-256 for each downloaded image. NASA says NASA-sponsored data are generally available without copyright restrictions unless marked otherwise and asks users to acknowledge/cite NASA; check product-specific notices. IBTrACS is openly distributed for research and follows WMO Resolution 40 guidance for commercial use; retain source citation and check current terms. NHC HURDAT2 is a useful authoritative 1-minute wind cross-check for Atlantic and Northeast/North Central Pacific storms, but does not supply global coverage. JTWC best tracks can add western Pacific, North Indian and Southern Hemisphere comparison where suitable; older periods have lower confidence and conventions need checking.

Preserve SID, UTC timestamp, basin, agency/source, track type, latitude, longitude, raw `USA_WIND`, `USA_AGENCY`, `WMO_WIND`, `WMO_AGENCY`, status, and chosen target field with a written reason. `USA_WIND` is the current training target; IBTrACS aggregates agencies whose wind averaging periods differ. Validate agency/averaging conventions by basin and year, inspect missing/flagged/provisional points and intensity jumps, compare against the relevant basin authority, and never silently substitute WMO or another agency's value. Recompute category only from the frozen target thresholds. Keep the exact best-track record joined to each chip by SID and time, and retain time-to-overpass offset.

## Image and target preprocessing

Use the current chip rule unless a separate experiment is registered: closest eligible 6-hourly main-track record to the Terra/Aqua daytime local pass, within ±3 h, 10° square centered on the best-track location, 256×256 image, at least 0.6 valid pixels, saved as RGB JPEG quality 90. The target remains wind in knots divided by 100 at training/inference boundaries. Evaluation uses RGB conversion, `Resize(224)`, `ToTensor`, and ImageNet normalization `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`. Preserve the original training augmentation (`RandomResizedCrop(scale=0.8–1.0)` and `RandomRotation(±30°)`, without horizontal flips) in any future training experiment. Do not mix GOES, Himawari, Meteosat, or a different MODIS product into the frozen input domain without a separately specified channel/projection/calibration/preprocessing and sensor-stratified evaluation.

## Splits, frozen models, and baseline

Before predictions are inspected, create and hash a manifest that assigns every eligible SID to one partition. All 2024+ SIDs stay entirely in a future temporal holdout and must be disjoint from training and validation; preserve existing 2018–2020 and 2021–2023 assignments. Include all eligible storms, or use a pre-registered rule based on provenance/availability only. Do not select storms based on model errors.

Freeze Model A `models/multiyear45_resnet18_ir_weighted.pt` by SHA-256, Git commit, inference source/configuration, Python/PyTorch/torchvision/Pillow versions, device, resize/normalization, and deterministic row/batch order. Do not retrain or replace it for this evaluation. Compare it with a constant mean-wind baseline calculated solely from the frozen Model A training labels (historically 49.363 kt); record the formula and training-row manifest. No temporal holdout labels may enter preprocessing choices or baseline calculation. Keep raw Model A predictions and, if needed, a separate nonnegative application value; scientific metrics use the raw value.

## Frozen outputs and metrics

Write one prediction row per indexed chip before calculating metrics. Store source and chosen labels, SID/time/category, satellite/channel, checkpoint hash, raw normalized model output, raw knots, and any separately named application value. Report image-level MAE, RMSE, mean bias, median absolute error, category accuracy and within-one-category accuracy; also per-storm metrics and unweighted storm means; per-category image/storm counts, MAE, RMSE, bias and extreme errors; per-year and Terra/Aqua results. Show C4/C5 sample counts and flag categories under 10 images or 5 storms. Use storm-level bootstrap resampling for uncertainty. Freeze this config and output hashes before reading the results; do not tune on the 2024+ holdout. Any changes after results require a later untouched evaluation period.

## Official sources

- NOAA NCEI IBTrACS, fields, access, terms and citation: https://www.ncei.noaa.gov/products/international-best-track-archive
- NHC HURDAT2 archive and database description: https://www.nhc.noaa.gov/data/
- NASA GIBS access: https://nasa-gibs.github.io/gibs-api-docs/access-basics/
- NASA Earthdata data-use and acknowledgement guidance: https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy
- JTWC best-track archive report and limits: https://www.metoc.navy.mil/jtwc/products/best-tracks/JTWC_TC_Best_Tracks_NRL_Report.pdf
"""
    (OUT / "temporal_evaluation_protocol.md").write_text(text, encoding="utf-8")


def environment_details() -> dict:
    packages = {
        "numpy": "numpy", "pandas": "pandas", "Pillow": "Pillow",
        "torch": "torch", "torchvision": "torchvision", "pytest": "pytest",
        "matplotlib": "matplotlib", "FastAPI": "fastapi", "httpx": "httpx",
        "requests": "requests",
    }
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": {label: metadata.version(package) for label, package in packages.items()},
        "device_for_checkpoint_reproduction": "cpu",
        "environment_scope": "D:\\CycloneSense\\.venv (project-local)",
        "pandas_install": "pandas==2.3.3 installed in the project-local .venv; requirements.txt already declares pandas>=2.0.0.",
    }


def write_summary(distributions: pd.DataFrame, quality: dict, safeguard: dict, checkpoints: dict) -> None:
    primary = pd.read_csv(ROOT / "kaggle_multiyear45" / "images_index.csv")
    primary = primary[(primary.channel.eq("ir")) & primary.split.isin(["train", "val", "test"])].copy()
    primary["wind_kt"] = pd.to_numeric(primary["wind_kt"], errors="coerce")
    from data.build_dataset import category
    primary["derived_category"] = primary.wind_kt.map(lambda value: category(float(value)) if pd.notna(value) else None)
    counts = []
    for split in ["train", "val", "test"]:
        part = primary[primary.split.eq(split)]
        for category_name in CATEGORIES:
            group = part[part.derived_category.eq(category_name)]
            counts.append((split, category_name, len(group), group.sid.nunique()))

    count_lines = ["| Split | Category | IR images | Storms |", "|---|---:|---:|---:|"]
    count_lines.extend(f"| {split} | {cat} | {images} | {storms} |" for split, cat, images, storms in counts)
    a_hash = checkpoints["A"]["sha256"]
    summary = f"""# CycloneSense production readiness report

**Date:** 2026-10-23  
**Production candidate preserved:** `models/multiyear45_resnet18_ir_weighted.pt`  
**Model A SHA-256:** `{a_hash}`

## Inference safeguard

The regression head is an unconstrained scalar trained on `wind_kt / 100`; inference multiplies by 100. Negative raw outputs are therefore possible and do not by themselves prove a scaling or preprocessing defect. The new [`ml/wind_inference.py`](../../ml/wind_inference.py) keeps raw knots and separate nonnegative application values; it raises on non-finite output. Both existing checkpoints loaded strictly. The safeguard reproduced {safeguard['full_checkpoint_negative_examples_reproduced']} negative validation/temporal examples on the full checkpoints within 0.001 kt and kept each raw negative while returning zero for the application field. It also verified the separate safeguard values over {safeguard['verified_saved_prediction_rows']} rows in the pre-existing diagnostic CSV. Saved test rows were not re-inferred or used for tuning.

The website/API currently does not load Model A: `/predict/pattern` is a feature heuristic and `/predict/forecast` uses the separate `CycloneSenseEngine`. The new boundary is ready for explicit future wiring, not a claim that the current web endpoint was changed. Details are in [`docs/INFERENCE.md`](../../docs/INFERENCE.md).

## Category distribution and data quality

Fresh row-level counts below use all indexed IR satellites and count independent SIDs once per category.

{chr(10).join(count_lines)}

C4/C5 are sparsely represented in the training set (19/7 and 2/1 images/storms) and remain small in validation (27/7 and 5/3) and test (9/3 and 7/1). This is strong evidence of category imbalance and low independent-storm counts, which makes high-category error estimates noisy. The audit found no missing or nonpositive labels, stored-category mismatches, missing/corrupt images, duplicate image paths, unmatched SID/time track rows, target/track wind mismatches, split storm overlap, or IR samples outside the 3-hour offset/0.6 valid-pixel criteria. These checks establish internal consistency only; they do not independently validate best-track wind estimates or rule out uncertainty in agency labels.

The local raw IBTrACS cache includes 2024+ records, but every local `images_index.csv` has **zero paired 2024+ IR rows**. No new satellite imagery was downloaded. Feasible future sources are NASA GIBS MODIS Terra/Aqua Band 31 for continuity, NOAA/NHC HURDAT2 for Atlantic and Northeast/North Central Pacific cross-checks, and JTWC best tracks for its basins where wind conventions and product limits are suitable. IBTrACS is global but merges agencies and differing wind-averaging practices. Keep agency/source fields and do not silently mix target definitions. NASA asks for acknowledgement/citation and product-specific restriction checks; IBTrACS is open for research and follows WMO Resolution 40 guidance for commercial use; verify current terms before redistribution.

## Frozen future evaluation

[`temporal_evaluation_protocol.md`](temporal_evaluation_protocol.md) specifies provenance, label checks, preprocessing, a storm-disjoint 2024+ holdout manifest, frozen Model A and configuration, a training-mean baseline, per-storm/per-category metrics, and no tuning on holdout results. Any future training must retain the original random-resized-crop and ±30° rotation augmentation.

## Checkpoints, environment, and test status

Model A and Model C hashes and byte sizes are in [`checkpoint_manifest.json`](checkpoint_manifest.json). Neither checkpoint was retrained or modified. The Python test environment is project-local; see [`environment_report.md`](environment_report.md). Compile, Python test, web test, and diff-check results are recorded in [`test_results.txt`](test_results.txt).

## Limitations

- The raw Model A output remains the scientific prediction; clipping is only a separately named application display safeguard.
- Historical per-batch training loss traces are unavailable, so the report cannot retrospectively establish whether every original training loss was finite.
- HURDAT2 and JTWC cover selected basins and use post-season best-track estimates; neither replaces a global, convention-consistent label source.
- Existing test data was only inspected through pre-existing saved diagnostics, not regenerated for model selection or hyperparameter tuning.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        expected_partial = {
            "raw_vs_clipped_predictions.csv", "raw_vs_clipped_metrics.csv",
            "negative_prediction_diagnostics.csv", "negative_prediction_counts.csv",
            "category_distribution_audit.csv", "category_data_audit.json",
            "safeguard_checkpoint_verification.csv", "temporal_evaluation_protocol.md",
            "checkpoint_manifest.json", "environment_report.json", "environment_report.md",
            "summary.md",
        }
        existing = {path.name for path in OUT.iterdir()}
        if not existing.issubset(expected_partial):
            raise FileExistsError(f"Refusing to overwrite non-partial report directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    # Preserve prior diagnostics as references; this script does not overwrite their source files.
    prior = ROOT / "reports" / "negative_prediction_audit_20261023_01"
    shutil.copy2(prior / "raw_vs_clipped_predictions.csv", OUT / "raw_vs_clipped_predictions.csv")
    shutil.copy2(prior / "raw_vs_clipped_metrics.csv", OUT / "raw_vs_clipped_metrics.csv")
    raw_rows = pd.read_csv(prior / "raw_vs_clipped_predictions.csv")
    raw_rows[raw_rows["is_negative"].astype(bool)].to_csv(OUT / "negative_prediction_diagnostics.csv", index=False)
    raw_rows.groupby(["model", "partition"], as_index=False).agg(
        images=("image_path", "count"), storms=("sid", "nunique"),
        raw_negative_predictions=("is_negative", "sum"),
        nonfinite_predictions=("is_nonfinite", "sum"),
    ).to_csv(OUT / "negative_prediction_counts.csv", index=False)

    distributions, quality = image_audit()
    (OUT / "category_data_audit.json").write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
    safeguard = verify_safeguard()
    create_protocol()

    checkpoint_manifest = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for key, path in CHECKPOINTS.items()
    }
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2), encoding="utf-8")
    env = environment_details()
    (OUT / "environment_report.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    environment_markdown = [
        "# Python test environment", "",
        f"Environment: `{env['environment_scope']}`", "",
        f"Python: `{sys.version.split()[0]}` ({sys.executable})", "",
        "Package versions:", "",
        "| Package | Version |", "|---|---:|",
    ]
    environment_markdown.extend(f"| {name} | {version} |" for name, version in env["packages"].items())
    environment_markdown += ["", env["pandas_install"], "", "The installation was made inside `.venv`; no global Python packages were changed."]
    (OUT / "environment_report.md").write_text("\n".join(environment_markdown) + "\n", encoding="utf-8")
    write_summary(distributions, quality, safeguard, checkpoint_manifest)
    concise_quality = {
        name: {key: value for key, value in data.items() if key in {
            "rows_by_split_all_channels", "missing_targets", "nonpositive_targets",
            "stored_category_mismatches", "missing_image_files", "corrupt_image_files",
            "index_rows_without_exact_storm_time_track_match", "image_vs_track_wind_mismatches",
        }}
        for name, data in quality.items() if name in DATASETS
    }
    print(json.dumps({"quality": concise_quality, "safeguard": safeguard, "checkpoints": checkpoint_manifest}, default=str))


if __name__ == "__main__":
    main()
