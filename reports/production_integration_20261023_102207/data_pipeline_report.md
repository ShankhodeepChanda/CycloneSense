# 2024+ satellite/track pipeline report

## Local inputs and outcome

The local source cache is `data/raw/ibtracs.since1980.list.v04r01.csv` (IBTrACS v04r01). Its SHA-256 is `00f30d316aff4e031d1bf657e0ba2893b7e779135be7ec44ef773325dff347e4`. The existing builder selected 3,262 positive-wind, six-hourly main-track records for 2024–2025 across 121 SIDs. It produced 1,612 Terra/Aqua nearest-pass IR job candidates before storm-overlap filtering, representing 120 candidate SIDs.

All 120 candidate SIDs occur in existing local dataset track/index tables with train/validation/test assignments. They were excluded by the strict leakage rule, leaving zero eligible storm jobs and an empty candidate manifest. All five discovered image indexes have zero IR rows at or after 2024. No satellite request was made, and no images or model metrics were fabricated.

The exact run evidence is in `candidate_data_final/readiness.json` and `evaluation_verified3/temporal_evaluation_readiness.json`. The evaluator also requires a paired `tracks.csv` and verifies exact SID/time, USA_WIND, coordinates, storm split, source URL/hash, image hash, image dimensions, and image validity before loading Model A. Evaluation did not invoke Model A because there were no valid independent pairs.

## Implemented pipeline

`scripts/prepare_temporal_2024plus.py`:

1. Reads the local cache only; there is no automatic track download.
2. Reuses `data.build_dataset.load_tracks` and `plan_chips` for six-hourly main fixes, the project's `USA_WIND` target choice, nearest Terra/Aqua daytime pass within ±3 hours, and the existing 10-degree chip geometry.
3. Normalizes SIDs and scans every local `images_index.csv` and `tracks.csv`; overlapping storm IDs are excluded before manifest creation.
4. Writes source URL/version, cache SHA-256, exact UTC time, pass date, sensor/layer, target/category, offset, `USA_AGENCY`, `USA_STATUS`, WMO fields, `IFLAG`, and `TRACK_TYPE` into the manifest.
5. Does not contact NASA by default. An explicit `--fetch --max-requests N` is required to retrieve IR chips, with a hard maximum of 500 requests per run and a new output directory. Retrieved images are SHA-256 recorded; the builder enforces 256×256 output and valid fraction ≥0.6.

`scripts/evaluate_temporal_2024plus.py`:

- Refuses existing output paths and checks all local SID tables again, excluding any overlap.
- Requires 2024+ IR rows and a paired `tracks.csv`; validates unique SID/time/satellite and image paths, exact storm/time/wind/coordinate joins, temporal-only track split, source agency/status provenance, IBTrACS source URL and SHA-256 (rechecking the source file when its recorded path is available), image SHA-256, label/category consistency, positive wind, ±3-hour offset, ≥0.6 valid pixels, and 256×256 readable image dimensions.
- Refuses to evaluate if the frozen Model A SHA-256 changes. Defaults to CPU for repeatability; the device and transform are recorded.
- Saves raw predictions and separate nonnegative values; raw and application MAE/RMSE/bias are separate. It also reports category and within-one accuracy, per-category/per-storm/year/satellite performance, negative raw output count, and a constant training-mean baseline computed only from the existing IR training split.
- Writes readiness blockers without model inference or metrics when valid pairs do not exist.

Reproduce the local dry run and a later bounded fetch/evaluation with the commands in README. Do not tune on the resulting temporal holdout.

## Provenance, access, and label alignment

- **IBTrACS v04r01:** NCEI describes a public global archive assembled from multiple agencies, with voluntary citation and an open research use policy that follows WMO Resolution 40 guidance for commercial use. The source is updated regularly and includes provisional/reanalysis distinctions; the local cache does not record its original retrieval date. Preserve the file hash and record the date of any future fresh retrieval. See [NCEI IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) and the [v04r01 technical documentation](https://www.ncei.noaa.gov/sites/default/files/2025-04/IBTrACS_version4r01_Technical_Details.pdf).
- **NASA GIBS / MODIS:** GIBS publishes public standards-based WMS/WMTS visualization services. The current builder retrieves daily Terra/Aqua Band 31 rendered imagery, not raw calibrated radiance. NASA asks for attribution and product-specific restriction checks; GIBS visualization access does not substitute for checking the underlying product terms. See [GIBS access basics](https://nasa-gibs.github.io/gibs-api-docs/access-basics/) and [NASA Earthdata use/citation guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy).
- **Label alignment:** retain `USA_AGENCY`, `USA_STATUS`, `WMO_WIND`, `WMO_AGENCY`, `IFLAG`, and `TRACK_TYPE`. Do not treat all agency wind averages as interchangeable or substitute WMO winds without a separately justified target definition. Review provisional status, season finalization, label continuity, exact SID/time joins, category thresholds, timestamp offset, pixel coverage, duplicate chip hashes, and image/sensor domain before evaluation.
- **Alternative cross-checks:** NOAA/NHC HURDAT2 supplies six-hourly best-track wind/position data for the Atlantic and Northeast/North-Central Pacific, currently through 2025; it is basin-limited and should be matched by storm/time rather than appended as duplicate samples. Check NHC terms and revision status before redistribution. [NHC data archive](https://www.nhc.noaa.gov/data/). JTWC historical best tracks are a useful basin-specific cross-check; its own report documents that not all archive fields are quality-controlled and that final QC can lag. Review current access and use terms before redistribution. [JTWC best-track report](https://www.metoc.navy.mil/jtwc/products/best-tracks/JTWC_TC_Best_Tracks_NRL_Report.pdf).

The NHC/JTWC sources can improve independent label review or provide additional unique storms where compatible. They must not be silently mixed into the existing `USA_WIND` target. No source data were downloaded or added to the repository during this run.
