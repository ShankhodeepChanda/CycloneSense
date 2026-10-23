# CycloneSense production readiness report

**Date:** 2026-10-23  
**Production candidate preserved:** `models/multiyear45_resnet18_ir_weighted.pt`  
**Model A SHA-256:** `3ac72bfbd2675e2aa79ce131aa6eef138a4867dfb5ca0063bb4ac5b19068d6d8`

## Inference safeguard

The regression head is an unconstrained scalar trained on `wind_kt / 100`; inference multiplies by 100. Negative raw outputs are therefore possible and do not by themselves prove a scaling or preprocessing defect. The new [`ml/wind_inference.py`](../../ml/wind_inference.py) keeps raw knots and separate nonnegative application values; it raises on non-finite output. Both existing checkpoints loaded strictly. The safeguard reproduced 8 negative validation/temporal examples on the full checkpoints within 0.001 kt and kept each raw negative while returning zero for the application field. It also verified the separate safeguard values over 3304 rows in the pre-existing diagnostic CSV. Saved test rows were not re-inferred or used for tuning.

The website/API currently does not load Model A: `/predict/pattern` is a feature heuristic and `/predict/forecast` uses the separate `CycloneSenseEngine`. The new boundary is ready for explicit future wiring, not a claim that the current web endpoint was changed. Details are in [`docs/INFERENCE.md`](../../docs/INFERENCE.md).

## Category distribution and data quality

Fresh row-level counts below use all indexed IR satellites and count independent SIDs once per category.

| Split | Category | IR images | Storms |
|---|---:|---:|---:|
| train | TD | 187 | 15 |
| train | TS | 171 | 14 |
| train | C1 | 46 | 12 |
| train | C2 | 32 | 10 |
| train | C3 | 30 | 8 |
| train | C4 | 19 | 7 |
| train | C5 | 2 | 1 |
| val | TD | 114 | 14 |
| val | TS | 101 | 15 |
| val | C1 | 34 | 11 |
| val | C2 | 19 | 8 |
| val | C3 | 19 | 8 |
| val | C4 | 27 | 7 |
| val | C5 | 5 | 3 |
| test | TD | 103 | 14 |
| test | TS | 145 | 15 |
| test | C1 | 40 | 11 |
| test | C2 | 20 | 8 |
| test | C3 | 10 | 6 |
| test | C4 | 9 | 3 |
| test | C5 | 7 | 1 |

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
