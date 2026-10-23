# CycloneSense production candidate audit

**Date:** 2026-10-22  
**Frozen candidate:** `models/multiyear45_resnet18_ir_weighted.pt`  
**SHA-256:** `3ac72bfbd2675e2aa79ce131aa6eef138a4867dfb5ca0063bb4ac5b19068d6d8`

No production model was retrained or changed. The original test split was not used for tuning or new scoring; its negative counts below are read from a prediction CSV saved before this audit.

## Negative predictions

Training divides `wind_kt` by 100; evaluation multiplies the linear regression output by 100. Both checkpoints have the expected IR/target metadata, load strictly, and contain finite parameters. The one-output `Linear` head has no nonnegative activation, so negative wind values are possible by design. The training/evaluation image paths agree on RGB, resize to 224, tensor conversion, and ImageNet normalization. Model A’s fresh train/validation predictions match its saved CSV to floating-point precision; temporal predictions match within 0.0005 kt rounding. All 3,304 A/C prediction rows were finite.

| Partition | Model A | Model C |
| --- | ---: | ---: |
| 2018–2020 train, in-sample | 9 / 487 | 9 / 487 |
| 2018–2020 validation | 3 / 319 | 7 / 319 |
| 2018–2020 test, prior saved CSV only | 3 / 334 | 7 / 334 |
| 2021–2023 temporal holdout | 3 / 512 | 9 / 512 |

Examples: A predicted −7.11 kt for IRIS (actual 30 kt, 2018-03-24) and −1.60 kt for WALAKA (120 kt, 2018-10-03); C predicted −11.62 kt for WALAKA and −15.11 kt for FAXAI (actual 35 kt, 2019-09-04). Every negative row is in `negative_prediction_diagnostics.csv`. For app display, use a separate `max(raw_prediction_kt, 0)` field; retain the raw value and use it for research metrics.

The saved metric files contain no per-batch training-loss history, so historical loss finiteness cannot be verified. Saved validation metrics and the reproduced inference outputs are finite. `ml/inference.py` is a separate morphology classifier, not the wind regressor.

## C4/C5 data

| Split | C4 images / storms | C5 images / storms |
| --- | ---: | ---: |
| Train | 19 / 7 | 2 / 1 |
| Validation | 27 / 7 | 5 / 3 |
| Test | 9 / 3 | 7 / 1 |

Training weights are about 2.25 for C4 and capped at 3.0 for C5, but cannot add independent storms. Validation MAE is 35.69 kt (A) / 29.17 kt (C) for C4, and 32.36 / 39.51 kt for C5. The C5 validation sample is only five images from three storms, so these estimates are uncertain.

Internal checks found no missing images/targets, duplicate paths, storm leakage, category-threshold mismatches, or differences between indexed winds and source-track winds at matched SID/time. All indexed labels have `valid_fraction >=0.663` and satellite offsets <=3 h. This confirms internal consistency, not independent meteorological label accuracy. The processed data omit row-level `USA_AGENCY`; preserving it would help audit best-track labels, whose agency conventions differ ([NOAA IBTrACS overview](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C01552/html), [column docs](https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf)).

The local raw IBTrACS cache has 2024+ tracks, including positive-wind main-track records in 2024 and 2025. No repository image index contains paired IR chips after 2023; no new data were downloaded. The protocol proposes continuing NASA GIBS MODIS Band 31 for compatibility ([Terra](https://gibs.earthdata.nasa.gov/layer-metadata/v1.0/MODIS_Terra_Brightness_Temp_Band31_Day.json), [Aqua](https://gibs.earthdata.nasa.gov/layer-metadata/v1.0/MODIS_Aqua_Brightness_Temp_Band31_Day.json)); NOAA GOES is open but region/sensor specific ([data listing](https://registry.opendata.aws/noaa-goes/)); EUMETSAT archive access is product-licensed ([policy](https://www.eumetsat.int/legal-framework/data-policy)).

## Protocol and checks

[`temporal_evaluation_protocol.md`](temporal_evaluation_protocol.md) freezes Model A and preprocessing, keeps 2024+ storms SID-disjoint and sealed, defines the fixed training-mean baseline, preserves raw predictions, and specifies per-storm/per-category metrics. It also flags the existing temporal builder’s top-five-by-job-count selection for a label-independent future manifest. Any future training should keep the existing random-resized-crop and ±30° rotation augmentation enabled.

Compilation passed for 16 Python files. Relevant tests passed: **12 passed**. The training smoke test used a temporary synthetic fixture and did not touch either checkpoint. `git diff --check` is clean; Git status shows only this new report directory as untracked.

## Artifacts

- `negative_prediction_diagnostics.csv`, `negative_prediction_counts.csv`, `negative_prediction_summary.json`, `raw_prediction_reproduction.csv`
- `category_distribution_by_split.csv`, `category_data_audit.json`, `validation_error_by_category.csv`
- `temporal_evaluation_protocol.md`
- Reproduction scripts: `reproduce_raw_predictions.py`, `audit_categories.py`
