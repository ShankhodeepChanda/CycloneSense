# Storm-level error analysis: weighted learning-rate comparison

**Evaluation:** original 2018–2020 storm-held-out IR test split; 334 images from 15 storms. Predictions are aligned by `image_path`, and actual targets match across all three checkpoints.

This is diagnostic analysis of a test split already examined in prior work. It is not a final unbiased model-selection estimate.

## Overall image-level results

| Model | Experiment | Images | Storms | MAE (kt) | RMSE (kt) | Category accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| A | Original weighted (lr=3e-4) | 334 | 15 | 16.42 | 20.86 | 0.476 |
| B | Weighted, no augmentation | 334 | 15 | 18.15 | 23.59 | 0.428 |
| C | Weighted, lr=1e-4 | 334 | 15 | 18.93 | 24.98 | 0.458 |

## Model C versus Model A by storm

Improvement is defined as MAE(A) − MAE(C), so positive values favor Model C. The stored change column C−A uses the opposite sign.

Model C improved 3 storms, worsened 12, and tied 0. Mean/median storm MAE change C−A: 1.99 / 1.73 kt.

Paired storm bootstrap 95% percentile interval for mean C−A MAE: [0.21, 3.67] kt, resampling 15 storms. This interval is exploratory and unstable with so few storms.

### Five largest improvements for C

| sid | name | images | C improvement (kt) |
| --- | --- | --- | --- |
| 2018250N12170 | MANGKHUT | 22 | 6.23 |
| 2019326S08163 | RITA | 16 | 1.10 |
| 2018227N11145 | SOULIK | 22 | 0.59 |
| 2019243N06136 | LINGLING | 16 | -0.04 |
| 2020244N25146 | HAISHEN | 12 | -0.51 |

### Five largest regressions for C

| sid | name | images | C improvement (kt) |
| --- | --- | --- | --- |
| 2019216N16147 | KROSA | 26 | -8.63 |
| 2018311N07179 | USAGI | 36 | -6.00 |
| 2020291N06141 | SAUDEL | 20 | -5.55 |
| 2018204N15137 | JONGDARI | 26 | -4.10 |
| 2018265N35315 | LESLIE | 43 | -4.03 |

## High-intensity categories

| category | model | images | storms | actual_mean_kt | predicted_mean_kt | mae_kt | rmse_kt | mean_bias_kt | median_absolute_error_kt | maximum_absolute_error_kt | low_sample_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 | A | 10 | 6 | 102.5 | 114.24 | 15.84 | 21.79 | 11.74 | 10.67 | 48.3 | False |
| C3 | B | 10 | 6 | 102.5 | 95.77 | 17.84 | 21.52 | -6.73 | 19.81 | 38.31 | False |
| C3 | C | 10 | 6 | 102.5 | 103.52 | 14.92 | 17.5 | 1.02 | 13.65 | 29.01 | False |
| C4 | A | 9 | 3 | 122.78 | 121.02 | 10.7 | 13.43 | -1.75 | 10.38 | 21.81 | True |
| C4 | B | 9 | 3 | 122.78 | 103.31 | 19.47 | 23.23 | -19.47 | 28.54 | 34.87 | True |
| C4 | C | 9 | 3 | 122.78 | 120.53 | 13.29 | 16.23 | -2.25 | 11.59 | 37.19 | True |
| C5 | A | 7 | 1 | 147.14 | 117.54 | 29.6 | 38.99 | -29.6 | 15.32 | 86.68 | True |
| C5 | B | 7 | 1 | 147.14 | 81.44 | 65.71 | 67.39 | -65.71 | 64.16 | 87.72 | True |
| C5 | C | 7 | 1 | 147.14 | 126.77 | 20.95 | 32.12 | -20.37 | 14.46 | 73.76 | True |

Per-storm C3/C4/C5 comparisons are in `high_category_storm_comparison.csv`; category-specific storm counts and C-versus-A consistency are also in `summary.json`. Categories flagged low-sample have fewer than 10 images or fewer than 5 storms.

## Prediction checks

Negative predictions: {"A": 3, "B": 2, "C": 7}. Errors ≥50 kt and ≥30 kt prediction steps within 48 hours are counted in `summary.json`; individual cases are in `error_cases.csv`.

Plots: `per_storm_mae_A_vs_C.png`, `actual_vs_predicted_A_vs_C.png`, `category_mae_A_B_C.png`, `storm_series_most_improved.png`, and `storm_series_most_regressed.png`.

## Limitations

Only 15 independent storms are in this split, so images are correlated and storm-level estimates are noisy. C3–C5 may contain very few storms. The 2021–2023 temporal-holdout files and train/validation predictions were not used for this test-split comparison. These results do not establish operational forecasting reliability.

## Artifacts

- `aligned_test_predictions.csv` — raw, aligned predictions for all three checkpoints.
- `storm_level_comparison.csv` — storm-by-storm A/B/C metrics and paired changes.
- `category_comparison.csv` — category metrics for all three models.
- `high_category_storm_comparison.csv` — C3/C4/C5 storm-level consistency data.
- `error_cases.csv` — most severe errors, negative predictions, paired wins/losses, and prediction spikes.
- `summary.json` — machine-readable analysis and metric-file cross-checks.