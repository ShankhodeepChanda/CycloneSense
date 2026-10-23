# CycloneSense negative prediction audit

Rows: 3304; raw negatives: 50; non-finite outputs: 0.

## Findings

- Model A and C use a one-value unconstrained ResNet18 regression head trained on wind_kt/100; inference multiplies by 100.
- Repeated predictions and existing evaluation transforms were compared on negative and normal validation and temporal examples.
- Results are consistent with the unconstrained regression head. No target-scale, preprocessing, or checkpoint-loading mismatch was found in the executed checks.
- Zero clipping is appropriate as a display safeguard for nonnegative wind. It leaves category results unchanged because negative winds already map to TD; clipping does not repair the learned regression function.
- Raw predictions and raw metrics remain separate. Previously examined test rows were included only for this diagnostic, not for tuning or model selection.

## Negative predictions by partition

| Model | Partition | Images | Negative | Percent | Median neg. (kt) | Minimum (kt) | < -5 | < -10 | < -20 | Nonfinite |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2018_2020_test_previously_saved | 334 | 3 | 0.90% | -2.831814765930176 | -12.165169715881348 | 1 | 1 | 0 | 0 |
| A | 2018_2020_train_in_sample | 487 | 9 | 1.85% | -7.369033813476562 | -19.23912048339844 | 5 | 4 | 0 | 0 |
| A | 2018_2020_val_storm_heldout | 319 | 3 | 0.94% | -7.106749534606934 | -14.284605979919434 | 2 | 1 | 0 | 0 |
| A | 2021_2023_temporal_holdout | 512 | 3 | 0.59% | -4.1141676902771 | -5.48913049697876 | 1 | 0 | 0 | 0 |
| C | 2018_2020_test_previously_saved | 334 | 7 | 2.10% | -8.482599258422852 | -16.866209030151367 | 4 | 3 | 0 | 0 |
| C | 2018_2020_train_in_sample | 487 | 9 | 1.85% | -4.70826530456543 | -18.307523727416992 | 4 | 3 | 0 | 0 |
| C | 2018_2020_val_storm_heldout | 319 | 7 | 2.19% | -6.942971229553223 | -15.107064247131348 | 5 | 3 | 0 | 0 |
| C | 2021_2023_temporal_holdout | 512 | 9 | 1.76% | -3.596697330474853 | -11.468172073364258 | 4 | 1 | 0 | 0 |

## Raw versus clipped metrics

| Model | Partition | Raw MAE | Clip MAE | Raw RMSE | Clip RMSE | Raw bias | Clip bias | Raw cat. acc. | Clip cat. acc. | Changed | Mean correction (kt) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2018_2020_test_previously_saved | 16.424 | 16.377 | 20.864 | 20.815 | 7.821 | 7.867 | 0.476 | 0.476 | 3 | 5.199 |
| A | 2018_2020_train_in_sample | 13.995 | 13.850 | 17.210 | 16.952 | 4.337 | 4.482 | 0.503 | 0.503 | 9 | 7.847 |
| A | 2018_2020_val_storm_heldout | 17.313 | 17.241 | 22.978 | 22.876 | 1.014 | 1.086 | 0.448 | 0.448 | 3 | 7.665 |
| A | 2021_2023_temporal_holdout | 17.212 | 17.192 | 22.066 | 22.034 | 3.899 | 3.919 | 0.416 | 0.416 | 3 | 3.413 |
| C | 2018_2020_test_previously_saved | 18.932 | 18.734 | 24.982 | 24.711 | 8.937 | 9.135 | 0.458 | 0.458 | 7 | 9.447 |
| C | 2018_2020_train_in_sample | 16.756 | 16.617 | 21.463 | 21.274 | 5.351 | 5.490 | 0.466 | 0.466 | 9 | 7.507 |
| C | 2018_2020_val_storm_heldout | 20.831 | 20.640 | 26.917 | 26.556 | 0.677 | 0.867 | 0.411 | 0.411 | 7 | 8.685 |
| C | 2021_2023_temporal_holdout | 20.432 | 20.351 | 25.797 | 25.697 | 0.866 | 0.947 | 0.408 | 0.408 | 9 | 4.615 |

For actual wind y >= 0 and a raw prediction p < 0, |0-y| <= |p-y|; clipping cannot increase that row's absolute error. Bias moves upward. The category conversion already maps every value below 34 kt to TD.

## A/C negative-image overlap

| Partition | Aligned images | Both negative | A only | C only | Union |
|---|---:|---:|---:|---:|---:|
| 2018_2020_train_in_sample | 487 | 1 | 8 | 8 | 17 |
| 2018_2020_val_storm_heldout | 319 | 1 | 2 | 6 | 9 |
| 2018_2020_test_previously_saved | 334 | 0 | 3 | 7 | 10 |
| 2021_2023_temporal_holdout | 512 | 1 | 2 | 8 | 11 |

## Reproducibility

Repeated checks: 8; each selected validation or temporal example was run three times per model. Detailed results are in reproducibility_checks.csv.

## Limits

- 2018-2020 test predictions were read from the existing saved A/C prediction CSV, not regenerated here.
- No per-batch historical training-loss trace exists, so original training losses cannot be exhaustively checked.
- Training rows are in-sample; validation, temporal, and test comparisons are diagnostic only.

## Artifacts

- negative_cases.csv: all negative rows, normalized output, absolute error.
- partition_summary.csv: rates, severity counts, median, minimum, and non-finite counts.
- model_a_model_c_alignment.csv: path joins with target, storm, and timestamp checks.
- raw_vs_clipped_metrics.csv and raw_vs_clipped_predictions.csv: separate raw and display-clipped metrics and values.
- reproducibility_checks.csv: repeated inference and evaluation-pipeline comparisons.
- summary.json: machine-readable report and checkpoint hashes.

Checkpoint hashes were identical before and after. Existing checkpoints, metrics, and prediction files were not written.
