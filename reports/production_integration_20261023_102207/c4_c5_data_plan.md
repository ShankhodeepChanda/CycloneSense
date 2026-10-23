# C4/C5 data coverage and acquisition plan

## Verified IR image and storm counts

Counts were recomputed from the current `kaggle_multiyear45/images_index.csv`, restricting to IR and deriving category again from `wind_kt`.

| Split | Category | Images | Independent storms |
|---|---|---:|---:|
| train | C4 | 19 | 7 |
| train | C5 | 2 | 1 |
| validation | C4 | 27 | 7 |
| validation | C5 | 5 | 3 |
| test | C4 | 9 | 3 |
| test | C5 | 7 | 1 |

Full distribution: [`category_distribution_audit.csv`](category_distribution_audit.csv).

## What the evidence supports

Training has very few high-category images and storms, especially C5; estimates from validation/test also involve few independent storms. That makes category-specific results unstable and supports limited coverage/class imbalance as an observable concern. The historic readiness audit found no category/target mismatches, invalid timestamps, missing or corrupt indexed images, duplicate paths, split SID overlap, exact SID/time join failures, target-to-track mismatches, or image coverage/offset violations in the audited 2018–2020 data. Those checks establish internal consistency, not independent truth of the underlying wind estimates. This audit cannot establish that data imbalance caused model underperformance.

## Prioritized acquisition and validation

1. **Add independent high-wind storms first.** Build a candidate inventory from completed seasons and seek distinct C4/C5 SIDs, basins, and storm structures. Count SIDs, not just frames. Do not create the appearance of new C5 data by oversampling the current two images.
2. **Keep the current image domain.** Prefer Terra/Aqua daily MODIS Band 31 GIBS layers to preserve the model's palette-rendered IR input domain. Check actual date/layer availability before requesting; reject invalid images using the current offset, 256×256, and valid-pixel gates. GIBS imagery is a rendered browse layer, not calibrated radiance.
3. **Keep targets comparable.** Continue the existing `USA_WIND` field only when its source agency/status supports the frozen target interpretation. Preserve `USA_AGENCY`, `USA_STATUS`, `IFLAG`, `WMO_WIND`, and `WMO_AGENCY`; validate high-wind labels against the responsible agency's post-season best track. Do not convert/swap averaging periods silently.
4. **Use external best tracks as cross-checks or explicitly harmonized sources.** HURDAT2 is a useful six-hourly, 1-minute-wind source for Atlantic and selected Pacific basins, not global coverage. JTWC can add Pacific/Indian Ocean history, but inspect record-level quality and access terms. If labels from these sources are used as targets, document agency conventions and evaluate source-stratified; do not duplicate the same storm/time as independent evidence.
5. **Rebuild splits by storm ID before any future training.** Hash/assign whole SIDs, check for SID and image-content duplicates across splits, and reserve the new temporal period. Retain the original augmentation strategy if future experiments are authorized. Do not touch Model A in this acquisition task.

IBTrACS is publicly available for research with source citations and WMO Resolution 40 guidance for commercial usage; NASA asks for acknowledgement and product-specific terms review. See the source links and detailed caveats in [`data_pipeline_report.md`](data_pipeline_report.md). No downloads, dataset mutation, or training occurred here.
