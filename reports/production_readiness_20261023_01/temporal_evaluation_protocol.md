# Frozen protocol for a future 2024+ temporal evaluation

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
