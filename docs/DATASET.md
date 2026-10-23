# Dataset

Built by `data/build_dataset.py` (see README section 2). Sources: NOAA IBTrACS v04r01 (labels), NASA GIBS
MODIS Terra/Aqua (images). Split by storm ID, 70/15/15, deterministic (SHA-1 of the storm ID).

`tracks.csv`: sid, name, season, basin, time, lat, lon, wind_kt (USA 1-min), pres_hpa, wmo_wind_kt,
wmo_pres_hpa, dist2land_km, category (TD/TS/C1-C5), dlat, dlon, dwind (6 h), dwind_24h, ri_24h
(1 if wind gain >= 30 kt in 24 h, empty if unknown), split.

`images_index.csv`: image_path, channel (truecolor/ir), satellite, layer, valid_fraction, chip_deg, plus the
track fields above and offset_h (hours between the track time and the satellite pass).

## 2024+ temporal candidates

`scripts/prepare_temporal_2024plus.py` reads the local IBTrACS v04r01 cache and creates an IR candidate manifest using the existing nearest-pass and chip rules. It compares normalized SIDs against every local image index and track table, excludes overlaps, and retains IBTrACS agency/status provenance. The target follows the existing builder's `USA_WIND` choice; agency averaging and provisional status are not harmonized, so both must be reviewed before evaluation. It does not fetch imagery by default. To explicitly retrieve a bounded sample, pass `--fetch --max-requests N` (1–500) and a new `--out-dir`; the output records image hashes and a readiness report. It never modifies existing train/validation/test datasets.

`scripts/evaluate_temporal_2024plus.py` takes a separately prepared output directory. It excludes any SID overlapping local dataset indexes/tracks, validates timestamp, category, offset, pixel coverage, image dimensions/content/hash, checks the frozen Model A digest, and then computes raw and separately clipped metrics with per-storm/per-category reporting. If no paired images exist it emits a readiness JSON and no metrics. See [API.md](API.md) for the unrelated wind inference endpoint contract and README for runnable commands.
