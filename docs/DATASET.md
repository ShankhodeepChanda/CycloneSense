# Dataset

Built by `data/build_dataset.py` (see README section 2). Sources: NOAA IBTrACS v04r01 (labels), NASA GIBS
MODIS Terra/Aqua (images). Split by storm ID, 70/15/15, deterministic (SHA-1 of the storm ID).

`tracks.csv`: sid, name, season, basin, time, lat, lon, wind_kt (USA 1-min), pres_hpa, wmo_wind_kt,
wmo_pres_hpa, dist2land_km, category (TD/TS/C1-C5), dlat, dlon, dwind (6 h), dwind_24h, ri_24h
(1 if wind gain >= 30 kt in 24 h, empty if unknown), split.

`images_index.csv`: image_path, channel (truecolor/ir), satellite, layer, valid_fraction, chip_deg, plus the
track fields above and offset_h (hours between the track time and the satellite pass).
