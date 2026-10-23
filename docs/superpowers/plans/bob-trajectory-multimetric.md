# Plan: Multi-Metric Bay of Bengal Cyclone Path & Intensity Forecasting

Spec: the user's mission brief (Phases 1-4), treated as the binding authority.

## Context (verified against the repo, 2026-10-23)

- `kaggle_trial/tracks.csv` = 9580 IBTrACS rows, 19 cols, `basin` in {WP,SI,EP,SP,NI}.
  **NI = 570 rows / 22 storms** incl. AMPHAN, FANI, MAHA, VAYU, TITLI, NIVAR, BUREVI, GATI,
  NISARGA, MEKUNU, LUBAN, GAJA, PHETHAI, SAGAR, DAYE, PAWAN, HIKAA, KYARR, BULBUL, PABUK.
  MOCHA/REMAL/YAAS/HUDHUD are **not** in the repo data (coverage is 2018-2021).
- Columns usable per step: lat, lon, dlat, dlon, wind_kt, pres_hpa, dist2land_km.
  Nulls in NI: pres_hpa 4, dlat/dlon 20 (first fix of each storm).
- `kaggle_trial/images_index.csv` = 415 chips total, **only 28 for NI** (14 ir + 14 truecolor).
- `models/*.pt` is gitignored and **no checkpoint is present** (Model A's wind .pt is absent too),
  so `/predict/wind` currently 503s. Trained weights must be reproducible from a script.
- torch 2.14.1+**cpu**, python 3.14, no CUDA.
- `public/coast.json` = global MultiLineString, 16070 vertices. Nearest-vertex haversine
  reproduces IBTrACS `dist2land_km` at corr 0.945, median abs err 5.4 km (verified).
- GDACS `geteventlist` returns **only** `Point_Centroid` per storm. The per-event
  `polygons/getgeometry` endpoint returns the real track: `Line_Line_N` 2-point segments
  (chain -> ordered centres), `Point_Polygon_Point_N` polygons whose `polygonlabel` is
  `DD/MM HH:MM UTC` per point, `Poly_Cones`, and event-level `severitydata` only
  (no per-point wind, no pressure anywhere).

## Global Constraints

1. Do not break: NASA GIBS tiles, Open-Meteo city panel, GDACS markers, `/predict/wind`.
2. Degrade honestly: when the trajectory checkpoint is absent the API must still answer,
   labelled as a physics extrapolation, never as a model prediction.
3. Bay of Bengal / NIO focus: 5-25 N, 80-95 E demo storms.
4. Everything verifiable on CPU in minutes. `pytest` and `npm run build` must pass.
5. Never present an invented number as observed. Imputed inputs are declared as imputed.

## Interfaces

- T1 produces `ml/sequence_dataset.py`: `FEATURES`, `TARGET_LEADS`, `INPUT_STEPS`,
  `CycloneSequenceDataset`, `build_sequence_windows(df, ...)`, `normalise`/`denormalise`.
  T2 consumes `FEATURES` (len 7), `TARGET_LEADS` (len 8), `INPUT_STEPS` (4).
- T2 produces `ml/trajectory_model.py`: `BayOfBengalCycloneForecaster(use_vision: bool)`,
  forward(tab[B,T,7], image[B,T,3,224,224]|None, image_mask[B,T]|None)
  -> {"delta": [B,H,2], "intensity": [B,H,2]}. T3 and T4 consume it.
- T3 produces `models/bob_trajectory_gru.pt` with keys
  `model_state_dict, use_vision, feature_names, target_leads, input_steps, norm,
  lead_track_rmse_km, metrics`. T4 consumes every one of those keys.
- T4 produces `app/trajectory.py`: `TrajectoryEngine.forecast(...) -> dict` matching the
  spec's JSON, and `app/schemas.py::TrajectoryResponse`. T5 proxies it, T6 renders it.
- T5 produces `POST /api/predict/trajectory` on Express. T6 consumes it.

## Task 1 - NI sequence pipeline (`ml/sequence_dataset.py`)

Steps:
1. Write `tests/test_sequence_dataset.py` (RED first): basin filter keeps only NI;
   windows are strictly 6-hourly and contiguous (a time gap splits them); window shape
   is (4,7) input / (8,2) coords / (8,2) intensity; pres_hpa NaN is imputed and a
   `was_imputed` flag records it; dlat/dlon NaN on a storm's first fix -> 0.0;
   no window spans two storms; splits are grouped by `sid` with no sid in both.
   Expected: fails, module missing.
2. Implement. Targets are **cumulative deltas from t0** in degrees plus absolute
   wind_kt/pres_hpa; normalisation constants are module-level and exported.
3. `python -m pytest tests/test_sequence_dataset.py -q` Expected: all pass.
4. Print the real NI window count. Expected: >= 250 windows.
5. Commit.

## Task 2 - Multi-modal model (`ml/trajectory_model.py`)

Steps:
1. Write `tests/test_trajectory_model.py` (RED): tabular-only forward gives the documented
   shapes; `use_vision=True` accepts images + mask; a fully-masked image batch stays finite
   and keeps the documented shapes; eval-mode forward is deterministic.
   Expected: fails, module missing.
2. Implement: ResNet-18 trunk (ImageNet weights when available, early layers frozen) -> 128d
   per step, learned missing-image embedding for masked steps; tabular MLP 7->64; fuse -> GRU
   (hidden 128) -> trajectory head (H*2) and intensity head (H*2).
3. Run the tests. Expected: all pass.
4. Commit.

## Task 3 - Training (`ml/train_trajectory.py`)

Steps:
1. Write `tests/test_train_trajectory.py` (RED): haversine_km matches a known great-circle
   distance to <1 km; the loss falls over a short overfit run on one batch; the saved
   checkpoint carries every key T4 needs.
   Expected: fails, module missing.
2. Implement: group split by sid, Haversine + MAE trajectory loss, MSE intensity loss,
   per-lead validation track RMSE written into the checkpoint as `lead_track_rmse_km`
   (the source of the forecast's uncertainty radii - not a hardcoded cone).
3. Run tests. Expected: pass.
4. Train the shipped checkpoint on CPU: global pretrain (all basins) -> NI fine-tune.
   Expected: `models/bob_trajectory_gru.pt` exists; val track RMSE at +24h beats the
   linear-extrapolation baseline the script also reports.
5. Commit (checkpoint itself is gitignored; metrics JSON is committed).

## Task 4 - Real inference behind the API (`app/trajectory.py`, `app/schemas.py`, `app/main.py`)

Steps:
1. Write `tests/test_trajectory_api.py` (RED): `POST /predict/trajectory` with 4 AMPHAN
   observations returns the spec's shape (cyclone_id, region, forecast_generated_at,
   current_metrics, waypoints[forecast_hour/lat/lon/wind_kt/pressure_hpa/uncertainty_radius_km],
   landfall_estimate); uncertainty radii increase with lead time; a 1-observation request
   still answers; missing pressure is imputed and declared; absent checkpoint falls back to
   the labelled physics path; `/predict/forecast` and `/predict/wind` keep their existing
   contracts. Expected: fails.
2. Implement `app/trajectory.py`: coast.json nearest-vertex `dist2land_km`, Atkinson-Holliday
   wind->pressure imputation (declared), checkpoint inference, per-lead uncertainty from
   `lead_track_rmse_km`, landfall scan against the coastline with a NIO coastal gazetteer.
3. Rewire `app/model.py::predict_trajectory` to delegate to the trained engine, keeping
   `ForecastResponse` field-for-field compatible.
4. Run the whole python suite. Expected: all pass, including the pre-existing tests.
5. Commit.

## Task 5 - Express proxy (`server/trajectoryProxy.ts`, `server.ts`, `server/gdacs.ts`)

Steps:
1. Write `tests/trajectoryProxy.test.ts` and extend `tests/gdacs.test.ts` (RED):
   JSON is forwarded to FastAPI and the status/body come back; upstream down -> 503 with a
   readable detail; non-JSON content-type -> 415; `parseCycloneTrack` chains `Line_Line_N`
   segments into ordered centres, reads `polygonlabel` timestamps, and splits observed from
   GDACS-forecast at `todate`. Expected: fails.
2. Implement; register in `server.ts` next to the wind proxy.
3. `npm run test:web` Expected: all pass.
4. Commit.

## Task 6 - Map trajectory visualisation (`src/trajectory.ts`, `src/LiveMap.tsx`, `src/App.tsx`, `src/WindEstimator.tsx`)

Steps:
1. Write `tests/trajectory.test.ts` (RED): `conePolygon(waypoints)` returns a closed ring
   whose width grows with `uncertainty_radius_km`; km->degree conversion respects latitude;
   `trackLatLngs` orders points by time. Expected: fails.
2. Implement `src/trajectory.ts`, then render in `LiveMap.tsx`: solid historical polyline,
   dashed forecast polyline, +12/+24/+48h waypoint markers, semi-transparent cone polygon.
   All new layers are removed on deselect; existing layers untouched.
3. `src/App.tsx`: fetch the forecast for the selected NIO storm, add the "Trajectory &
   multi-metric details" panel (landfall timing/location, dP, peak wind).
4. `src/WindEstimator.tsx`: optional central-pressure input posted alongside the image.
5. `npm run test:web && npm run build` Expected: both clean.
6. Commit.

## Review Focus

- Degradation paths: no checkpoint, 1 observation, all-NaN pressure, storm outside the NIO box,
  GDACS geometry missing or malformed, antimeridian/negative longitudes.
- Honesty: every imputed or extrapolated number is labelled as such in the response and the UI.
- Leakage: grouped-by-sid split, and no target column reachable from the input features.
- Leaflet lifecycle: layers added per selection must not accumulate across re-renders.
