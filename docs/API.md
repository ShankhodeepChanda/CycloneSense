# API reference

The Python service is started with `python run_server.py --host 0.0.0.0 --port 8000`. The Node website proxies wind uploads at same-origin `/api/predict/wind` to the Python route using `PYTHON_API_URL`. Other existing routes keep their current contracts.

## `POST /predict/wind`

Multipart form upload, field name `file`. Supported Pillow-readable image formats are accepted when the MIME type begins with `image/`. The upload is limited to 10 MB and decoded images to 40 megapixels. The model was trained on storm-centred, palette-rendered MODIS Band 31 infrared chips; arbitrary visible imagery or another sensor can be decoded, but its estimate is outside the validated input domain. The web client uses the same-origin proxy route `/api/predict/wind`.

Preprocessing uses Pillow RGB conversion, torchvision `Resize(224)` (short edge 224, aspect ratio preserved), `ToTensor()`, and ImageNet channel normalization with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. This is deterministic evaluation preprocessing; training still uses its existing random-resized crop and rotation augmentation.

Success response (HTTP 200):

```json
{
  "status": "success",
  "model": "Model A ResNet-18 IR wind regression",
  "checkpoint": "multiyear45_resnet18_ir_weighted.pt",
  "unit": "kt",
  "normalized_model_output": -0.05,
  "raw_prediction_kt": -5.0,
  "application_prediction_kt": 0.0,
  "predicted_wind_kt": 0.0,
  "category": "TD",
  "disclaimer": "Research estimate from a satellite image; not an official warning or operational intensity estimate."
}
```

The sample demonstrates field meanings; it is not a measured model result. The saved Model A produces the normalized target `wind_kt/100`; `raw_prediction_kt` is the raw output multiplied by 100. Only `application_prediction_kt` is bounded below by zero. `predicted_wind_kt` aliases the application value for straightforward frontend compatibility. `category` is derived from the raw prediction using the project's existing wind thresholds.

| Status | Meaning |
|---|---|
| 400 | Empty or undecodable image |
| 413 | Upload larger than 10 MB |
| 415 | Request is not an image upload |
| 422 | `pressure_hpa` was supplied but is not a number in 800-1050 |
| 503 | Model checkpoint unavailable, inference failed, or model returned non-finite output |

Error responses use FastAPI's `{"detail":"..."}` shape. Stack traces and checkpoint file paths are logged server-side only. `/predict/pattern` remains the separate image-feature baseline; it does not call Model A.

### Optional central pressure

The form accepts an optional `pressure_hpa` field (800-1050). When present the response gains a `multi_metric` block holding a second, independent intensity estimate derived from that pressure through the Atkinson-Holliday (1977) relation, the pressure implied by the image estimate, their difference in knots, and an `agreement` verdict of `close` (within 10 kt), `moderate` (within 25 kt) or `poor`.

Model A takes pixels only. The supplied pressure never enters the image estimate, and `tests/test_wind_api.py` pins that: `raw_prediction_kt` is identical with and without a pressure. The block is `null` when no pressure is sent.

## `POST /predict/trajectory`

JSON. Forecasts track and intensity at +6h through +48h from a sequence of observed fixes. The Node website proxies it at same-origin `/api/predict/trajectory`.

```json
{
  "cyclone_id": "AMPHAN",
  "observations": [
    { "lat": 12.5, "lon": 86.4, "wind_kt": 120, "pressure_hpa": 932, "timestamp": "2020-05-17T18:00:00Z" },
    { "lat": 14.1, "lon": 86.4, "wind_kt": 145, "pressure_hpa": 901, "timestamp": "2020-05-18T12:00:00Z" }
  ]
}
```

Only `lat` and `lon` are required per observation. The model consumes the last four fixes; a shorter sequence is padded by repeating the earliest, and the response reports `padded_input_steps` in `inputs_imputed`. Central pressure is derived from wind, or wind from pressure, through Atkinson-Holliday when one is missing, and the derived field is named in `inputs_imputed`. `dist2land_km` is always computed from `public/coast.json` rather than supplied.

Success response (HTTP 200), abridged:

```json
{
  "status": "success",
  "cyclone_id": "AMPHAN",
  "region": "Bay of Bengal",
  "forecast_generated_at": "2026-10-23T18:30:00+00:00",
  "source": "trained-sequence-model",
  "model": "BayOfBengalCycloneForecaster (bob-tabular-gru, basins=NI)",
  "checkpoint": "bob_trajectory_gru.pt",
  "observations_used": 4,
  "inputs_imputed": ["dist2land_km (derived from the Natural Earth coastline)"],
  "current_metrics": { "lat": 14.1, "lon": 86.4, "wind_kt": 145.0, "pressure_hpa": 901.0, "dist2land_km": 519.1 },
  "waypoints": [
    { "forecast_hour": 24, "lat": 17.43, "lon": 87.32, "wind_kt": 124.8, "pressure_hpa": 931.9, "uncertainty_radius_km": 139.8, "dist2land_km": 298.0 }
  ],
  "landfall_estimate": { "projected": false, "location": null, "lat": null, "lon": null, "eta_hours": null, "note": "No coastal crossing within 48 h. ...", "threshold_km": 35.0 },
  "summary": { "peak_wind_kt": 145.0, "min_pressure_hpa": 901.0, "peak_pressure_hpa": 963.9, "pressure_change_hpa": 62.9, "wind_change_kt": -67.9, "max_deepening_hpa": 0.0, "intensity_class": "Super Cyclonic Storm (SuCS)", "forecast_track_km": 880.7 },
  "validation": { "lead_track_rmse_km": [29.4, 61.8, 97.8, 139.8, 176.7, 217.8, 257.8, 301.5], "metrics": {} },
  "disclaimer": "Research forecast ... Verify against IMD and RSMC New Delhi advisories."
}
```

The sample shows field meanings; the numbers are one real run on AMPHAN fixes, not a guarantee. `waypoints` always covers the eight trained leads: 6, 12, 18, 24, 30, 36, 42 and 48 hours.

`uncertainty_radius_km` is the checkpoint's measured validation track RMSE at that lead, not a chosen shape. `validation.lead_track_rmse_km` repeats the whole set so a client can show what the cone means.

`source` states where the numbers came from. With no checkpoint installed it is `physics-extrapolation`: the engine extrapolates the last observed motion vector, weakens the storm over land, and uses the measured per-lead RMSE of that same baseline as the cone. It is never presented as a model prediction.

`dist2land_km` is `null` wherever `public/coast.json` cannot answer. That file is an Eastern Hemisphere coastline holding no American coast, and no point on Earth lies more than about 2 690 km from land, so a nearest coast beyond 2 800 km is treated as missing coverage rather than reported as a distance. `landfall_estimate.note` then says landfall was not evaluated.

| Status | Meaning |
|---|---|
| 422 | No observations, a lat/lon outside range, or no `wind_kt` and no `pressure_hpa` anywhere in the sequence |
| 503 | Forecast failed unexpectedly |

Intensity cannot be inferred from position alone, so a sequence carrying neither wind nor pressure is refused rather than given an invented intensity.

## `POST /predict/forecast`

Unchanged contract, now driven by the same trained model. It publishes +12h, +24h, +36h, +48h and +72h. The model is trained to +48h, so the +72h point continues the +42h-to-+48h motion vector with a widened cone; `cone_radius_km` values come from the same measured per-lead RMSE. `landfall_intercept` is `null` when no crossing is projected, and its `tidal_coincidence` field states that tide and surge are not modelled by this project.

## `GET /api/cyclone-track` (Node)

`?eventId=<numeric GDACS event id>&episodeId=<numeric>`. Returns that storm's track from the GDACS per-event geometry endpoint, split into `observed` and `forecast` points at the event's `todate`, plus GDACS's own `cone`. The GDACS event list carries only one centroid per storm, so this is what supplies the map's history polyline. Non-numeric ids are refused with 400. A failed upstream fetch returns `{"ok": false, ...}`, with the last cached answer marked `stale` when one exists.

## Deployment wiring

- Local Node dev server: `PYTHON_API_URL` defaults to `http://127.0.0.1:8000`.
- Docker Compose: web service proxies to `http://api:8000`, and the Python image includes the existing Model A file.
- Static Vercel: set `VITE_API_URL` at build time to a separately hosted FastAPI origin. The browser then calls `<VITE_API_URL>/predict/wind`. The repository does not deploy or provision that service.
