# Architecture

## Current website and serving path

- React/Vite renders the live dashboard, timeline, and map. NASA GIBS provides map imagery; the existing GDACS and Open-Meteo adapters provide cyclone and weather data.
- The production Node/Express server serves the built website and the existing cyclone feed route. It streams same-origin `POST /api/predict/wind` multipart uploads to the Python service using `PYTHON_API_URL`.
- FastAPI exposes Model A at `POST /predict/wind`. It loads `models/multiyear45_resnet18_ir_weighted.pt` once on first request in each process and uses the shared preprocessing/inference boundary in `ml/wind_inference.py`.
- The Research page displays the nonnegative application field and retains raw knots in the response. The zero floor is presentation-only. `/predict/pattern` remains a heuristic image-analysis baseline; `/predict/forecast` remains the trajectory engine.
- Docker Compose connects the Node and Python services. The root Node-only Dockerfile does not contain a Python runtime; configure a separate FastAPI origin when using it alone. A static Vercel build needs `VITE_API_URL` set to the separate API origin at build time.

## Research data path

`data/build_dataset.py` selects IBTrACS `USA_WIND` labels and joins them with nearest-pass Terra/Aqua MODIS imagery from NASA GIBS. It stores a rendered 256×256 IR chip index; the imagery is not raw calibrated radiance. The 2024+ preparation/evaluation scripts reuse that contract, record agency/source provenance, compare storm IDs across existing local split tables, and refuse evaluation without valid, independent pairs. See [DATASET.md](DATASET.md) and the production integration report for the current local readiness result.

Keep satellite source, channel, UTC timestamp, and track agency explicit when extending the data. Do not treat images from other sensors or visible channels as interchangeable with the current MODIS IR model input.
