# Model A website/API integration

## Architecture and route

The active website is the root React/Vite app served by Node/Express. The Python FastAPI service remains the ML process. The website now posts multipart field `file` to same-origin `/api/predict/wind`; Express streams that request to FastAPI at `PYTHON_API_URL` (default `http://127.0.0.1:8000`). Docker Compose sets the internal URL to `http://api:8000`.

FastAPI exposes `POST /predict/wind` and loads `models/multiyear45_resnet18_ir_weighted.pt` through `ml.wind_inference.load_wind_model`. The process caches the model after the first successful load. The checkpoint uses a ResNet-18 IR head trained against `wind_kt / 100`; inference converts the normalized scalar to knots by multiplying by 100.

The input path converts Pillow-readable images to RGB and applies the deterministic evaluation transform `Resize(224) → ToTensor → ImageNet Normalize`. Uploads are limited to 10 MB and decoded dimensions to 40 megapixels. The model was trained with storm-centred MODIS Band 31 style infrared chips; the route cannot identify a sensor or reject visible imagery on scientific-domain grounds, so other inputs are technically decodable but not validated.

## Response contract

| Field | Meaning |
|---|---|
| `normalized_model_output` | Unscaled model scalar trained against `wind_kt/100` |
| `raw_prediction_kt` | Unmodified model scalar multiplied by 100; retained for research |
| `application_prediction_kt` | `max(raw_prediction_kt, 0)` for UI use |
| `predicted_wind_kt` | Backward-friendly alias of the application value |
| `category` | Existing project category thresholds applied to the raw prediction |
| `model`, `checkpoint`, `unit`, `disclaimer` | Identifies the research model, checkpoint, knots, and non-operational status |

The route returns 400 for empty or invalid images, 413 above 10 MB, 415 for non-image MIME types, and 503 for checkpoint loading/inference/non-finite output failures. It does not replace a failed prediction with zero. Existing `/predict/pattern` and `/predict/forecast` contracts are unchanged.

The UI distinguishes the application estimate from the raw estimate, has file-size/type validation, loading/disabled/error states, an `aria-live` result region, a timeout, and a retry path through the same submit action. The spinner respects reduced-motion preferences.

## Serving boundaries

- **Docker Compose:** web and API services are wired; `Dockerfile.python` includes the existing Model A checkpoint. `.dockerignore` passes through only that checkpoint and `data/__init__.py`/`data/build_dataset.py`, leaving raw and processed datasets outside the build context. A Docker build was not run in this environment.
- **Root Node image by itself:** serves the site and proxy but does not include FastAPI or the checkpoint. Set `PYTHON_API_URL` to a separately available Python service.
- **Static Vercel:** has no Python model runtime. Set `VITE_API_URL` at Vite build time to a separately hosted FastAPI origin; deployment and CORS behavior for that external host were not tested here.
- **Public API hardening:** image bounds and error handling are enforced, but the upload route has no authentication or rate limit. Use an appropriate gateway/rate limit before exposing an unrestricted public inference service.
- **Browser verification:** the Vite production build and Express proxy tests pass. The attached isolated browser could not connect to the localhost dev server, so no rendered-browser or device screenshot is claimed.

No external deployment was attempted or claimed.
