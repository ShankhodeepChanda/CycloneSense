# Wind regression inference boundary

`ml/wind_inference.py` is the shared inference boundary for the IR wind-speed regression checkpoints. It defaults to the preserved production candidate `models/multiyear45_resnet18_ir_weighted.pt`; Model C can be passed explicitly for diagnostics. Checkpoint metadata and the state dictionary are loaded strictly. The model output is a normalized scalar trained against `wind_kt / 100`, so inference multiplies that scalar by 100 to produce knots.

```python
from ml.wind_inference import load_wind_model, predict_wind

model, device = load_wind_model()  # Model A by default
batch = predict_wind(model, preprocessed_images, device=device)
research_values_kt = batch.raw_prediction_kt
display_values_kt = batch.application_prediction_kt
```

`raw_prediction_kt` is kept unchanged for research metrics and diagnostics. `application_prediction_kt` is a separate field bounded below by zero for display. A NaN or infinity raises `NonFiniteModelOutputError` and rejects the batch; callers must surface that inference failure rather than substituting zero. Arrays returned by the helper are read-only.

Preprocessing must match training/evaluation: convert to RGB, resize to 224 pixels, convert to tensor, then normalize with ImageNet means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. Training augmentation remains the existing random resized crop (scale 0.8–1.0) and random rotation (±30°), without horizontal flips.

## Application wiring

FastAPI route `POST /predict/wind` loads Model A lazily once per process through `load_wind_model`, preprocesses an uploaded image with `preprocess_wind_image`, and returns the response described in [API.md](API.md). The website's research estimator posts to same-origin `/api/predict/wind`; the Express server streams the multipart request to `PYTHON_API_URL` (default `http://127.0.0.1:8000`). Docker Compose sets the internal API address. The static Vercel build must use `VITE_API_URL` for a separately hosted API origin.

The endpoint returns `raw_prediction_kt` unchanged for scientific use and a separate `application_prediction_kt = max(raw, 0)` display safeguard. `predicted_wind_kt` is a compatibility alias for the display value. `NonFiniteModelOutputError` maps to HTTP 503 and never becomes a zero prediction. Invalid files return 400/413/415; a missing or incompatible checkpoint returns 503. Existing `/predict/pattern` and `/predict/forecast` behavior remains separate and unchanged.

## Checks

The regression tests cover positive, negative, zero, and non-finite outputs; raw-output preservation; category behavior; strict metadata/state loading; scaling; preprocessing; API failures; existing pattern-route compatibility; and the production route with Model A when the checkpoint is available. No checkpoint is trained or modified by this module.
