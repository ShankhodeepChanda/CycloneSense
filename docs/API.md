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
| 503 | Model checkpoint unavailable, inference failed, or model returned non-finite output |

Error responses use FastAPI's `{"detail":"..."}` shape. Stack traces and checkpoint file paths are logged server-side only. `/predict/pattern` remains the separate image-feature baseline; it does not call Model A. `/predict/forecast` remains the trajectory API.

## Deployment wiring

- Local Node dev server: `PYTHON_API_URL` defaults to `http://127.0.0.1:8000`.
- Docker Compose: web service proxies to `http://api:8000`, and the Python image includes the existing Model A file.
- Static Vercel: set `VITE_API_URL` at build time to a separately hosted FastAPI origin. The browser then calls `<VITE_API_URL>/predict/wind`. The repository does not deploy or provision that service.
