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

## Application wiring status

The current website does not load the wind regression checkpoint. `/predict/pattern` is a separate image-feature heuristic, while `/predict/forecast` calls the existing `CycloneSenseEngine` trajectory routine. The map uses external cyclone/weather feeds. Do not describe those routes as Model A inference. When Model A is connected to a route, call this boundary after the shared preprocessing step, return the application value for display, and retain the raw value in research logs and metrics. Convert `NonFiniteModelOutputError` into an explicit failed inference response.

## Checks

The regression tests cover positive, negative, zero, and non-finite outputs; raw-output preservation; category behavior; strict metadata/state loading; and the factor-of-100 conversion. The production readiness report records CPU checks against both saved checkpoints. No checkpoint is trained or modified by this module.
