# Model plan
Baseline: deterministic morphology heuristic, included only to make the demo immediately runnable.
# CycloneSense AI Model Architecture & Training

Competition model:
1. EfficientNet/ResNet or compact ViT for satellite imagery.
2. MLP for environmental metadata.
3. Feature fusion.
4. Pattern classification head.
5. LSTM/Temporal Transformer for track and intensity.
## 1. Vision Morphology Model (`CyclonePatternCNN`)
A deep convolutional neural network designed for satellite remote sensing and infrared morphology classification.

Metrics: macro-F1, balanced accuracy, per-class recall; track MAE in km; wind MAE; intensity F1.
### Architecture
- **Input**: 3-channel satellite imagery $(3 \times 128 \times 128)$.
- **Backbone**:
  - Initial Strided Convolution $(5 \times 5, \text{stride}=2, \text{filters}=32)$ + BatchNorm + ReLU.
  - Residual Block 1: 32 -> 64 channels $(\text{stride}=2)$.
  - Residual Block 2: 64 -> 128 channels $(\text{stride}=2)$.
  - Residual Block 3: 128 -> 256 channels $(\text{stride}=2)$.
  - Residual Block 4: 256 -> 256 channels $(\text{stride}=1)$.
- **Classifier Head**:
  - Global Adaptive Average Pooling $(1 \times 1)$.
  - Linear $(256 \to 128)$ + Dropout $(0.3)$ + ReLU + Linear $(128 \to 7)$.
- **Explainability**:
  - Built-in **Grad-CAM** hook registering backward gradients on the final residual feature map to highlight eyewall and spiral band regions.
- **Artifact**: Saved to `models/cyclone_model.pt`.

Add Grad-CAM/attention maps for explainability.
---

## 2. Temporal Track & Intensity Forecaster (`CycloneTrackLSTM`) - design sketch, not built
The section below was a plan. No `models/forecast_model.pt` was ever produced and nothing
loads it. The forecaster the API actually serves is section 2b.

### Architecture
- **Input**: Normalized sliding sequence of 4 consecutive historical observations $(B, 4, 8)$:
  `[lat, lon, wind_kts, pressure_hpa, dlat, dlon, dwind, dpressure]`.
- **Core**: 2-layer LSTM $(\text{hidden}=64, \text{dropout}=0.2)$.
- **Multi-Task Heads**:
  - **Track Position Head**: Linear $(64 \to 32) \to \text{Linear}(32 \to 2)$ predicting $(\text{next\_lat}, \text{next\_lon})$.
  - **Wind Speed Head**: Linear $(64 \to 32) \to \text{Linear}(32 \to 1)$ predicting future maximum sustained wind.
  - **Intensity Classification Head**: Linear $(64 \to 32) \to \text{Linear}(32 \to 4)$ predicting cyclone strength category.
- **Loss Function**:
  $$\mathcal{L} = \text{MSE}(\text{pos}) + 0.1 \cdot \text{SmoothL1}(\text{wind}) + 0.5 \cdot \text{CrossEntropy}(\text{class})$$
- **Artifact**: Saved to `models/forecast_model.pt`.

---

## 3. Training & Inference
- **Device Support**: Hardware accelerated on NVIDIA GPUs (CUDA) with automatic CPU fallback.
- **API Integration**: FastAPI backend dynamically loads trained weights on startup and runs real-time inference with cached memory state.


---

## 2b. Bay of Bengal track and intensity forecaster (`BayOfBengalCycloneForecaster`)

The model the API actually serves. Defined in `ml/trajectory_model.py`, trained by
`ml/train_trajectory.py`, served through `app/trajectory.py`.

### Inputs

Four consecutive 6-hourly best-track fixes (t-18h, t-12h, t-6h, t0). Seven features per
step: `lat`, `lon`, `dlat`, `dlon`, `wind_kt`, `pres_hpa`, `dist2land_km`. Normalisation
constants are fixed in `ml/sequence_dataset.py` and travel in the checkpoint, so training
and serving share one frame.

Windows are cut per storm and only from runs spaced exactly 6 hours apart, so a missing fix
is never bridged into a sequence that was not observed. Position and wind must be observed
or the window is dropped; central pressure is imputed (forward-fill, then window mean, then
Atkinson-Holliday from wind) and the window records that it was.

### Architecture

- **Vision branch**: ResNet-18 with `conv1`, `bn1`, `layer1`-`layer3` frozen, `layer4` and a
  linear projection trainable, giving 128-d per timestep. A step with no satellite chip uses
  a learned missing-image embedding and never reaches the trunk, so a fully unchipped
  sequence costs nothing and takes no gradient from zero pixels.
- **Tabular branch**: MLP 7 -> 64 -> 64 with LayerNorm.
- **Fusion and sequence core**: concatenation -> linear -> GELU -> single-layer GRU, hidden 128.
- **Heads**: a trajectory head emitting cumulative (dlat, dlon) at each of eight leads, and an
  intensity head emitting (wind_kt, pres_hpa) at each lead.

Tabular-only the model is 149,344 parameters; with the vision branch about 11.4 M.

### Training

Trajectory loss is great-circle distance in kilometres between predicted and observed
positions, so the quantity minimised is the quantity reported, plus a degree-space MAE term
for early gradient stability. Intensity loss is MSE on normalised wind and pressure.

Validation is **grouped by storm id**. Windows inside one storm overlap by up to 11 of their
12 fixes, so a random split would leak nearly every validation target into training.

Two stages produced the shipped checkpoint. Stage one excludes the North Indian Ocean
**by storm, not by basin**: PABUK, BULBUL and KROVANH each carry both NI and WP rows in
`kaggle_trial/tracks.csv`, so a plain `--basins WP,SI,EP,SP` filter would still train stage
one on storms stage two holds out. `train()` refuses that combination rather than publishing
a contaminated error as the forecast cone.

    python -m ml.train_trajectory --basins WP,SI,EP,SP --exclude-storms-in-basins NI \
        --epochs 45 --batch-size 64 --checkpoint models/bob_trajectory_global.pt

    python -m ml.train_trajectory --basins NI --epochs 200 --lr 5e-4 --patience 60 \
        --init-from models/bob_trajectory_global.pt \
        --checkpoint models/bob_trajectory_gru.pt

### Measured results

North Indian Ocean fine-tune: 330 windows over 20 storms, 247 training and 83 validation
windows, validation held out by storm (LUBAN, GAJA, PABUK, PAWAN, BUREVI). Per-lead
validation track RMSE in km, against persistence of the last observed 6-hourly motion
vector:

| Lead | Model | Linear baseline |
|---|---|---|
| +6 h | 32.2 | **30.7** |
| +12 h | 65.5 | 67.4 |
| +18 h | **106.5** | 111.7 |
| +24 h | **153.7** | 160.7 |
| +30 h | **197.5** | 210.8 |
| +36 h | **237.7** | 259.1 |
| +42 h | **285.9** | 309.8 |
| +48 h | **337.2** | 364.1 |

Wind MAE 13.5 kt and pressure MAE 8.4 hPa, both averaged over the eight leads.

The model beats persistence at seven of the eight leads and **loses at +6 h**, where simply
continuing the last observed motion vector is already very hard to improve on. The margin
grows with lead time, from 3% at +12 h to 7% at +48 h, which is the shape you would expect:
persistence degrades as a track curves, and curvature is what the model has learned.

An earlier run of this recipe pre-trained on `--basins all`, which showed stage one four of
the five storms stage two held out. That reported +24 h 139.8 km and +48 h 301.5 km and
claimed every lead; the figures above are what the same recipe gives once the stages are
storm-disjoint, so the contamination was worth about 10% of the result.

These are not operational-grade errors: IMD's +24 h track error is roughly 80-100 km. 330
windows is a small dataset, and the figures should be read as a research result on this
repository's data rather than as forecast skill.

Two caveats travel with the numbers, both recorded in the checkpoint's `metrics`:
`model_selected_on_validation_set` is true, so the weights were chosen by lowest validation
error and then scored on that same set, which is a selection-biased estimate; and
`validation_is_training_set` reports when a dataset was too small to split at all.

The per-lead RMSE column is written into the checkpoint as `lead_track_rmse_km` and becomes
the API's cone of uncertainty, so the cone is a measurement rather than a chosen shape.

### The vision branch is built, but the shipped checkpoint is tabular-only

`kaggle_trial/images_index.csv` holds 28 North Indian Ocean chips covering 14 timesteps,
against 570 track fixes. That is not enough to form a single four-step image sequence, so
training the ResNet trunk on this basin would fit noise. The branch, the per-step image mask
and the dataset's chip loading are implemented and tested; the checkpoint records
`use_vision: false` and inference follows it. With a real chip archive the same script
trains the multi-modal model through `--use-vision --images-index ... --image-root ...`.
