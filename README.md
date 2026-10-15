# CycloneSense

A live satellite map of the Bay of Bengal and Arabian Sea, plus a pipeline that builds a real
tropical-cyclone dataset (for Kaggle) and trains an intensity model on it.

> Research project. Not an official warning system. Follow IMD (https://mausam.imd.gov.in/) for warnings.

## 1. Website (simple, public)

Map of India's seas with NASA GIBS satellite layers (daily true colour, daily thermal IR, near real-time
Himawari IR) and a list of active North Indian Ocean cyclones from GDACS. If the feed is down or no
cyclone is active, the site says so; nothing on it is simulated.

```bash
npm ci
npm run dev          # http://localhost:3000
npm run build && npm start
npm run test:web     # feed parser tests
```

Deploy with the existing `Dockerfile` (Cloud Run, Render, Railway; the server reads `PORT`).

## 2. Real dataset -> Kaggle

`data/build_dataset.py` joins NOAA IBTrACS best-track labels (all basins) with real NASA GIBS MODIS
Terra/Aqua chips (true colour + Band-31 IR) centred on each storm. Needs internet.

```bash
pip install -r requirements.txt
python data/build_dataset.py --check                    # one request per layer: confirms GIBS works
python data/build_dataset.py --limit 50                 # small trial
python data/build_dataset.py --start-year 2003 --workers 8 --kaggle-user YOUR_KAGGLE_USERNAME
kaggle datasets create -p kaggle_dataset -r zip         # needs ~/.kaggle/kaggle.json
```

Output: `tracks.csv` (every 6-hourly record, 6 h deltas, 24 h rapid-intensification flag),
`images_index.csv` (one row per image), `images/`, `dataset-metadata.json`. Re-running resumes.
Splits are by storm ID (70/15/15), never by image.

Limits: one daytime pass per day (no night IR); storms near the dateline are skipped; the IR chip is
GIBS's colour-palette rendering, not raw radiance; wind is USA 1-minute wind. IBTrACS has no Dvorak
pattern labels, so none are included.

## 3. Train and evaluate

```bash
python -m ml.train_chips --data-dir kaggle_dataset --channel truecolor --epochs 12
```

Also runs in a Kaggle notebook (`--data-dir /kaggle/input/<dataset>`, GPU on). It writes
`metrics.json` with test MAE/RMSE, category accuracy, per-category error, and the "predict the
training mean" baseline to compare against. No horizontal flips (they mirror the rotation sense).

## Tests

```bash
pytest tests/test_dataset_pipeline.py tests/test_train_chips.py
npm run test:web
```

## Not included yet

The FastAPI service in `app/` and the 7-class Dvorak classifier in `ml/model.py` are not connected to the
website and have no trained checkpoint in this repo. Wire the model from step 3 into the site once you
have real metrics worth showing.

Data: NOAA NCEI IBTrACS (Knapp et al. 2010); NASA GIBS / EOSDIS; JMA Himawari; GDACS. Check each source's
terms before redistributing.
