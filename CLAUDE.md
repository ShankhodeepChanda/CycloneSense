# CycloneSense — working notes

Cyclone track and intensity forecasting for the Bay of Bengal, with a live satellite
dashboard. Two processes: a **Vite + Express** frontend on `:3000` and a **FastAPI**
service on `:8000`. The browser only ever talks to `:3000`, which proxies `/api/*`.

```bash
python run_server.py --port 8000     # ML API
npm run dev                          # dashboard -> http://localhost:3000
```

Verify everything (all four must stay clean):

```bash
python -m pytest -q    # 240 passed, 1 skipped
npm run test:web       # 66 passed
npm run typecheck      # tsc --noEmit over src, server, tests, api
npm run build          # vite + esbuild
```

Install: `requirements.txt` is **numpy only** (see Vercel below); use
`requirements-ml.txt` for training and the full API, `requirements-dev.txt` for both plus
pytest.

---

## The core discipline: never present an invented number as observed

This is the project's distinguishing constraint and most of the design follows from it.
Before changing anything that produces a number a user sees, check it against this list.

- **Every derived input is declared.** `inputs_imputed` in the forecast response names
  anything the engine filled in — pressure derived from wind, a padded input sequence, a
  rescaled observation cadence, an unavailable coastline distance.
- **A refusal beats a guess.** A sequence with no wind *and* no pressure anywhere is
  rejected with 422; intensity cannot be inferred from position alone.
- **The cone is a measurement.** `uncertainty_radius_km` is the checkpoint's per-lead
  validation RMSE (`lead_track_rmse_km`), not a shape chosen to look convincing.
- **The fallback denies the model.** With no checkpoint the API answers by extrapolating
  the last motion vector, reports `source: "physics-extrapolation"`, and serves a
  disclaimer that says *"This is not a model forecast."* Do not let the trained-model
  disclaimer leak onto that path — that bug shipped once.
- **Demo results state whether the model trained on that storm.** `provenance` is read
  from the checkpoint's own train/val lists, and `build_demo_storms` refuses to export a
  set with no held-out storm.

A previous version of `app/model.py` fabricated a Gujarat landfall, a 0.942 confidence and
a storm-surge figure for *every* request regardless of input. That is the thing to not go
back to.

---

## The model

`BayOfBengalCycloneForecaster` (`ml/trajectory_model.py`): ResNet-18 vision branch →
128-d per step, tabular MLP (7 features) → 64-d, fused → GRU(128) → trajectory head and
intensity head. Input is four 6-hourly fixes; output is eight leads, +6h to +48h.

**The shipped checkpoint is tabular-only** (`use_vision: false`, 149,344 parameters). The
vision branch is built, wired and tested, but the repo has 28 North Indian Ocean chips
covering 14 timesteps against 570 track fixes — not enough for a single four-step image
sequence, so training the trunk here would fit noise. `--use-vision --images-index …
--image-root …` trains it once a real chip archive exists.

### Training

```bash
python -m ml.train_trajectory --basins WP,SI,EP,SP --exclude-storms-in-basins NI \
    --epochs 45 --batch-size 64 --checkpoint models/bob_trajectory_global.pt
python -m ml.train_trajectory --basins NI --epochs 200 --lr 5e-4 --patience 60 \
    --init-from models/bob_trajectory_global.pt --checkpoint models/bob_trajectory_gru.pt
python -m ml.numpy_forecaster --checkpoint models/bob_trajectory_gru.pt   # export for serving
python -m scripts.export_demo_storms                                      # refresh demo labels
```

`--exclude-storms-in-basins NI` is not optional. **A basin filter does not hold a storm
out**: PABUK, BULBUL and KROVANH each carry both NI and WP rows, so a plain `--basins
WP,…` pretrain still trains on storms the NI fine-tune holds out. `train()` refuses that
combination, because the resulting validation RMSE is what the API publishes as its cone.

An earlier run pretrained on `--basins all` and reported +24h 139.8 km / +48h 301.5 km,
"beats the baseline at every lead". Those were contaminated. The honest numbers:

| Lead | Model | Persistence baseline |
|---|---|---|
| +6 h | 32.2 | **30.7** (baseline wins) |
| +12 h | 65.5 | 67.4 |
| +24 h | **153.7** | 160.7 |
| +48 h | **337.2** | 364.1 |

Wind MAE 13.5 kt, pressure MAE 8.4 hPa. Beats persistence at seven of eight leads, loses
at +6h. Not operational grade — IMD's +24h error is roughly 80-100 km — and 330 training
windows is a small dataset. `metrics` also records `model_selected_on_validation_set:
true`, so these are selection-biased, not a clean held-out estimate. A third never-touched
test split is the obvious next improvement.

---

## Serving runs without PyTorch

torch is **534 MB**; a Vercel serverless function is capped at **250 MB**. So
`ml/numpy_forecaster.py` reproduces the forward pass in numpy (34 MB) from
`models/bob_trajectory_gru.npz` (544 KB, committed). `app/trajectory.py` prefers the
`.npz` over the `.pt` wherever both exist and reports `runtime: "numpy" | "torch"`.

`tests/test_numpy_forecaster.py` pins numpy against PyTorch on random, real and extreme
inputs **including the shipped checkpoint**, and asserts the serving path imports neither
torch nor pandas. **Do not weaken those tests** — a hand-written forward pass is only
trustworthy because they exist, and a divergence would be a wrong forecast rather than a
failing test. pandas is lazily imported in `ml/sequence_dataset.py` for the same reason
(67 MB that no forecast touches).

---

## Gotchas that cost real time

- **Vercel reads only the project-root `requirements.txt`** for a Python function; a file
  beside the entrypoint is ignored. The root file therefore stays minimal. Putting the ML
  stack there made pip resolve 5.8 GB and the build aborted.
- **Relative imports in `server/` and `api/` need the `.js` extension.** Node ESM requires
  it; `tsx`, `tsc` and all 66 web tests resolve the extensionless form happily, so only a
  real deployment catches it. One missing extension took down every Node function.
- **`public/coast.json` is an Eastern Hemisphere coastline** — zero vertices between 165°W
  and 20°W, so no American coast. `distance_to_land_km` guards twice: the longitude must
  fall in a populated 5° bin, and the nearest vertex must be inside the Point Nemo bound
  (2 800 km). Outside that it returns `None`, never a number. Unguarded it reported
  6 997 km off Mexico and quoted it in a landfall note.
- **GDACS's event list returns only `Point_Centroid`** — one position per storm. The full
  track comes from the per-event `polygons/getgeometry` endpoint (`server/gdacsTrack.ts`):
  `Line_Line_N` segments chain into ordered centres, `Point_Polygon_Point_N` labels give
  per-fix timestamps, `todate` splits observed from GDACS's own forecast. Dates may or may
  not carry a zone, so parse both ways; appending `"Z"` unconditionally yielded `NaN` and
  silently relabelled every forecast point as an observation.
- **GDACS answers in 6-8 s and sometimes times out.** The budget is 25 s and failures
  degrade to a cached or stale answer. The feed being briefly unreachable is normal.
- **Observation spacing matters.** The model's motion features are per 6 hours, so a
  12-hourly feed doubles every displacement. `prepare_sequence` reads supplied timestamps,
  rescales, and declares it. Always send `timestamp`.
- **Nothing in the test suite renders the page.** Layout, accessible names and map
  behaviour are only verifiable by opening it. Four real defects were found that way and
  none were catchable by tests.

---

## The hindcast demo

GDACS reports no active cyclone for most of the year — sometimes none on Earth — and every
dashboard feature sits behind a live storm. `public/demo-storms.json` (generated by
`scripts/export_demo_storms.py`) holds six real Bay of Bengal landfalling storms. Picking
one forecasts from a chosen fix and overlays what the storm actually did, so the error is
visible rather than claimed.

AMPHAN, FANI, TITLI and NIVAR were **trained on**; GAJA and BUREVI were **held out** and
are the honest cases. Lead a demo with BUREVI (+24h 106 km, +48h 122 km) and have an
answer ready for GAJA (+48h 478 km, a recurving storm and the weakest case).

---

## State as of 2026-10-24

Treat this section as stale on sight; check `gh pr list` and `git log origin/main`.

- **Live demo: https://cyclonesense.vercel.app** — deployed from a local directory via
  `vercel --prod` under the personal scope `the-samurai`, *not* git-connected. Pushing to
  the repo does not redeploy it.
- The repo owner has two **git-connected** Vercel projects, `cyclone-sense` and
  `cyclone-sense-2klx`, which build every PR. Their logs are not readable from this
  account; diagnose their failures by reproducing locally.
- **PR #33** (`fix/vercel-build-config`) — the requirements split and the `.js` extension.
  `main` is undeployable until this merges; it was left out when PR #32 was merged one
  commit early.
- **PR #34** (`feat/hindcast-demo`) — stacked on #33. Merge #33 first.
- **PR #30** (owner's dashboard redesign) **conflicts** with `src/App.tsx`,
  `src/LiveMap.tsx` and `src/styles.css`. Coordinate before merging either side.
- `/predict/wind` returns 503 everywhere: Model A's checkpoint was never committed, and it
  needs torch, so it cannot deploy to Vercel at all. `api/predict/wind.ts` says so rather
  than 404ing.

### Known gaps

- Phone-width layout unverified.
- Reported errors are selection-biased; no held-out test split distinct from validation.
- `/export/geojson` still builds its LineString from three hardcoded Arabian Sea fixtures
  while labelling the feature with the caller's `cyclone_id` (pre-existing).
- `_interpolate_path` plus a 16 070-vertex scan runs twice when no landfall is found.
- `pressure_mae_hpa` is partly measured against imputed truth (4 of 330 windows), unlabelled.
