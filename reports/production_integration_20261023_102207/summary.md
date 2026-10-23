# CycloneSense production integration summary

**Checked:** 2026-10-23  
**Repository:** `D:\CycloneSense`  
**Model A:** `models/multiyear45_resnet18_ir_weighted.pt`  
**Model A SHA-256:** `3ac72bfbd2675e2aa79ce131aa6eef138a4867dfb5ca0063bb4ac5b19068d6d8`

## Completed

- Connected Model A to FastAPI `POST /predict/wind`, cached per process, using the existing shared inference boundary and the evaluation-time RGB/Resize(224)/ImageNet normalization.
- Added the Research page image estimator and same-origin Express multipart proxy at `/api/predict/wind`. Raw knot predictions remain separate from the zero-floored application value.
- Packaged the preserved checkpoint in the Python Docker image and configured Docker Compose to route web requests to FastAPI.
- Added a dry-run-first 2024+ track/imagery preparation command and a frozen, storm-disjoint evaluation command. No imagery was downloaded and no training was performed.
- Verified current C4/C5 distributions and wrote a prioritized data acquisition plan.
- Updated README, API, inference, architecture, and dataset documentation.

## Findings and limits

The API route and UI integration are implemented, but there is no external deployment in this task. The React estimator requires the Python service. In Docker Compose the services are wired together; the root Node-only image needs a separately configured API, and static Vercel needs `VITE_API_URL` set to a separately hosted FastAPI service. The local browser preview could not connect to the isolated dev server, so the frontend was verified through the production build, code review, and a live in-process proxy test rather than a rendered browser screenshot.

The local IBTrACS cache has 3,262 eligible six-hourly USA_WIND records for 2024–2025, but all 120 candidate storm IDs with planned satellite jobs overlap SIDs in existing local train/validation/test track tables. The existing image indexes contain zero paired 2024+ IR rows. Under the requested storm-disjoint rule, no 2024+ evaluation set can currently be formed; no evaluation metrics were generated.

Training-split coverage remains sparse: C4 has 19 images from 7 storms and C5 has 2 images from 1 storm. This supports a class-imbalance concern, but does not establish label errors or explain model error by itself.

The Model A checkpoint and historical metrics/prediction artifacts were fingerprinted before and after this work. All 58 protected artifacts match, including the unchanged Model A digest. See the [artifact integrity record](artifact_integrity.json), [integration report](integration_report.md), [pipeline report](data_pipeline_report.md), [C4/C5 plan](c4_c5_data_plan.md), [readiness JSON](temporal_evaluation_readiness.json), and [test results](test_results.txt).
