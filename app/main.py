"""FastAPI entry point for CycloneSense AI."""
from __future__ import annotations
import hashlib, io, logging, threading, time
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageStat, UnidentifiedImageError
from app.schemas import ForecastRequest, ForecastResponse, PatternResponse, HealthResponse, WindPredictionResponse
from app.model import engine
from data.build_dataset import category as wind_category
from ml.wind_inference import (
    InvalidWindImageError, NonFiniteModelOutputError, load_wind_model,
    predict_wind, preprocess_wind_image,
)

START_TIME = time.time()
CLASSES = ["clear", "developing", "curved_band", "central_dense_overcast", "eye", "sheared", "dissipating"]
MAX_WIND_UPLOAD_BYTES = 10 * 1024 * 1024
_wind_model = None
_wind_device = None
_wind_model_lock = threading.Lock()
LOGGER = logging.getLogger(__name__)


def _get_wind_model():
    """Load Model A at most once per API process; failures are retried on next request."""
    global _wind_model, _wind_device
    if _wind_model is None:
        with _wind_model_lock:
            if _wind_model is None:
                model, device = load_wind_model()
                _wind_model, _wind_device = model, device
    return _wind_model, _wind_device

def analyze_uploaded_image(data: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(data)).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid readable image.") from exc
    image.thumbnail((256, 256))
    stat = ImageStat.Stat(image)
    mean, std = stat.mean[0] / 255.0, stat.stddev[0] / 255.0
    hist = image.histogram(); total = max(1, image.width * image.height)
    dark, bright = sum(hist[:64]) / total, sum(hist[192:]) / total
    scores = {name: 0.01 for name in CLASSES}
    if std < 0.10: scores["clear"] = 0.78
    elif std > 0.30 and bright > 0.20 and dark > 0.08: scores["eye"], scores["central_dense_overcast"] = 0.62, 0.16
    elif bright > 0.38: scores["central_dense_overcast"], scores["developing"] = 0.58, 0.18
    elif std > 0.20: scores["curved_band"], scores["developing"] = 0.48, 0.25
    else: scores["developing"], scores["dissipating"] = 0.42, 0.24
    s = sum(scores.values()); probs = {k: round(v/s, 4) for k,v in scores.items()}; pattern = max(probs, key=probs.get)
    small = image.resize((12,12)); px = list(small.getdata()); grid = [[abs(px[r*12+c]/255.0-mean) for c in range(12)] for r in range(12)]; m = max(max(row) for row in grid) or 1.0; grid = [[round(v/m,3) for v in row] for row in grid]
    return {"status":"success", "pattern_predicted":pattern, "dvorak_taxonomy":pattern.replace("_"," ").upper(), "confidence":probs[pattern], "probabilities":probs, "min_brightness_temp_kelvin":round(285-mean*85,2), "estimated_central_pressure_hpa":round(1012-probs[pattern]*55,1), "grad_cam_saliency_hash":"sha256:"+hashlib.sha256(data).hexdigest()[:32], "explanation":f"Uploaded image analyzed at {image.width}x{image.height} using transparent brightness and contrast features. Research CV baseline; not a trained meteorological classifier.", "grad_cam_grid":grid, "disclaimer":"Research baseline only. Verify against official IMD/RSMC advisories."}

app = FastAPI(title="CycloneSense AI", description="Cyclone image analysis and short-term forecasting.", version="1.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(status="ok", service="cyclonesense-api-python", version="1.3.0", uptime_seconds=round(time.time()-START_TIME,1), hardware={"engine":engine.device,"gpu":engine.device,"model_mode":"image-analysis-baseline"}, models_loaded={"image_analyzer":True,"trajectory_engine":True,"gradcam_engine":False,"wind_regression":_wind_model is not None}, telemetry={"satellite_sources":["INSAT-3D/3DR VHRR","Himawai-9","GOES-16"],"spatial_coverage":"North Indian Ocean"})

@app.post("/predict/pattern", response_model=PatternResponse)
async def predict_cyclone_pattern(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(415,"Please upload an image file.")
    data = await file.read()
    if not data: raise HTTPException(400,"The uploaded image is empty.")
    if len(data) > 10*1024*1024: raise HTTPException(413,"Image exceeds the 10 MB limit.")
    try: return PatternResponse(**analyze_uploaded_image(data))
    except ValueError as exc: raise HTTPException(400,str(exc)) from exc


@app.post("/predict/wind", response_model=WindPredictionResponse)
async def predict_wind_speed(file: UploadFile = File(...)):
    """Estimate wind from one IR-style image using the preserved Model A checkpoint."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, "Please upload an image file (PNG, JPEG, or another supported image format).")
    data = await file.read(MAX_WIND_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded image is empty.")
    if len(data) > MAX_WIND_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the 10 MB limit.")
    try:
        image_tensor = preprocess_wind_image(data)
    except InvalidWindImageError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        model, device = await run_in_threadpool(_get_wind_model)
    except Exception as exc:
        LOGGER.exception("Could not load the configured wind checkpoint")
        raise HTTPException(503, "The wind estimation model is unavailable. Check the Model A checkpoint installation.") from exc
    try:
        prediction = await run_in_threadpool(predict_wind, model, image_tensor, device)
    except NonFiniteModelOutputError as exc:
        LOGGER.error("Wind model emitted a non-finite output: %s", exc)
        raise HTTPException(503, "The wind model returned an invalid numeric result. No estimate was produced.") from exc
    except Exception as exc:
        LOGGER.exception("Wind prediction failed")
        raise HTTPException(503, "Wind estimation failed. Please try a different satellite image.") from exc
    raw = float(prediction.raw_prediction_kt[0])
    application = float(prediction.application_prediction_kt[0])
    return WindPredictionResponse(
        model="Model A ResNet-18 IR wind regression",
        checkpoint=Path("models/multiyear45_resnet18_ir_weighted.pt").name,
        normalized_model_output=float(prediction.normalized_model_output[0]),
        raw_prediction_kt=raw,
        application_prediction_kt=application,
        predicted_wind_kt=application,
        category=wind_category(raw),
    )

@app.post("/predict/forecast", response_model=ForecastResponse)
def predict_cyclone_forecast(payload: ForecastRequest):
    try: return ForecastResponse(**engine.predict_trajectory(payload.cyclone_id,[o.model_dump() for o in payload.observations]))
    except Exception as exc: raise HTTPException(500,str(exc)) from exc

@app.get("/cyclones/active")
def list_active_cyclones(): return {"status":"success","active_systems":[],"message":"Live registry is not connected."}

@app.get("/export/geojson")
def export_geojson(cyclone_id: str="demo"):
    result=engine.predict_trajectory(cyclone_id,[{"lat":14.2,"lon":72.1,"wind_kts":45,"pressure_hpa":995},{"lat":14.7,"lon":72.8,"wind_kts":52,"pressure_hpa":989},{"lat":15.1,"lon":73.5,"wind_kts":58,"pressure_hpa":984}]); points=[[73.5,15.1]]+[[p["pred_lon"],p["pred_lat"]] for p in result["prognostic_trajectory"]]
    return {"type":"FeatureCollection","features":[{"type":"Feature","properties":{"cyclone_id":cyclone_id},"geometry":{"type":"LineString","coordinates":points}}]}

@app.get("/export/bulletin", response_class=PlainTextResponse)
def export_bulletin(cyclone_id: str="demo"): return f"CYCLONESENSE AI RESEARCH BULLETIN\nCYCLONE ID: {cyclone_id}\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())}\n\nVerify against official IMD/RSMC advisories."
