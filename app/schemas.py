"""Data schemas and Pydantic V2 models for CycloneSense AI."""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Historical or current meteorological fix observation for a cyclone."""
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    wind_kts: float = Field(..., ge=0.0, le=300.0, description="1-minute sustained maximum wind in knots")
    pressure_hpa: float = Field(..., ge=800.0, le=1050.0, description="Estimated minimum central atmospheric pressure in hPa")
    timestamp: Optional[str] = Field(None, description="ISO-8601 observation timestamp")


class ForecastRequest(BaseModel):
    """Request payload for multi-step prognostic track and intensity extrapolation."""
    cyclone_id: str = Field(..., min_length=1, max_length=64, description="Standard cyclone identifier (e.g. ARB-2026-02)")
    observations: List[Observation] = Field(..., min_length=1, description="Sequential meteorological observations (earliest to latest)")


class TrajectoryPoint(BaseModel):
    """Prognostic storm waypoint projected by the recurrent temporal model."""
    tau_hours: int = Field(..., description="Prognostic lead time (+12h, +24h, +36h, +48h, +72h)")
    pred_lat: float = Field(..., description="Predicted latitude")
    pred_lon: float = Field(..., description="Predicted longitude")
    pred_wind_kts: float = Field(..., description="Predicted sustained surface wind in knots")
    pred_pressure_hpa: float = Field(..., description="Predicted central pressure in hPa")
    cone_radius_km: float = Field(..., description="90% track cone of uncertainty radius in km")
    is_landfall: bool = Field(False, description="True if cyclone center crosses coastline at this tau")
    landfall_location: Optional[str] = Field(None, description="Landmark or district name if coastal crossing occurs")


class CoastalIntercept(BaseModel):
    """Predicted cyclone coastal crossing location, ETA, and storm surge context."""
    lat: float
    lon: float
    location: str
    eta_hours: float
    confidence_window_hours: float = 2.5
    tidal_coincidence: str


class ForecastResponse(BaseModel):
    """Prognostic 72-hour forecast response output."""
    status: str = "success"
    cyclone_id: str
    prognostic_trajectory: List[TrajectoryPoint]
    intensity_class: str
    confidence_index: float
    rapid_intensification_detected: bool
    landfall_intercept: Optional[CoastalIntercept] = None


class TrajectoryObservation(BaseModel):
    """One observed fix for the multi-metric forecaster.

    Only position is required. Wind and central pressure are optional because live feeds
    carry neither per fix; whatever is missing is derived and declared in the response.
    """
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    wind_kt: Optional[float] = Field(None, ge=0.0, le=300.0, description="Sustained maximum wind in knots")
    pressure_hpa: Optional[float] = Field(None, ge=800.0, le=1050.0, description="Minimum central pressure in hPa")
    timestamp: Optional[str] = Field(None, description="ISO-8601 observation timestamp")


class TrajectoryRequest(BaseModel):
    """Request for a +6h to +48h track and intensity forecast."""
    cyclone_id: str = Field(..., min_length=1, max_length=64, description="Storm name or feed identifier")
    observations: List[TrajectoryObservation] = Field(
        ..., min_length=1, description="Sequential observations, earliest to latest"
    )


class TrajectoryWaypoint(BaseModel):
    """One forecast waypoint with its cone radius."""
    forecast_hour: int = Field(..., description="Lead time in hours (+6 .. +48)")
    lat: float
    lon: float
    wind_kt: float = Field(..., description="Predicted sustained wind in knots")
    pressure_hpa: float = Field(..., description="Predicted central pressure in hPa")
    uncertainty_radius_km: float = Field(
        ..., description="Cone radius from the checkpoint's measured validation RMSE at this lead"
    )
    dist2land_km: Optional[float] = Field(
        None, description="Distance to the nearest coastline, or null where the shipped coastline has no coverage"
    )


class CurrentMetrics(BaseModel):
    """The latest observed state the forecast starts from."""
    lat: float
    lon: float
    wind_kt: float
    pressure_hpa: float
    dist2land_km: Optional[float] = None


class LandfallEstimate(BaseModel):
    """Projected coastal crossing, or an explicit statement that none is projected."""
    projected: bool
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    eta_hours: Optional[float] = None
    note: str
    threshold_km: float


class ForecastSummary(BaseModel):
    """Multi-metric headline numbers for the dashboard panel."""
    peak_wind_kt: float = Field(..., description="Strongest wind anywhere on the forecast path")
    min_pressure_hpa: float = Field(..., description="Deepest central pressure on the path")
    peak_pressure_hpa: float = Field(..., description="Highest central pressure on the path")
    pressure_change_hpa: float = Field(..., description="Signed net change from now to +48 h")
    wind_change_kt: float = Field(..., description="Signed net change from now to +48 h")
    max_deepening_hpa: float = Field(..., description="Deepest point minus now; zero or negative")
    intensity_class: str
    forecast_track_km: float


class TrajectoryValidation(BaseModel):
    """What the error figures behind the cone actually are."""
    lead_track_rmse_km: List[float]
    metrics: Dict[str, Any] = {}


class TrajectoryResponse(BaseModel):
    """Multi-metric trajectory forecast for a North Indian Ocean cyclone."""
    status: str = "success"
    cyclone_id: str
    region: str
    forecast_generated_at: str
    source: str = Field(..., description="'trained-sequence-model' or 'physics-extrapolation'")
    model: str
    checkpoint: Optional[str] = None
    observations_used: int
    inputs_imputed: List[str] = Field(
        default_factory=list, description="Inputs the engine derived rather than received"
    )
    current_metrics: CurrentMetrics
    waypoints: List[TrajectoryWaypoint]
    landfall_estimate: LandfallEstimate
    summary: ForecastSummary
    validation: TrajectoryValidation
    disclaimer: str


class PatternRequest(BaseModel):
    """Request to classify tropical cyclone satellite imagery into Dvorak structural patterns."""
    preset_pattern: Optional[str] = Field("eye", description="Standard benchmark preset: 'eye', 'central_dense_overcast', 'curved_band', 'developing', 'sheared', 'dissipating', 'clear'")
    image_base64: Optional[str] = Field(None, description="Base64-encoded satellite swath (TIR1/VIS/WV)")
    explainability_engine: Optional[str] = Field("GradCAM_ViT", description="Explainable AI method: 'GradCAM_ViT' or 'IntegratedGradients'")


class PatternResponse(BaseModel):
    """Dvorak morphological classification output with Grad-CAM heatmap grid."""
    status: str = "success"
    pattern_predicted: str
    dvorak_taxonomy: str
    confidence: float
    probabilities: Dict[str, float]
    min_brightness_temp_kelvin: float
    estimated_central_pressure_hpa: float
    grad_cam_saliency_hash: str
    explanation: str
    grad_cam_grid: List[List[float]]
    disclaimer: str = "Operational Cyclone Research Decision Support · Verify against official IMD/RSMC advisories"


class WindMultiMetric(BaseModel):
    """A second, independent intensity estimate from an observed central pressure.

    Model A reads pixels only. This block sits beside its output so the two estimates can
    be compared; it never changes the image estimate.
    """
    observed_pressure_hpa: float = Field(..., description="Central pressure supplied by the user")
    pressure_derived_wind_kt: float = Field(..., description="Wind implied by that pressure (Atkinson-Holliday)")
    image_derived_pressure_hpa: float = Field(..., description="Pressure implied by the image wind estimate")
    image_minus_pressure_kt: float = Field(..., description="Image estimate minus pressure-derived estimate")
    agreement: str = Field(..., description="'close' within 10 kt, 'moderate' within 25 kt, else 'poor'")
    relation: str = "Atkinson-Holliday (1977) wind-pressure relationship for the North Indian Ocean"


class WindPredictionResponse(BaseModel):
    """Wind estimate response with raw science output and a separate display safeguard."""
    status: str = "success"
    model: str
    checkpoint: str
    unit: str = "kt"
    normalized_model_output: float
    raw_prediction_kt: float
    application_prediction_kt: float
    predicted_wind_kt: float
    category: str
    multi_metric: Optional[WindMultiMetric] = Field(
        None, description="Present only when a central pressure was supplied alongside the image"
    )
    disclaimer: str = "Research estimate from a satellite image; not an official warning or operational intensity estimate."


class HealthResponse(BaseModel):
    """System health, hardware state, and model inference telemetry."""
    status: str
    service: str
    version: str
    uptime_seconds: float
    hardware: Dict[str, Any]
    models_loaded: Dict[str, bool]
    telemetry: Dict[str, Any]
