"""Trajectory and intensity forecasting behind the CycloneSense API.

The engine runs the trained sequence model from ``ml/trajectory_model.py`` over the last
four 6-hourly observations of a storm and returns eight waypoints, +6h to +48h, each with
position, wind, central pressure and a cone radius.

Three rules hold throughout:

* cone radii come from the checkpoint's measured per-lead validation RMSE, never from a
  shape chosen to look convincing;
* any input the caller did not supply (central pressure is missing from every live feed
  this project reads) is derived from what was supplied and named in ``inputs_imputed``;
* with no checkpoint installed the engine still answers, but it says the numbers came
  from a straight-line extrapolation rather than from a model.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ml.sequence_dataset import (
    DELTA_SCALE_DEG,
    FEATURES,
    INPUT_STEPS,
    NORM,
    PRESSURE_CEILING_HPA,
    PRESSURE_FLOOR_HPA,
    STEP_HOURS,
    TARGET_LEADS,
    denormalise_intensity,
    normalise,
    pressure_from_wind,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_CHECKPOINT = PROJECT_ROOT / "models" / "bob_trajectory_gru.pt"
COASTLINE_PATH = PROJECT_ROOT / "public" / "coast.json"

EARTH_RADIUS_KM = 6371.0
WIND_CEILING_KT = 200.0

#: A storm centre this close to the coastline is treated as making landfall.
LANDFALL_THRESHOLD_KM = 35.0

#: Hours between the interpolated points the landfall scan walks along the forecast path.
LANDFALL_SCAN_STEP_H = 1.0

#: Per-lead RMSE of straight-line extrapolation on this repository's North Indian Ocean
#: validation storms, measured by ml/train_trajectory.py. Used as the cone for the
#: fallback path, so even the fallback's uncertainty is a measurement.
BASELINE_LEAD_RMSE_KM: list[float] = [30.7, 67.4, 111.7, 160.7, 210.8, 259.1, 309.8, 364.1]

#: Wind decay applied per 6-hourly step by the fallback once a projected position is
#: within DECAY_ONSET_KM of land. Overland weakening is the one effect a straight line
#: cannot be allowed to ignore.
FALLBACK_OVERLAND_DECAY = 0.86
DECAY_ONSET_KM = 60.0

_ADVISORY = "Not an official warning. Verify against IMD and RSMC New Delhi advisories."

#: Served when the extrapolation fallback produced the numbers. It has to deny the model
#: outright: the UI prints this line directly under the model name.
FALLBACK_DISCLAIMER = (
    "This is not a model forecast. With no trained checkpoint installed, the track is a "
    "straight-line extrapolation of the last observed motion vector and the cone is that "
    f"method's own measured error. {_ADVISORY}"
)


def trained_disclaimer(metadata: dict[str, Any]) -> str:
    """Describe the actual training and validation sets, read from the checkpoint.

    Counting from the checkpoint rather than hardcoding a number keeps this honest across
    retrains; a figure written here by hand drifts the moment the model is trained again.
    """
    trained_on = len(metadata.get("train_storms") or [])
    held_out = len(metadata.get("val_storms") or [])
    if trained_on and held_out:
        scope = (
            f"a sequence model trained on {trained_on} North Indian Ocean storms and "
            f"validated on {held_out} held-out storms"
        )
    else:
        scope = "a sequence model trained on North Indian Ocean best-track windows"
    return f"Research forecast from {scope}. {_ADVISORY}"


@dataclass(frozen=True)
class Box:
    """A latitude/longitude box, inclusive on every edge."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon


BAY_OF_BENGAL = Box(5.0, 25.0, 80.0, 95.0)
ARABIAN_SEA = Box(5.0, 27.0, 50.0, 80.0)
NORTH_INDIAN_OCEAN = Box(0.0, 32.0, 40.0, 100.0)


def classify_region(lat: float, lon: float) -> str:
    """Name the basin a position sits in, without claiming a storm is local when it is not."""
    if BAY_OF_BENGAL.contains(lat, lon):
        return "Bay of Bengal"
    if ARABIAN_SEA.contains(lat, lon):
        return "Arabian Sea"
    if NORTH_INDIAN_OCEAN.contains(lat, lon):
        return "North Indian Ocean"
    return "Outside the North Indian Ocean"


def wind_from_pressure(pressure_hpa: float) -> float:
    """Atkinson-Holliday forward relation: wind in knots from central pressure."""
    drop = max(0.0, 1010.0 - float(pressure_hpa))
    return float(np.clip(6.7 * (drop ** 0.644), 0.0, WIND_CEILING_KT))


def wrap_degrees(delta: float) -> float:
    """Shortest signed angular step, so a dateline crossing is 0.7 degrees, not -359.3."""
    return (float(delta) + 180.0) % 360.0 - 180.0


def wrap_longitude(lon: float) -> float:
    """Fold a longitude back into [-180, 180)."""
    return (float(lon) + 180.0) % 360.0 - 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two positions."""
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    inner = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, inner))))


#: Named stretches of North Indian Ocean coast, used only to label a projected crossing.
COASTAL_GAZETTEER: list[tuple[str, float, float]] = [
    ("West Bengal / Sundarbans coast", 21.65, 88.30),
    ("Bangladesh coast near Khulna", 22.25, 89.60),
    ("Bangladesh coast near Barishal", 22.20, 90.50),
    ("Bangladesh coast near Chattogram", 22.30, 91.80),
    ("Odisha coast near Paradip", 20.26, 86.67),
    ("Odisha coast near Puri", 19.80, 85.83),
    ("Odisha coast near Gopalpur", 19.26, 84.90),
    ("Andhra Pradesh coast near Visakhapatnam", 17.69, 83.22),
    ("Andhra Pradesh coast near Machilipatnam", 16.17, 81.14),
    ("Andhra Pradesh coast near Nellore", 14.45, 80.07),
    ("Tamil Nadu coast near Chennai", 13.08, 80.27),
    ("Tamil Nadu coast near Nagapattinam", 10.77, 79.84),
    ("Sri Lanka east coast near Trincomalee", 8.57, 81.23),
    ("Sri Lanka coast near Jaffna", 9.66, 80.01),
    ("Myanmar coast near Sittwe", 20.15, 92.90),
    ("Myanmar coast near Pathein", 16.30, 94.50),
    ("Myanmar coast near Yangon", 16.84, 96.15),
    ("Andaman Islands near Port Blair", 11.62, 92.73),
    ("Kerala coast near Kochi", 9.93, 76.27),
    ("Karnataka coast near Mangaluru", 12.91, 74.86),
    ("Maharashtra coast near Mumbai", 19.08, 72.88),
    ("Gujarat coast near Veraval", 20.90, 70.37),
    ("Gujarat / Kutch coast near Mandvi", 22.83, 69.35),
    ("Pakistan coast near Karachi", 24.86, 67.01),
    ("Oman coast near Salalah", 17.02, 54.09),
    ("Oman coast near Muscat", 23.59, 58.41),
    ("Somalia coast near Caluula", 11.97, 50.75),
]


#: A gazetteer entry further than this from the crossing describes somewhere else.
#: At 400 km, Great Nicobar was being reported as a landfall near Port Blair.
GAZETTEER_RADIUS_KM = 150.0


def nearest_coastal_place(lat: float, lon: float) -> str:
    """The gazetteer entry closest to a position, or a plain description if none is near."""
    name, distance = min(
        ((entry[0], haversine_km(lat, lon, entry[1], entry[2])) for entry in COASTAL_GAZETTEER),
        key=lambda pair: pair[1],
    )
    if distance > GAZETTEER_RADIUS_KM:
        return f"coast near {abs(lat):.1f}°{'N' if lat >= 0 else 'S'} {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
    return name


class CoastlineIndex:
    """Nearest-coastline distance from the Natural Earth coastline the map already ships.

    The model was trained on IBTrACS ``dist2land_km``; this reproduces that feature at
    serving time (correlation 0.945, median error 5.4 km over the North Indian Ocean
    fixes in this repository), so the input distribution at inference matches training.

    The shipped file is an **Eastern Hemisphere** coastline: it holds no American coast.
    No point on Earth lies more than about 2 690 km from land (Point Nemo), so a nearest
    vertex beyond :data:`COVERAGE_LIMIT_KM` means this file does not cover that position,
    and the distance is reported as unavailable instead of as a number that is wrong by
    thousands of kilometres. GDACS lists storms worldwide, so this really does happen.
    """

    #: Beyond this, a "nearest coast" result is evidence of missing data, not of distance.
    COVERAGE_LIMIT_KM = 2800.0

    #: Width of the longitude bins used to detect the file's data holes.
    COVERAGE_BIN_DEG = 5.0

    _shared: "CoastlineIndex | None" = None
    _shared_lock = threading.Lock()

    def __init__(self, path: str | Path = COASTLINE_PATH):
        self.path = Path(path)
        geometry = json.loads(self.path.read_text(encoding="utf-8"))
        lines: Iterable[Sequence[Sequence[float]]] = geometry.get("coordinates", [])
        points = np.array(list(itertools.chain(*lines)), dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError(f"{self.path} holds no coastline vertices.")
        self.lon = np.radians(points[:, 0])
        self.lat = np.radians(points[:, 1])
        self._lonlat = points
        # Which longitude bins hold any coastline at all. The shipped file has one
        # contiguous hole from 165W to 20W (the Americas), and a nearest-vertex answer
        # for a position inside it describes the wrong continent: off the Azores it
        # returns mainland Portugal, ~1900 km away, which passes the distance guard.
        edges = np.arange(-180.0, 180.0 + self.COVERAGE_BIN_DEG, self.COVERAGE_BIN_DEG)
        counts, _ = np.histogram(points[:, 0], bins=edges)
        self._bin_edges = edges
        self._bin_populated = counts > 0

    @classmethod
    def shared(cls) -> "CoastlineIndex":
        """One parsed copy per process; the file is 16 000 vertices of static geometry."""
        if cls._shared is None:
            with cls._shared_lock:
                if cls._shared is None:
                    cls._shared = cls()
        return cls._shared

    def _distances(self, lat: float, lon: float) -> np.ndarray:
        phi, lam = math.radians(lat), math.radians(lon)
        inner = (
            np.sin((self.lat - phi) / 2.0) ** 2
            + math.cos(phi) * np.cos(self.lat) * np.sin((self.lon - lam) / 2.0) ** 2
        )
        return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))

    def nearest(self, lat: float, lon: float) -> tuple[float, tuple[float, float]]:
        """Distance in km to the nearest coastline vertex, and that vertex."""
        distances = self._distances(lat, lon)
        index = int(np.argmin(distances))
        vertex = self._lonlat[index]
        return float(distances[index]), (float(vertex[1]), float(vertex[0]))

    def _longitude_is_mapped(self, lon: float) -> bool:
        index = int(np.searchsorted(self._bin_edges, wrap_longitude(lon), side="right")) - 1
        index = min(max(index, 0), len(self._bin_populated) - 1)
        return bool(self._bin_populated[index])

    def distance_to_land_km(self, lat: float, lon: float) -> float | None:
        """Distance in km to the nearest coast, or None where this file has no coverage.

        Two guards, because either alone lets a wrong number through: the longitude must
        fall in a band this file actually maps, and the nearest vertex must be inside the
        Point Nemo bound.
        """
        if not self._longitude_is_mapped(lon):
            return None
        nearest = float(np.min(self._distances(lat, lon)))
        return None if nearest > self.COVERAGE_LIMIT_KM else nearest

    def covers(self, lat: float, lon: float) -> bool:
        """True when this coastline can answer for a position at all."""
        return self.distance_to_land_km(lat, lon) is not None


@dataclass
class PreparedSequence:
    """The model-ready input sequence plus an account of what had to be filled in."""

    features: np.ndarray  # (INPUT_STEPS, len(FEATURES)) raw units
    lat0: float
    lon0: float
    wind0: float
    pressure0: float
    dist2land0: float | None
    observations_used: int
    imputed: list[str]


def _parse_timestamp(value: Any) -> float | None:
    """Observation time in epoch hours, or None when it is absent or unreadable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() / 3600.0


def _spacing_hours(earlier: dict[str, Any], later: dict[str, Any]) -> float | None:
    """Hours between two observations, or None when either time is unknown."""
    if earlier.get("at") is None or later.get("at") is None:
        return None
    gap = later["at"] - earlier["at"]
    return gap if gap > 0 else None


def _first_finite(values: Sequence[float | None]) -> float | None:
    for value in values:
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def prepare_sequence(
    observations: Sequence[dict[str, Any]],
    coastline: CoastlineIndex,
    input_steps: int = INPUT_STEPS,
) -> PreparedSequence:
    """Turn caller observations into the fixed-length input the model was trained on.

    Fewer than ``input_steps`` observations are padded at the front by repeating the
    earliest fix, which is declared rather than hidden: a one-observation request gets a
    forecast with no motion history, and the response says so.
    """
    if not observations:
        raise ValueError("At least one observation is required to forecast a trajectory.")

    cleaned: list[dict[str, Any]] = []
    for raw in observations:
        lat = raw.get("lat")
        lon = raw.get("lon")
        if lat is None or lon is None or not (math.isfinite(float(lat)) and math.isfinite(float(lon))):
            raise ValueError("Every observation needs a finite lat and lon.")
        wind = raw.get("wind_kt", raw.get("wind_kts"))
        pressure = raw.get("pressure_hpa", raw.get("pres_hpa"))
        cleaned.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "wind_kt": None if wind is None else float(wind),
                "pressure_hpa": None if pressure is None else float(pressure),
                "at": _parse_timestamp(raw.get("timestamp")),
            }
        )

    imputed: list[str] = []
    winds = [step["wind_kt"] for step in cleaned]
    pressures = [step["pressure_hpa"] for step in cleaned]
    if _first_finite(winds) is None and _first_finite(pressures) is None:
        raise ValueError(
            "At least one observation must carry wind_kt or pressure_hpa; "
            "intensity cannot be inferred from position alone."
        )

    # Fill wind and pressure from each other, then forward and backward within the track.
    for step in cleaned:
        if step["wind_kt"] is None and step["pressure_hpa"] is not None:
            step["wind_kt"] = wind_from_pressure(step["pressure_hpa"])
            if "wind_kt" not in imputed:
                imputed.append("wind_kt")
        if step["pressure_hpa"] is None and step["wind_kt"] is not None:
            step["pressure_hpa"] = pressure_from_wind(step["wind_kt"])
            if "pressure_hpa" not in imputed:
                imputed.append("pressure_hpa")
    for field in ("wind_kt", "pressure_hpa"):
        known = _first_finite([step[field] for step in cleaned])
        for index, step in enumerate(cleaned):
            if step[field] is None:
                step[field] = known
                if field not in imputed:
                    imputed.append(field)
            else:
                known = step[field]
        # Backward fill anything still unset at the front of the track.
        later = None
        for step in reversed(cleaned):
            if step[field] is None:
                step[field] = later
                if field not in imputed:
                    imputed.append(field)
            else:
                later = step[field]

    observations_used = len(cleaned)
    window = cleaned[-input_steps:]
    if len(window) < input_steps:
        window = [dict(window[0]) for _ in range(input_steps - len(window))] + window
        imputed.append("padded_input_steps")

    rows: list[list[float]] = []
    uncovered = False
    rescaled_from: list[float] = []
    for index, step in enumerate(window):
        if index == 0:
            previous = cleaned[-input_steps - 1] if len(cleaned) > input_steps else None
        else:
            previous = window[index - 1]
        if previous is None:
            dlat = dlon = 0.0
        else:
            # The model's motion features are displacement per 6 hours. When the caller
            # supplied timestamps, honour them: a 12-hourly feed otherwise hands the model
            # a step twice the size it was trained on, and every lead inherits the error.
            spacing = _spacing_hours(previous, step)
            factor = 1.0 if spacing is None else float(np.clip(STEP_HOURS / spacing, 0.25, 4.0))
            if spacing is not None and abs(spacing - STEP_HOURS) > 0.1 * STEP_HOURS:
                rescaled_from.append(spacing)
            dlat = (step["lat"] - previous["lat"]) * factor
            dlon = wrap_degrees(step["lon"] - previous["lon"]) * factor
        distance = coastline.distance_to_land_km(step["lat"], step["lon"])
        if distance is None:
            # Feed the model the neutral value for this feature rather than a wrong one.
            distance = NORM["dist2land_km"][0]
            uncovered = True
        values = {
            "lat": step["lat"],
            "lon": step["lon"],
            "dlat": dlat,
            "dlon": dlon,
            "wind_kt": step["wind_kt"],
            "pres_hpa": step["pressure_hpa"],
            "dist2land_km": distance,
        }
        rows.append([float(values[name]) for name in FEATURES])

    if uncovered:
        imputed.append(
            "dist2land_km unavailable: the shipped coastline does not cover this position, "
            f"so the model was given its neutral value of {NORM['dist2land_km'][0]:.0f} km"
        )
    if rescaled_from:
        worst = max(rescaled_from, key=lambda hours: abs(hours - STEP_HOURS))
        imputed.append(
            f"motion rescaled from {worst:.1f} h observation spacing to the model's "
            f"{STEP_HOURS} h step"
        )
    elif len(cleaned) > 1 and all(step["at"] is None for step in cleaned):
        imputed.append(
            f"observation spacing assumed to be {STEP_HOURS}-hourly (no timestamps supplied)"
        )

    last = window[-1]
    last_distance = coastline.distance_to_land_km(last["lat"], last["lon"])
    return PreparedSequence(
        features=np.array(rows, dtype=np.float64),
        lat0=last["lat"],
        lon0=last["lon"],
        wind0=float(last["wind_kt"]),
        pressure0=float(last["pressure_hpa"]),
        dist2land0=last_distance,
        observations_used=observations_used,
        imputed=imputed,
    )


def _round_or_none(value: float | None, digits: int = 1) -> float | None:
    """Keep an unavailable measurement as null all the way to the response."""
    return None if value is None else round(float(value), digits)


def _clamp_intensity(wind_kt: float, pressure_hpa: float) -> tuple[float, float]:
    wind = float(np.clip(wind_kt, 0.0, WIND_CEILING_KT))
    pressure = float(np.clip(pressure_hpa, PRESSURE_FLOOR_HPA, PRESSURE_CEILING_HPA))
    return wind, pressure


def _interpolate_path(
    lat0: float, lon0: float, waypoints: Sequence[dict[str, Any]], step_hours: float
) -> list[tuple[float, float, float]]:
    """Densify the forecast path to (hour, lat, lon) samples for the landfall scan."""
    anchors = [(0.0, lat0, lon0)] + [
        (float(point["forecast_hour"]), float(point["lat"]), float(point["lon"])) for point in waypoints
    ]
    samples: list[tuple[float, float, float]] = []
    for (hour_a, lat_a, lon_a), (hour_b, lat_b, lon_b) in zip(anchors, anchors[1:]):
        span = hour_b - hour_a
        if span <= 0:
            continue
        count = max(1, int(round(span / step_hours)))
        for index in range(1, count + 1):
            fraction = index / count
            samples.append(
                (hour_a + fraction * span, lat_a + fraction * (lat_b - lat_a), lon_a + fraction * (lon_b - lon_a))
            )
    return samples


def _landfall_estimate(
    lat0: float,
    lon0: float,
    waypoints: Sequence[dict[str, Any]],
    coastline: CoastlineIndex,
) -> dict[str, Any]:
    """First point on the forecast path that reaches the coast, if any does within 48 h."""
    start_distance = coastline.distance_to_land_km(lat0, lon0)
    if start_distance is None:
        return {
            "projected": False,
            "location": None,
            "lat": None,
            "lon": None,
            "eta_hours": None,
            "note": (
                "Landfall was not evaluated: the shipped coastline covers the Eastern "
                "Hemisphere only, and this storm is outside it."
            ),
            "threshold_km": LANDFALL_THRESHOLD_KM,
        }
    if start_distance <= LANDFALL_THRESHOLD_KM:
        return {
            "projected": True,
            "location": nearest_coastal_place(lat0, lon0),
            "lat": round(lat0, 2),
            "lon": round(lon0, 2),
            "eta_hours": 0.0,
            "note": "The storm centre is already at the coast at the latest observation.",
            "threshold_km": LANDFALL_THRESHOLD_KM,
        }

    for hour, lat, lon in _interpolate_path(lat0, lon0, waypoints, LANDFALL_SCAN_STEP_H):
        distance = coastline.distance_to_land_km(lat, lon)
        if distance is not None and distance <= LANDFALL_THRESHOLD_KM:
            return {
                "projected": True,
                "location": nearest_coastal_place(lat, lon),
                "lat": round(lat, 2),
                "lon": round(lon, 2),
                "eta_hours": round(hour, 1),
                "note": (
                    f"Projected crossing: the forecast centre comes within "
                    f"{LANDFALL_THRESHOLD_KM:.0f} km of the coastline."
                ),
                "threshold_km": LANDFALL_THRESHOLD_KM,
            }

    scanned = [
        (coastline.distance_to_land_km(lat, lon), hour)
        for hour, lat, lon in _interpolate_path(lat0, lon0, waypoints, LANDFALL_SCAN_STEP_H)
    ]
    covered = [(distance, hour) for distance, hour in scanned if distance is not None]
    if not covered:
        return {
            "projected": False, "location": None, "lat": None, "lon": None, "eta_hours": None,
            "note": (
                "Landfall was not evaluated: the forecast track leaves the area the shipped "
                "coastline covers."
            ),
            "threshold_km": LANDFALL_THRESHOLD_KM,
        }
    closest = min(covered)
    return {
        "projected": False,
        "location": None,
        "lat": None,
        "lon": None,
        "eta_hours": None,
        "note": (
            f"No coastal crossing within {max(TARGET_LEADS)} h. The forecast track stays "
            f"{closest[0]:.0f} km offshore at its closest, near +{closest[1]:.0f} h."
        ),
        "threshold_km": LANDFALL_THRESHOLD_KM,
    }


def _intensity_class(peak_wind_kt: float) -> str:
    """IMD intensity nomenclature for the peak wind along the forecast path."""
    for threshold, label in (
        (120, "Super Cyclonic Storm (SuCS)"),
        (90, "Extremely Severe Cyclonic Storm (ESCS)"),
        (64, "Very Severe Cyclonic Storm (VSCS)"),
        (48, "Severe Cyclonic Storm (SCS)"),
        (34, "Cyclonic Storm (CS)"),
        (28, "Deep Depression (DD)"),
    ):
        if peak_wind_kt >= threshold:
            return label
    return "Depression (D)"


class TrajectoryEngine:
    """Serves track and intensity forecasts, with or without a trained checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        coastline: CoastlineIndex | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_TRAJECTORY_CHECKPOINT
        #: Exported torch-free weights, preferred when present. The deployed API cannot
        #: import torch (534 MB against a 250 MB function limit), so serving reads these.
        self.exported_path = self.checkpoint_path.with_suffix(".npz")
        self._coastline = coastline
        self._model = None
        self._runtime = "none"
        self._meta: dict[str, Any] = {}
        self._load_attempted = False
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def coastline(self) -> CoastlineIndex:
        if self._coastline is None:
            self._coastline = CoastlineIndex.shared()
        return self._coastline

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True

            # Exported weights first: they need only numpy, so this is the path that runs
            # in deployment. The torch checkpoint is the local and training-time path.
            if self.exported_path.exists():
                try:
                    from ml.numpy_forecaster import NumpyForecaster

                    self._model, self._meta = NumpyForecaster.load(self.exported_path)
                    self._runtime = "numpy"
                    return
                except Exception as exc:
                    self._load_error = f"Could not load {self.exported_path}: {exc}"
                    LOGGER.warning("%s Trying the torch checkpoint.", self._load_error)

            if not self.checkpoint_path.exists():
                if not self._load_error:
                    self._load_error = (
                        f"No trajectory weights at {self.exported_path} or {self.checkpoint_path}."
                    )
                LOGGER.info("%s Falling back to straight-line extrapolation.", self._load_error)
                return
            try:
                from ml.trajectory_model import load_forecaster

                self._model, self._meta = load_forecaster(self.checkpoint_path)
                self._runtime = "torch"
            except Exception as exc:  # a bad checkpoint must not take the API down
                self._model, self._meta, self._runtime = None, {}, "none"
                self._load_error = f"Could not load {self.checkpoint_path}: {exc}"
                LOGGER.warning("%s Falling back to straight-line extrapolation.", self._load_error)

    @property
    def available(self) -> bool:
        """True when a trained checkpoint is loaded and inference will use it."""
        self._ensure_loaded()
        return self._model is not None

    @property
    def runtime(self) -> str:
        """Which inference path is live: "numpy", "torch", or "none" for the fallback."""
        self._ensure_loaded()
        return self._runtime

    @property
    def metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._meta)

    def lead_radii_km(self) -> list[float]:
        """Cone radius per lead: the checkpoint's measured error, else the baseline's."""
        self._ensure_loaded()
        measured = list(self._meta.get("lead_track_rmse_km") or [])
        if len(measured) == len(TARGET_LEADS) and all(value > 0 for value in measured):
            return [float(value) for value in measured]
        return list(BASELINE_LEAD_RMSE_KM)

    def _model_deltas(self, prepared: PreparedSequence) -> tuple[np.ndarray, np.ndarray]:
        """Run whichever runtime is loaded. Both are pinned to the same outputs by test."""
        features = normalise(prepared.features)[None, ...].astype(np.float32)
        if self._runtime == "numpy":
            outputs = self._model.forward(features)
            delta = np.asarray(outputs["delta"])[0]
            intensity = np.asarray(outputs["intensity"])[0]
        else:
            import torch

            with torch.inference_mode():
                predicted = self._model(torch.tensor(features))
            delta = predicted["delta"][0].cpu().numpy()
            intensity = predicted["intensity"][0].cpu().numpy()
        return delta * DELTA_SCALE_DEG, denormalise_intensity(intensity)

    def _extrapolated_deltas(self, prepared: PreparedSequence) -> tuple[np.ndarray, np.ndarray]:
        """Straight-line motion with overland weakening. The labelled fallback."""
        dlat = float(prepared.features[-1, FEATURES.index("dlat")])
        dlon = float(prepared.features[-1, FEATURES.index("dlon")])
        if dlat == 0.0 and dlon == 0.0 and prepared.features.shape[0] > 1:
            dlat = float(np.mean(prepared.features[1:, FEATURES.index("dlat")]))
            dlon = float(np.mean(prepared.features[1:, FEATURES.index("dlon")]))

        deltas = np.zeros((len(TARGET_LEADS), 2), dtype=np.float64)
        intensity = np.zeros((len(TARGET_LEADS), 2), dtype=np.float64)
        wind = prepared.wind0
        # Anchor pressure to the observed value and move it only by the change the wind
        # change implies. Recomputing it outright would report a trend this method does
        # not predict: Atkinson-Holliday runs several hPa below this basin's own fixes, so
        # a constant-wind forecast came out as a 9 hPa deepening.
        reference = pressure_from_wind(prepared.wind0)
        for step in range(len(TARGET_LEADS)):
            deltas[step] = ((step + 1) * dlat, (step + 1) * dlon)
            projected_lat = prepared.lat0 + deltas[step][0]
            projected_lon = wrap_longitude(prepared.lon0 + deltas[step][1])
            distance = self.coastline.distance_to_land_km(projected_lat, projected_lon)
            if distance is not None and distance < DECAY_ONSET_KM:
                wind *= FALLBACK_OVERLAND_DECAY
            pressure = prepared.pressure0 + (pressure_from_wind(wind) - reference)
            intensity[step] = (wind, pressure)
        return deltas, intensity

    def forecast(
        self,
        cyclone_id: str,
        observations: Sequence[dict[str, Any]],
        generated_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Forecast position, wind and pressure at +6h through +48h."""
        prepared = prepare_sequence(observations, self.coastline)
        self._ensure_loaded()

        if self._model is not None:
            deltas, intensity = self._model_deltas(prepared)
            source = "trained-sequence-model"
            model_name = (
                "BayOfBengalCycloneForecaster "
                f"({self._meta.get('architecture', 'bob-gru')}, "
                f"basins={self._meta.get('basins', 'NI')})"
            )
            disclaimer = trained_disclaimer(self._meta)
        else:
            deltas, intensity = self._extrapolated_deltas(prepared)
            source = "physics-extrapolation"
            model_name = "Straight-line extrapolation of the last observed motion vector"
            disclaimer = FALLBACK_DISCLAIMER

        radii = self.lead_radii_km()
        waypoints: list[dict[str, Any]] = []
        for index, lead in enumerate(TARGET_LEADS):
            lat = prepared.lat0 + float(deltas[index][0])
            lon = wrap_longitude(prepared.lon0 + float(deltas[index][1]))
            wind, pressure = _clamp_intensity(float(intensity[index][0]), float(intensity[index][1]))
            waypoints.append(
                {
                    "forecast_hour": int(lead),
                    "lat": round(lat, 2),
                    "lon": round(lon, 2),
                    "wind_kt": round(wind, 1),
                    "pressure_hpa": round(pressure, 1),
                    "uncertainty_radius_km": round(float(radii[index]), 1),
                    "dist2land_km": _round_or_none(self.coastline.distance_to_land_km(lat, lon)),
                }
            )

        winds_along_path = [point["wind_kt"] for point in waypoints]
        pressures_along_path = [point["pressure_hpa"] for point in waypoints]
        peak_wind = max(winds_along_path + [prepared.wind0])
        min_pressure = min(pressures_along_path + [prepared.pressure0])
        final = waypoints[-1]
        track_km = 0.0
        previous = (prepared.lat0, prepared.lon0)
        for point in waypoints:
            track_km += haversine_km(previous[0], previous[1], point["lat"], point["lon"])
            previous = (point["lat"], point["lon"])

        return {
            "status": "success",
            "cyclone_id": cyclone_id,
            "region": classify_region(prepared.lat0, prepared.lon0),
            "forecast_generated_at": (generated_at or datetime.now(timezone.utc)).isoformat(),
            "source": source,
            "model": model_name,
            "runtime": self._runtime,
            "checkpoint": self.checkpoint_path.name if self._model is not None else None,
            "observations_used": prepared.observations_used,
            "inputs_imputed": prepared.imputed,
            "current_metrics": {
                "lat": round(prepared.lat0, 2),
                "lon": round(prepared.lon0, 2),
                "wind_kt": round(prepared.wind0, 1),
                "pressure_hpa": round(prepared.pressure0, 1),
                "dist2land_km": _round_or_none(prepared.dist2land0),
            },
            "waypoints": waypoints,
            "landfall_estimate": _landfall_estimate(
                prepared.lat0, prepared.lon0, waypoints, self.coastline
            ),
            "summary": {
                "peak_wind_kt": round(peak_wind, 1),
                "min_pressure_hpa": round(min_pressure, 1),
                "peak_pressure_hpa": round(max(pressures_along_path + [prepared.pressure0]), 1),
                # Net change over the whole forecast, signed: a storm that fills after
                # landfall must not report the same zero as one that never changes.
                "pressure_change_hpa": round(final["pressure_hpa"] - prepared.pressure0, 1),
                "wind_change_kt": round(final["wind_kt"] - prepared.wind0, 1),
                # Deepest point reached anywhere on the path, which is what a warning
                # cares about even when the storm fills again by +48 h.
                "max_deepening_hpa": round(min_pressure - prepared.pressure0, 1),
                "intensity_class": _intensity_class(peak_wind),
                "forecast_track_km": round(track_km, 1),
            },
            "validation": {
                "lead_track_rmse_km": [round(value, 1) for value in radii],
                "metrics": self._meta.get("metrics", {}),
            },
            "disclaimer": disclaimer,
        }

    def legacy_forecast(self, cyclone_id: str, observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """The /predict/forecast shape, driven by the same model.

        That endpoint publishes +12h to +72h. The model is trained to +48h, so the +72h
        point continues the +42h-to-+48h motion vector and its cone is widened; the
        response marks it as extrapolated past the trained horizon.
        """
        if not observations:
            # This used to substitute a hardcoded Arabian Sea storm and return it as a
            # successful forecast for whatever id the caller passed.
            raise ValueError("At least one observation is required to forecast a trajectory.")
        forecast = self.forecast(cyclone_id, observations)
        by_hour = {point["forecast_hour"]: point for point in forecast["waypoints"]}

        trajectory: list[dict[str, Any]] = []
        for tau in (12, 24, 36, 48):
            point = by_hour[tau]
            trajectory.append(
                {
                    "tau_hours": tau,
                    "pred_lat": point["lat"],
                    "pred_lon": point["lon"],
                    "pred_wind_kts": point["wind_kt"],
                    "pred_pressure_hpa": point["pressure_hpa"],
                    "cone_radius_km": point["uncertainty_radius_km"],
                    "is_landfall": False,
                    "landfall_location": None,
                    "extrapolated_beyond_training": False,
                }
            )

        at42, at48 = by_hour[42], by_hour[48]
        steps_beyond = 4  # +72 h is four 6-hourly steps past the trained horizon
        lat72 = at48["lat"] + steps_beyond * (at48["lat"] - at42["lat"])
        lon72 = wrap_longitude(at48["lon"] + steps_beyond * wrap_degrees(at48["lon"] - at42["lon"]))
        wind72 = max(0.0, at48["wind_kt"] - 15.0)  # unconstrained beyond training; weaken
        trajectory.append(
            {
                "tau_hours": 72,
                "pred_lat": round(lat72, 2),
                "pred_lon": round(lon72, 2),
                "pred_wind_kts": round(wind72, 1),
                "pred_pressure_hpa": round(pressure_from_wind(wind72), 1),
                "cone_radius_km": round(at48["uncertainty_radius_km"] * 1.45, 1),
                "is_landfall": False,
                "landfall_location": None,
                "extrapolated_beyond_training": True,
            }
        )

        landfall = forecast["landfall_estimate"]
        for point in trajectory:
            if landfall["projected"] and landfall["eta_hours"] is not None and point["tau_hours"] >= landfall["eta_hours"]:
                point["is_landfall"] = True
                point["landfall_location"] = landfall["location"]
                break

        # Only observations that actually carry a wind can evidence intensification.
        # Coercing a missing wind to 0.0 made a pressure-only first fix look like a 60 kt
        # jump; it also made a genuine 0 kt indistinguishable from no measurement.
        winds = [
            float(value)
            for value in (
                observation.get("wind_kt", observation.get("wind_kts")) for observation in observations
            )
            if value is not None and math.isfinite(float(value))
        ]
        rapid_intensification = len(winds) >= 2 and (winds[-1] - winds[0]) >= 25.0

        intercept = None
        if landfall["projected"] and landfall["lat"] is not None:
            intercept = {
                "lat": landfall["lat"],
                "lon": landfall["lon"],
                "location": landfall["location"],
                "eta_hours": landfall["eta_hours"],
                "confidence_window_hours": 3.0,
                "tidal_coincidence": (
                    "Not evaluated: tide and storm-surge coincidence need a tide model "
                    "this project does not include."
                ),
            }

        return {
            "status": "success",
            "cyclone_id": cyclone_id,
            "prognostic_trajectory": trajectory,
            "intensity_class": forecast["summary"]["intensity_class"],
            "confidence_index": round(_confidence_index(forecast), 3),
            "rapid_intensification_detected": rapid_intensification,
            "landfall_intercept": intercept,
        }


def _confidence_index(forecast: dict[str, Any]) -> float:
    """A reported confidence must come from measured error, not from a flattering constant.

    Scaled from the +24h cone: 150 km of validation RMSE maps to roughly 0.5.
    """
    radii = forecast["validation"]["lead_track_rmse_km"]
    at24 = radii[TARGET_LEADS.index(24)] if len(radii) == len(TARGET_LEADS) else 160.0
    penalty = 0.25 if forecast["source"] != "trained-sequence-model" else 0.0
    return float(np.clip(1.0 - at24 / 300.0 - penalty, 0.05, 0.95))


#: Process-wide engine used by the API. Construct your own to point at another checkpoint.
engine = TrajectoryEngine()
