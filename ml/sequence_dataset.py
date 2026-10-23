"""Sliding-window sequence pipeline for North Indian Ocean cyclone forecasting.

A window is four consecutive 6-hourly best-track fixes (t-18h, t-12h, t-6h, t0) paired
with eight forecast targets (+6h .. +48h). Windows are cut per storm and only from runs
of fixes that are exactly 6 hours apart, so a missing fix never gets bridged into a
sequence that was never observed.

Per-step tabular features are the metrics the forecast actually depends on: position,
recent motion, intensity, central pressure, and distance to land. Satellite chips are
optional and carried per step with a presence mask, because the repository's North
Indian Ocean chip coverage is far sparser than its track coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# pandas is imported inside the functions that build datasets rather than at module
# level. app/trajectory.py imports this module for its normalisation frame alone, and
# that path is deployed to a serverless function with a 250 MB budget: pandas is 67 MB
# of it and nothing in the serving path touches a DataFrame.

FEATURES: list[str] = [
    "lat",
    "lon",
    "dlat",
    "dlon",
    "wind_kt",
    "pres_hpa",
    "dist2land_km",
]

#: Input sequence length: t-18h, t-12h, t-6h, t0.
INPUT_STEPS = 4

#: Best-track sampling interval in hours. IBTrACS subsets in this repo are 6-hourly.
STEP_HOURS = 6

#: Forecast lead times in hours, +6h through +48h.
TARGET_LEADS: list[int] = [6, 12, 18, 24, 30, 36, 42, 48]

#: (centre, scale) per feature, chosen for the North Indian Ocean rather than fitted,
#: so that training, inference and any future refit all share one fixed frame.
NORM: dict[str, tuple[float, float]] = {
    "lat": (15.0, 8.0),
    "lon": (85.0, 10.0),
    "dlat": (0.0, 0.6),
    "dlon": (0.0, 0.6),
    "wind_kt": (60.0, 40.0),
    "pres_hpa": (985.0, 25.0),
    "dist2land_km": (300.0, 300.0),
}

#: Trajectory targets are offsets in degrees; this divisor keeps them near unit scale.
DELTA_SCALE_DEG = 5.0

#: (centre, scale) for the intensity targets, matching the NORM entries they mirror.
TARGET_NORM: dict[str, tuple[float, float]] = {
    "wind_kt": NORM["wind_kt"],
    "pres_hpa": NORM["pres_hpa"],
}

IMAGE_SIZE = 224

# Atkinson-Holliday (1977) wind-pressure relationship, the standard fallback in the
# North Indian Ocean: Vmax[kt] = 6.7 * (1010 - Pc[hPa]) ** 0.644.
_AH_COEFFICIENT = 6.7
_AH_EXPONENT = 0.644
_AH_ENVIRONMENTAL_HPA = 1010.0
PRESSURE_FLOOR_HPA = 880.0
PRESSURE_CEILING_HPA = 1012.0


def pressure_from_wind(wind_kt: float) -> float:
    """Estimate central pressure from sustained wind using Atkinson-Holliday.

    Used only to fill a gap left by the source data. Every caller must label the result
    as derived, never as an observation.
    """
    wind = max(float(wind_kt), 0.0)
    drop = (wind / _AH_COEFFICIENT) ** (1.0 / _AH_EXPONENT)
    return float(np.clip(_AH_ENVIRONMENTAL_HPA - drop, PRESSURE_FLOOR_HPA, PRESSURE_CEILING_HPA))


@dataclass(frozen=True)
class SequenceWindow:
    """One training example: an observed input sequence and its forecast targets."""

    sid: str
    name: str
    basin: str
    t0: "pd.Timestamp"
    features: np.ndarray  # (INPUT_STEPS, len(FEATURES)) in raw physical units
    target_coords: np.ndarray  # (len(TARGET_LEADS), 2) cumulative dlat/dlon from t0, degrees
    target_intensity: np.ndarray  # (len(TARGET_LEADS), 2) absolute wind_kt and pres_hpa
    image_paths: tuple[str | None, ...]
    was_imputed: dict[str, bool]


def load_tracks(path: str | Path) -> "pd.DataFrame":
    """Read an IBTrACS-derived tracks CSV with `time` parsed and rows ordered per storm."""
    import pandas as pd

    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.sort_values(["sid", "time"]).reset_index(drop=True)


def _image_lookup(images_index, channel: str) -> dict:
    import pandas as pd

    if images_index is None or len(images_index) == 0:
        return {}
    index = images_index.copy()
    index["time"] = pd.to_datetime(index["time"])
    if "channel" in index.columns:
        preferred = index[index["channel"] == channel]
        index = preferred if len(preferred) else index
    return {
        (str(row.sid), pd.Timestamp(row.time)): str(row.image_path)
        for row in index.itertuples()
    }


def _contiguous_runs(group, step_hours: int) -> list:
    """Split one storm's fixes into runs spaced exactly `step_hours` apart."""
    import pandas as pd

    if len(group) == 0:
        return []
    gaps = group["time"].diff()
    expected = pd.Timedelta(hours=step_hours)
    break_at = (gaps != expected) & gaps.notna()
    run_id = break_at.cumsum()
    return [run for _, run in group.groupby(run_id, sort=True)]


def build_sequence_windows(
    df: "pd.DataFrame",
    basins: Sequence[str] | None = ("NI",),
    input_steps: int = INPUT_STEPS,
    leads: Sequence[int] = tuple(TARGET_LEADS),
    images_index: "pd.DataFrame | None" = None,
    step_hours: int = STEP_HOURS,
    image_channel: str = "ir",
) -> list[SequenceWindow]:
    """Cut every valid sliding window out of a best-track frame.

    `basins=None` keeps all basins, which is how the global pre-training pass is built.
    A window is dropped when position or wind is missing anywhere in it; central pressure
    is imputed instead, because it is absent from many fixes and from every live feed.
    """
    import pandas as pd

    leads = list(leads)
    if df.empty:
        return []

    frame = df.copy()
    frame["time"] = pd.to_datetime(frame["time"])
    if basins is not None:
        frame = frame[frame["basin"].isin(list(basins))]
    if frame.empty:
        return []

    for column in ("dlat", "dlon", "pres_hpa", "dist2land_km"):
        if column not in frame.columns:
            frame[column] = np.nan

    lookup = _image_lookup(images_index, image_channel)
    lead_offsets = [lead // step_hours for lead in leads]
    span = input_steps + max(lead_offsets)
    windows: list[SequenceWindow] = []

    for sid, group in frame.sort_values(["sid", "time"]).groupby("sid", sort=True):
        for run in _contiguous_runs(group, step_hours):
            run = run.reset_index(drop=True)
            if len(run) < span:
                continue
            for start in range(len(run) - span + 1):
                window = _build_one_window(
                    run.iloc[start : start + span],
                    sid=str(sid),
                    input_steps=input_steps,
                    lead_offsets=lead_offsets,
                    lookup=lookup,
                )
                if window is not None:
                    windows.append(window)
    return windows


def _build_one_window(
    rows,
    sid: str,
    input_steps: int,
    lead_offsets: Sequence[int],
    lookup: dict,
) -> SequenceWindow | None:
    import pandas as pd

    rows = rows.reset_index(drop=True)
    inputs = rows.iloc[:input_steps]
    t0_index = input_steps - 1

    # Position and intensity must be observed; a gap there would invent the storm.
    required = rows[["lat", "lon", "wind_kt"]].to_numpy(dtype=np.float64)
    if not np.isfinite(required).all():
        return None

    was_imputed = {name: False for name in FEATURES}

    pressure_raw = rows["pres_hpa"].to_numpy(dtype=np.float64)
    pressure = pressure_raw.copy()
    missing_pressure = ~np.isfinite(pressure)
    if missing_pressure.any():
        was_imputed["pres_hpa"] = True
        observed = pressure[~missing_pressure]
        winds = rows["wind_kt"].to_numpy(dtype=np.float64)
        for position in np.flatnonzero(missing_pressure):
            earlier = pressure[:position][np.isfinite(pressure[:position])]
            if earlier.size:  # forward-fill from the most recent observed fix
                pressure[position] = earlier[-1]
            elif observed.size:
                pressure[position] = float(observed.mean())
            else:  # the whole window is missing pressure: derive it from wind
                pressure[position] = pressure_from_wind(winds[position])

    motion = {}
    for column in ("dlat", "dlon"):
        values = inputs[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            was_imputed[column] = True
            values = np.nan_to_num(values, nan=0.0)
        motion[column] = values

    distance = inputs["dist2land_km"].to_numpy(dtype=np.float64)
    if not np.isfinite(distance).all():
        was_imputed["dist2land_km"] = True
        observed = distance[np.isfinite(distance)]
        fill = float(observed.mean()) if observed.size else NORM["dist2land_km"][0]
        distance = np.nan_to_num(distance, nan=fill)

    columns = {
        "lat": inputs["lat"].to_numpy(dtype=np.float64),
        "lon": inputs["lon"].to_numpy(dtype=np.float64),
        "dlat": motion["dlat"],
        "dlon": motion["dlon"],
        "wind_kt": inputs["wind_kt"].to_numpy(dtype=np.float64),
        "pres_hpa": pressure[:input_steps],
        "dist2land_km": distance,
    }
    features = np.stack([columns[name] for name in FEATURES], axis=1)

    lat0 = float(rows.loc[t0_index, "lat"])
    lon0 = float(rows.loc[t0_index, "lon"])
    target_rows = [t0_index + offset for offset in lead_offsets]
    target_coords = np.array(
        [[float(rows.loc[i, "lat"]) - lat0, float(rows.loc[i, "lon"]) - lon0] for i in target_rows],
        dtype=np.float64,
    )
    target_intensity = np.array(
        [[float(rows.loc[i, "wind_kt"]), float(pressure[i])] for i in target_rows],
        dtype=np.float64,
    )
    if not (np.isfinite(features).all() and np.isfinite(target_coords).all() and np.isfinite(target_intensity).all()):
        return None

    image_paths = tuple(
        lookup.get((sid, pd.Timestamp(time))) for time in inputs["time"]
    )
    name = str(rows.loc[t0_index, "name"]) if "name" in rows.columns else sid
    basin = str(rows.loc[t0_index, "basin"]) if "basin" in rows.columns else "unknown"

    return SequenceWindow(
        sid=sid,
        name=name,
        basin=basin,
        t0=pd.Timestamp(rows.loc[t0_index, "time"]),
        features=features,
        target_coords=target_coords,
        target_intensity=target_intensity,
        image_paths=image_paths,
        was_imputed=was_imputed,
    )


def _norm_arrays(names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    centres = np.array([NORM[name][0] for name in names], dtype=np.float64)
    scales = np.array([NORM[name][1] for name in names], dtype=np.float64)
    return centres, scales


def normalise(values: np.ndarray, features: Sequence[str] = tuple(FEATURES)) -> np.ndarray:
    """Centre and scale raw feature columns into the model's fixed input frame."""
    centres, scales = _norm_arrays(features)
    return (np.asarray(values, dtype=np.float64) - centres) / scales


def denormalise(values: np.ndarray, features: Sequence[str] = tuple(FEATURES)) -> np.ndarray:
    """Invert :func:`normalise`."""
    centres, scales = _norm_arrays(features)
    return np.asarray(values, dtype=np.float64) * scales + centres


def normalise_intensity(values: np.ndarray) -> np.ndarray:
    """Centre and scale (wind_kt, pres_hpa) pairs."""
    return normalise(values, features=("wind_kt", "pres_hpa"))


def denormalise_intensity(values: np.ndarray) -> np.ndarray:
    """Invert :func:`normalise_intensity`."""
    return denormalise(values, features=("wind_kt", "pres_hpa"))


def split_by_storm(
    windows: Sequence[SequenceWindow],
    val_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[SequenceWindow], list[SequenceWindow]]:
    """Split windows so that no storm contributes to both sides.

    Windows overlap heavily inside one storm, so a random split would leak almost every
    validation target into training.
    """
    storms = sorted({window.sid for window in windows})
    if len(storms) < 2:
        return list(windows), []
    order = np.random.default_rng(seed).permutation(len(storms))
    held = max(1, min(len(storms) - 1, int(round(val_fraction * len(storms)))))
    val_sids = {storms[i] for i in order[:held]}
    train = [w for w in windows if w.sid not in val_sids]
    val = [w for w in windows if w.sid in val_sids]
    return train, val


class CycloneSequenceDataset:
    """Torch dataset over :class:`SequenceWindow` objects.

    Tabular features arrive normalised; trajectory and intensity targets stay in physical
    units so that losses and reported errors are in degrees, knots and hectopascals.
    Images are returned per step with a mask: a masked step carries zeros, and the model
    substitutes a learned embedding for it.
    """

    def __init__(
        self,
        windows: Sequence[SequenceWindow],
        image_root: str | Path | None = None,
        image_size: int = IMAGE_SIZE,
        with_images: bool = True,
    ):
        self.windows = list(windows)
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_size = image_size
        # Tabular-only training sets this False: an all-zero (T,3,224,224) tensor per item
        # is 2.4 MB of nothing, and the vision branch would ignore it anyway.
        self.with_images = bool(with_images)
        self._transform = None

    def __len__(self) -> int:
        return len(self.windows)

    def _load_image(self, relative_path: str):
        from PIL import Image, UnidentifiedImageError
        from torchvision import transforms as T

        if self._transform is None:
            self._transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        path = self.image_root / relative_path if self.image_root else Path(relative_path)
        try:
            with Image.open(path) as handle:
                handle.load()
                return self._transform(handle.convert("RGB"))
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return None  # a chip that will not open is treated as absent, not as zeros

    def __getitem__(self, index: int) -> dict:
        import torch

        window = self.windows[index]
        steps = window.features.shape[0]
        item = {
            "tabular": torch.tensor(normalise(window.features), dtype=torch.float32),
            "image_mask": torch.zeros(steps, dtype=torch.float32),
            "target_delta": torch.tensor(window.target_coords, dtype=torch.float32),
            "target_intensity": torch.tensor(window.target_intensity, dtype=torch.float32),
        }
        if not self.with_images:
            return item

        images = torch.zeros(steps, 3, self.image_size, self.image_size, dtype=torch.float32)
        if self.image_root is not None:
            for step, relative in enumerate(window.image_paths):
                if not relative:
                    continue
                tensor = self._load_image(relative)
                if tensor is not None:
                    images[step] = tensor
                    item["image_mask"][step] = 1.0
        item["image"] = images
        return item


def describe_windows(windows: Iterable[SequenceWindow]) -> dict:
    """Summary used by the training script's logs and by the dataset smoke check."""
    windows = list(windows)
    if not windows:
        return {"windows": 0, "storms": 0, "names": []}
    return {
        "windows": len(windows),
        "storms": len({w.sid for w in windows}),
        "basins": sorted({w.basin for w in windows}),
        "names": sorted({w.name for w in windows}),
        "first_t0": str(min(w.t0 for w in windows)),
        "last_t0": str(max(w.t0 for w in windows)),
        "pressure_imputed_windows": sum(1 for w in windows if w.was_imputed["pres_hpa"]),
    }
