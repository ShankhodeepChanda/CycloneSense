"""Train the Bay of Bengal track and intensity forecaster.

Trajectory error is measured where it matters - in kilometres on the sphere - so the
loss is great-circle distance between the predicted and observed positions, with an MAE
term on the raw degree offsets to keep early gradients well behaved. Intensity is a
mean-squared error on normalised wind and pressure.

Validation is grouped by storm: windows inside one storm overlap by up to 11 of their 12
fixes, so a random split would put almost every validation target in training too. The
per-lead validation RMSE is written into the checkpoint, because that is what the API
turns into a cone of uncertainty.

Two-stage run (what produced the shipped checkpoint). Stage one deliberately excludes
the North Indian Ocean, by storm and not merely by basin: PABUK, BULBUL and KROVANH each
carry both NI and WP rows, so a plain `--basins WP,...` filter would still train stage one
on storms stage two holds out. The resulting validation RMSE is what the API publishes as
its cone of uncertainty, so :func:`train` refuses that combination outright.

    python -m ml.train_trajectory --basins WP,SI,EP,SP --exclude-storms-in-basins NI \
        --epochs 45 --batch-size 64 --checkpoint models/bob_trajectory_global.pt
    python -m ml.train_trajectory --basins NI --epochs 200 --lr 5e-4 --patience 60 \
        --init-from models/bob_trajectory_global.pt \
        --checkpoint models/bob_trajectory_gru.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ml.sequence_dataset import (
    DELTA_SCALE_DEG,
    FEATURES,
    NORM,
    TARGET_LEADS,
    CycloneSequenceDataset,
    SequenceWindow,
    build_sequence_windows,
    denormalise_intensity,
    describe_windows,
    load_tracks,
    normalise_intensity,
    split_by_storm,
)
from ml.trajectory_model import BayOfBengalCycloneForecaster, load_forecaster, save_forecaster_checkpoint

EARTH_RADIUS_KM = 6371.0
DEFAULT_CHECKPOINT = Path("models/bob_trajectory_gru.pt")
DEFAULT_METRICS = Path("models/bob_trajectory_metrics.json")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres. Scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2))
    inner = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def haversine_km_torch(
    lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor
) -> torch.Tensor:
    """Differentiable great-circle distance in kilometres."""
    to_rad = math.pi / 180.0
    lat1, lon1, lat2, lon2 = (value * to_rad for value in (lat1, lon1, lat2, lon2))
    inner = torch.sin((lat2 - lat1) / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2.0) ** 2
    # Clamp away from 1 so the arcsin derivative stays finite at coincident points.
    return 2.0 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(inner.clamp(1e-12, 1.0 - 1e-9)))


def origin_from_tabular(tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover raw (lat, lon) at t0 from a normalised input batch."""
    lat_centre, lat_scale = NORM["lat"]
    lon_centre, lon_scale = NORM["lon"]
    last = tabular[:, -1, :]
    lat = last[:, FEATURES.index("lat")] * lat_scale + lat_centre
    lon = last[:, FEATURES.index("lon")] * lon_scale + lon_centre
    return lat, lon


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    track_weight: float = 1.0,
    intensity_weight: float = 1.0,
    degree_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Track error in km plus a degree-space MAE, and MSE on normalised intensity."""
    lat0, lon0 = origin_from_tabular(batch["tabular"])
    lat0 = lat0.unsqueeze(1)
    lon0 = lon0.unsqueeze(1)

    predicted_delta = outputs["delta"] * DELTA_SCALE_DEG
    target_delta = batch["target_delta"]
    distance_km = haversine_km_torch(
        lat0 + predicted_delta[..., 0],
        lon0 + predicted_delta[..., 1],
        lat0 + target_delta[..., 0],
        lon0 + target_delta[..., 1],
    )
    track_km = distance_km.mean()
    degree_mae = (predicted_delta - target_delta).abs().mean()

    target_intensity = torch.as_tensor(
        normalise_intensity(batch["target_intensity"].detach().cpu().numpy()),
        dtype=outputs["intensity"].dtype,
        device=outputs["intensity"].device,
    )
    intensity = nn.functional.mse_loss(outputs["intensity"], target_intensity)

    total = track_weight * (track_km / 100.0) + degree_weight * degree_mae + intensity_weight * intensity
    return {"total": total, "track_km": track_km, "degree_mae": degree_mae, "intensity": intensity}


def run_epoch(
    model: BayOfBengalCycloneForecaster,
    loader,
    optimiser: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """One pass over the loader. Trains when an optimiser is given, else evaluates."""
    training = optimiser is not None
    model.train(training)
    totals: dict[str, float] = {}
    seen = 0
    for batch in loader:
        size = batch["tabular"].shape[0]
        with torch.set_grad_enabled(training):
            outputs = model(batch["tabular"], batch.get("image"), batch.get("image_mask"))
            losses = compute_losses(outputs, batch)
        if training:
            optimiser.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * size
        seen += size
    return {key: value / max(1, seen) for key, value in totals.items()}


def _predict_windows(
    model: BayOfBengalCycloneForecaster,
    windows: Sequence[SequenceWindow],
    batch_size: int = 64,
    with_images: bool = False,
    image_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted (delta degrees, intensity) arrays aligned with `windows`."""
    dataset = CycloneSequenceDataset(windows, image_root=image_root, with_images=with_images)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    deltas: list[np.ndarray] = []
    intensities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["tabular"], batch.get("image"), batch.get("image_mask"))
            deltas.append((outputs["delta"] * DELTA_SCALE_DEG).cpu().numpy())
            intensities.append(denormalise_intensity(outputs["intensity"].cpu().numpy()))
    return np.concatenate(deltas, axis=0), np.concatenate(intensities, axis=0)


def _per_lead_report(
    windows: Sequence[SequenceWindow],
    predicted_delta: np.ndarray,
    predicted_intensity: np.ndarray | None,
) -> dict[str, Any]:
    origin = np.array([[w.features[-1, FEATURES.index("lat")], w.features[-1, FEATURES.index("lon")]] for w in windows])
    truth_delta = np.stack([w.target_coords for w in windows])
    lat0 = origin[:, 0:1]
    lon0 = origin[:, 1:2]
    error_km = haversine_km(
        lat0 + predicted_delta[..., 0],
        lon0 + predicted_delta[..., 1],
        lat0 + truth_delta[..., 0],
        lon0 + truth_delta[..., 1],
    )
    report: dict[str, Any] = {
        "leads": list(TARGET_LEADS),
        "windows": len(windows),
        "storms": sorted({window.sid for window in windows}),
        "track_rmse_km": [float(np.sqrt(np.mean(error_km[:, k] ** 2))) for k in range(error_km.shape[1])],
        "track_mae_km": [float(np.mean(np.abs(error_km[:, k]))) for k in range(error_km.shape[1])],
    }
    if predicted_intensity is None:
        report["wind_mae_kt"] = [float("nan")] * len(TARGET_LEADS)
        report["pressure_mae_hpa"] = [float("nan")] * len(TARGET_LEADS)
        return report
    truth_intensity = np.stack([w.target_intensity for w in windows])
    difference = np.abs(predicted_intensity - truth_intensity)
    report["wind_mae_kt"] = [float(np.mean(difference[:, k, 0])) for k in range(difference.shape[1])]
    report["pressure_mae_hpa"] = [float(np.mean(difference[:, k, 1])) for k in range(difference.shape[1])]
    return report


def evaluate_per_lead(
    model: BayOfBengalCycloneForecaster,
    windows: Sequence[SequenceWindow],
    batch_size: int = 64,
    with_images: bool = False,
    image_root: str | Path | None = None,
) -> dict[str, Any]:
    """Track and intensity error at every forecast lead, in km, knots and hectopascals."""
    if not windows:
        raise ValueError("Cannot evaluate on an empty window set.")
    delta, intensity = _predict_windows(model, windows, batch_size, with_images, image_root)
    return _per_lead_report(windows, delta, intensity)


def linear_baseline_per_lead(windows: Sequence[SequenceWindow]) -> dict[str, Any]:
    """The forecast to beat: hold the last observed 6-hourly motion vector constant.

    This is the honest reference for a track model. A model that cannot beat straight-line
    extrapolation has learned nothing about how these storms actually curve.
    """
    if not windows:
        raise ValueError("Cannot score a baseline on an empty window set.")
    steps = np.array([lead // 6 for lead in TARGET_LEADS], dtype=np.float64)
    predicted = np.stack(
        [
            np.stack(
                [
                    steps * window.features[-1, FEATURES.index("dlat")],
                    steps * window.features[-1, FEATURES.index("dlon")],
                ],
                axis=1,
            )
            for window in windows
        ]
    )
    report = _per_lead_report(windows, predicted, None)
    report["method"] = "persistence of the last 6-hourly motion vector"
    return report


@dataclass
class TrainConfig:
    """Everything one training run needs. Defaults match the shipped checkpoint."""

    tracks: str | Path = "kaggle_trial/tracks.csv"
    images_index: str | Path | None = None
    image_root: str | Path | None = None
    basins: Sequence[str] | None = ("NI",)
    #: Drop every storm that has any fix in these basins. A basin filter alone does not
    #: hold a storm out: PABUK, BULBUL and KROVANH each carry both NI and WP rows, so a
    #: WP pretrain would train on storms an NI fine-tune holds out.
    exclude_storms_in_basins: Sequence[str] | None = None
    use_vision: bool = False
    pretrained_vision: bool = True
    epochs: int = 120
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.25
    seed: int = 0
    checkpoint: str | Path = DEFAULT_CHECKPOINT
    metrics_path: str | Path = DEFAULT_METRICS
    init_from: str | Path | None = None
    allow_pretrain_overlap: bool = False
    max_windows: int | None = None
    patience: int = 40
    log_every: int = 10
    extra: dict[str, Any] = field(default_factory=dict)


def _load_windows(config: TrainConfig) -> list[SequenceWindow]:
    import pandas as pd

    frame = load_tracks(config.tracks)
    if config.exclude_storms_in_basins:
        excluded = set(frame[frame["basin"].isin(list(config.exclude_storms_in_basins))]["sid"])
        frame = frame[~frame["sid"].isin(excluded)]
    index = pd.read_csv(config.images_index) if config.images_index else None
    windows = build_sequence_windows(frame, basins=config.basins, images_index=index)
    if not windows:
        basins = "all basins" if config.basins is None else ", ".join(config.basins)
        raise ValueError(
            f"{config.tracks} yielded no sequence windows for {basins}. "
            "A storm needs 12 consecutive 6-hourly fixes to form one window."
        )
    if config.max_windows is not None and len(windows) > config.max_windows:
        keep = np.random.default_rng(config.seed).permutation(len(windows))[: config.max_windows]
        windows = [windows[i] for i in sorted(keep)]
    return windows


def train(config: TrainConfig) -> dict[str, Any]:
    """Run one training stage and write the checkpoint plus a metrics JSON."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    windows = _load_windows(config)
    train_windows, val_windows = split_by_storm(windows, config.val_fraction, config.seed)
    if not val_windows:
        # A single-storm dataset cannot be split. Scoring on the training set is the only
        # option left, and metrics["validation_is_training_set"] records it so nothing
        # downstream mistakes training error for a held-out estimate.
        val_windows = train_windows

    validation_is_training_set = val_windows is train_windows

    model = BayOfBengalCycloneForecaster(use_vision=config.use_vision, pretrained=config.pretrained_vision)
    initialised_from = None
    pretrain_overlap: list[str] = []
    if config.init_from:
        source, source_meta = load_forecaster(config.init_from)
        if source.use_vision != model.use_vision:
            raise ValueError(
                f"{config.init_from} was trained with use_vision={source.use_vision}; "
                f"this run asks for use_vision={model.use_vision}."
            )
        # The validation RMSE from this run becomes the API's cone radius, so a storm the
        # pretrain already fitted must not be held out here. Refuse rather than report a
        # number that understates the uncertainty a user is shown.
        seen_in_pretrain = set(source_meta.get("train_storms") or [])
        pretrain_overlap = sorted({window.sid for window in val_windows} & seen_in_pretrain)
        if pretrain_overlap and not config.allow_pretrain_overlap:
            raise ValueError(
                f"{len(pretrain_overlap)} of this run's validation storms were already "
                f"trained on by {config.init_from}: {', '.join(pretrain_overlap)}. "
                "The resulting validation error is not out-of-sample, and it is published "
                "as the forecast cone. Exclude this basin from the pretrain (for example "
                "--basins WP,SI,EP,SP) or pass --allow-pretrain-overlap deliberately."
            )
        model.load_state_dict(source.state_dict(), strict=True)
        initialised_from = str(config.init_from)

    with_images = config.use_vision and config.image_root is not None
    make_loader = lambda subset, shuffle: torch.utils.data.DataLoader(  # noqa: E731
        CycloneSequenceDataset(subset, image_root=config.image_root, with_images=with_images),
        batch_size=config.batch_size,
        shuffle=shuffle,
    )
    train_loader = make_loader(train_windows, True)
    val_loader = make_loader(val_windows, False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(1, config.epochs))

    history: list[dict[str, float]] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_track_km = float("inf")
    since_best = 0
    for epoch in range(1, config.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimiser)
        val_stats = run_epoch(model, val_loader, None)
        schedule.step()
        history.append({"epoch": epoch, "train_total": train_stats["total"], "val_track_km": val_stats["track_km"]})
        if val_stats["track_km"] < best_track_km - 1e-6:
            best_track_km = val_stats["track_km"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= config.patience:
                break
        if config.log_every and epoch % config.log_every == 0:
            print(
                f"epoch {epoch:4d}  train {train_stats['total']:.4f}  "
                f"val track {val_stats['track_km']:7.1f} km  best {best_track_km:7.1f} km"
            )

    model.load_state_dict(best_state)
    validation = evaluate_per_lead(model, val_windows, config.batch_size, with_images, config.image_root)
    baseline = linear_baseline_per_lead(val_windows)
    metrics = {
        "val_track_mae_km": float(np.mean(validation["track_mae_km"])),
        "val_track_rmse_km": float(np.mean(validation["track_rmse_km"])),
        "val_wind_mae_kt": float(np.nanmean(validation["wind_mae_kt"])),
        "val_pressure_mae_hpa": float(np.nanmean(validation["pressure_mae_hpa"])),
        "baseline_track_mae_km": float(np.mean(baseline["track_mae_km"])),
        "epochs_run": len(history),
        "best_val_track_km": best_track_km,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        # The weights are chosen by lowest validation error and then scored on that same
        # set, so these figures are selection-biased, not a clean held-out estimate. A
        # consumer that publishes them (the API does, as the cone) should know.
        "model_selected_on_validation_set": True,
        "validation_is_training_set": bool(validation_is_training_set),
        "pretrain_overlap_storms": len(pretrain_overlap),
    }

    save_forecaster_checkpoint(
        model,
        config.checkpoint,
        lead_track_rmse_km=validation["track_rmse_km"],
        metrics=metrics,
        train_storms=sorted({w.sid for w in train_windows}),
        val_storms=sorted({w.sid for w in val_windows}),
        basins=list(config.basins) if config.basins else "all",
        initialised_from=initialised_from,
    )

    result = {
        "checkpoint": str(config.checkpoint),
        "initialised_from": initialised_from,
        "train_windows": len(train_windows),
        "val_windows": len(val_windows),
        "dataset": describe_windows(windows),
        "validation": validation,
        "baseline": baseline,
        "metrics": metrics,
        "history": history,
    }
    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({k: v for k, v in result.items() if k != "history"} | {"history": history[-20:]}, indent=2))
    return result


def _parse_basins(value: str | None) -> Sequence[str] | None:
    if value is None or value.strip().lower() in {"all", "*"}:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", default="kaggle_trial/tracks.csv")
    parser.add_argument("--images-index", default=None, help="CSV mapping storm fixes to satellite chips")
    parser.add_argument("--image-root", default=None, help="Directory the image paths are relative to")
    parser.add_argument("--basins", default="NI", help="Comma-separated basin codes, or 'all'")
    parser.add_argument(
        "--exclude-storms-in-basins",
        default=None,
        help="Drop every storm with any fix in these basins (e.g. NI), not just those rows",
    )
    parser.add_argument("--use-vision", action="store_true", help="Enable the ResNet-18 branch")
    parser.add_argument("--no-pretrained-vision", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS))
    parser.add_argument("--init-from", default=None)
    parser.add_argument(
        "--allow-pretrain-overlap",
        action="store_true",
        help="Permit validation storms the pretrain already trained on (not out-of-sample)",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args(argv)

    result = train(
        TrainConfig(
            tracks=args.tracks,
            images_index=args.images_index,
            image_root=args.image_root,
            basins=_parse_basins(args.basins),
            exclude_storms_in_basins=_parse_basins(args.exclude_storms_in_basins),
            use_vision=args.use_vision,
            pretrained_vision=not args.no_pretrained_vision,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            val_fraction=args.val_fraction,
            seed=args.seed,
            checkpoint=args.checkpoint,
            metrics_path=args.metrics_path,
            init_from=args.init_from,
            allow_pretrain_overlap=args.allow_pretrain_overlap,
            max_windows=args.max_windows,
            patience=args.patience,
        )
    )
    print(json.dumps({"dataset": result["dataset"], "metrics": result["metrics"]}, indent=2))
    print("\nper-lead validation track error (km):")
    for lead, model_rmse, base_rmse in zip(
        TARGET_LEADS, result["validation"]["track_rmse_km"], result["baseline"]["track_rmse_km"]
    ):
        verdict = "model" if model_rmse < base_rmse else "baseline"
        print(f"  +{lead:3d} h   model {model_rmse:7.1f}   linear baseline {base_rmse:7.1f}   better: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
