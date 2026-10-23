from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.build_dataset import category
from ml.model import CycloneImageClassifier

MODELS = {
    "A": ROOT / "models" / "multiyear45_resnet18_ir_weighted.pt",
    "C": ROOT / "models" / "multiyear45_resnet18_ir_weighted_lr1e4.pt",
}
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRANSFORM = T.Compose([T.Resize(224), T.ToTensor(), NORM])


class ChipRows(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame):
        self.root = root
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> torch.Tensor:
        rel = self.frame.iloc[i]["image_path"]
        with Image.open(self.root / rel) as image:
            return TRANSFORM(image.convert("RGB"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(dataset: str, wanted_split: str) -> tuple[Path, pd.DataFrame]:
    root = ROOT / dataset
    frame = pd.read_csv(root / "images_index.csv")
    frame = frame[(frame["channel"] == "ir") & (frame["split"] == wanted_split)].copy()
    frame["wind_kt"] = pd.to_numeric(frame["wind_kt"], errors="coerce")
    frame = frame.dropna(subset=["sid", "time", "image_path", "wind_kt"])
    if frame.empty:
        raise RuntimeError(f"No rows for {dataset}/{wanted_split}")
    if frame["image_path"].duplicated().any():
        raise RuntimeError(f"Duplicate image_path in {dataset}/{wanted_split}")
    missing = [rel for rel in frame["image_path"] if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} indexed images; first: {missing[0]}")
    frame["actual_wind_kt"] = frame["wind_kt"].astype(float)
    frame["actual_category"] = frame["actual_wind_kt"].map(category)
    return root, frame.reset_index(drop=True)


def reproduce(model_id: str, checkpoint_path: Path, root: Path, frame: pd.DataFrame, partition: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("channel") != "ir" or checkpoint.get("target") != "wind_kt/100":
        raise RuntimeError(f"Unexpected checkpoint metadata for Model {model_id}: {checkpoint.keys()}")
    if not all(bool(torch.isfinite(value).all()) for value in checkpoint["model_state_dict"].values()):
        raise RuntimeError(f"Non-finite parameter in Model {model_id}")
    model = CycloneImageClassifier(num_classes=1, pretrained=False).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loader = DataLoader(ChipRows(root, frame), batch_size=32, shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for batch in loader:
            batches.append(model(batch).squeeze(1).cpu().numpy() * 100.0)
    pred = np.concatenate(batches).astype(float)
    if len(pred) != len(frame):
        raise RuntimeError(f"Prediction count mismatch for Model {model_id}/{partition}")
    out = frame[["image_path", "sid", "name", "time", "actual_wind_kt", "actual_category", "split"]].copy()
    out = out.rename(columns={"split": "dataset_split"})
    out["model"] = model_id
    out["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
    out["partition"] = partition
    out["prediction_source"] = "reproduced_checkpoint_inference"
    out["predicted_wind_kt"] = pred  # raw linear-head output; intentionally unclipped
    out["is_negative"] = pred < 0.0
    out["is_nonfinite"] = ~np.isfinite(pred)
    return out


def add_saved_original_test(path: Path) -> pd.DataFrame:
    saved = pd.read_csv(path)
    rows = []
    for model_id, pred_col in [("A", "predicted_A_kt"), ("C", "predicted_C_kt")]:
        part = saved[["image_path", "sid", "name", "time", "actual_wind_kt", "actual_category", pred_col]].copy()
        part = part.rename(columns={pred_col: "predicted_wind_kt"})
        part["model"] = model_id
        part["checkpoint"] = str(MODELS[model_id].relative_to(ROOT))
        part["dataset_split"] = "test"
        part["partition"] = "2018_2020_test_previously_saved"
        part["prediction_source"] = str(path.relative_to(ROOT))
        part["is_negative"] = pd.to_numeric(part["predicted_wind_kt"], errors="coerce") < 0.0
        part["is_nonfinite"] = ~np.isfinite(pd.to_numeric(part["predicted_wind_kt"], errors="coerce"))
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def compare_saved(reproduced: pd.DataFrame, saved_path: Path, split_filter: set[str], partition: str) -> dict:
    current = reproduced[(reproduced.model == "A") & reproduced.dataset_split.isin(split_filter)]
    saved = pd.read_csv(saved_path)
    merged = current[["image_path", "predicted_wind_kt"]].merge(
        saved[["image_path", "predicted_wind_kt"]], on="image_path", suffixes=("_reproduced", "_saved"), validate="one_to_one"
    )
    delta = np.abs(merged["predicted_wind_kt_reproduced"] - merged["predicted_wind_kt_saved"])
    rounded_limit = 0.00051 if "temporal" in partition else 1e-5
    return {
        "partition": partition,
        "rows_compared": int(len(merged)),
        "missing_or_extra_rows": int(abs(len(current) - len(saved))),
        "max_abs_difference_kt": float(delta.max()) if len(delta) else None,
        "comparison_tolerance_kt": rounded_limit,
        "within_tolerance": bool(len(merged) == len(saved) and (delta <= rounded_limit).all()),
        "saved_file": str(saved_path.relative_to(ROOT)),
    }


def main() -> None:
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    reproduced = []
    run_specs = [
        ("kaggle_multiyear45", "train", "2018_2020_train_in_sample"),
        ("kaggle_multiyear45", "val", "2018_2020_val_storm_heldout"),
        ("kaggle_temporal_test", "temporal_test", "2021_2023_temporal_holdout"),
    ]
    for dataset, split, partition in run_specs:
        root, frame = load_rows(dataset, split)
        for model_id, checkpoint_path in MODELS.items():
            print(f"Reproducing Model {model_id}: {partition} ({len(frame)} images)", flush=True)
            reproduced.append(reproduce(model_id, checkpoint_path, root, frame, partition))
    reproduced_df = pd.concat(reproduced, ignore_index=True)

    saved_val = ROOT / "models" / "weighted_validation_predictions.csv"
    saved_temporal = ROOT / "models" / "temporal_test_predictions_weighted.csv"
    validation_crosscheck = compare_saved(
        reproduced_df, saved_val, {"train", "val"}, "2018_2020_train_and_validation"
    )
    temporal_crosscheck = compare_saved(
        reproduced_df, saved_temporal, {"temporal_test"}, "2021_2023_temporal"
    )
    if not validation_crosscheck["within_tolerance"]:
        raise RuntimeError(f"Model A train/validation predictions did not reproduce: {validation_crosscheck}")
    if not temporal_crosscheck["within_tolerance"]:
        raise RuntimeError(f"Model A temporal predictions did not reproduce: {temporal_crosscheck}")

    saved_test_path = ROOT / "models" / "storm_diagnostics_lr_comparison" / "aligned_test_predictions.csv"
    saved_test = add_saved_original_test(saved_test_path)
    columns = [
        "model", "checkpoint", "partition", "dataset_split", "prediction_source", "image_path", "sid", "name", "time",
        "actual_wind_kt", "actual_category", "predicted_wind_kt", "is_negative", "is_nonfinite",
    ]
    full = pd.concat([reproduced_df, saved_test], ignore_index=True)[columns].sort_values(
        ["model", "partition", "sid", "time", "image_path"]
    ).reset_index(drop=True)
    if int(full["is_nonfinite"].sum()) != 0:
        raise RuntimeError("Non-finite raw predictions detected")

    raw_path = OUT / "raw_prediction_reproduction.csv"
    negative_path = OUT / "negative_prediction_diagnostics.csv"
    full.to_csv(raw_path, index=False)
    negatives = full[full["is_negative"]].copy()
    negatives.to_csv(negative_path, index=False)

    summary_rows = []
    for (model_id, partition), group in full.groupby(["model", "partition"], sort=True):
        summary_rows.append({
            "model": model_id,
            "partition": partition,
            "images": int(len(group)),
            "storms": int(group["sid"].nunique()),
            "negative_predictions": int(group["is_negative"].sum()),
            "nonfinite_predictions": int(group["is_nonfinite"].sum()),
            "minimum_raw_prediction_kt": float(group["predicted_wind_kt"].min()),
            "maximum_raw_prediction_kt": float(group["predicted_wind_kt"].max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "negative_prediction_counts.csv", index=False)

    checkpoint_info = {}
    for model_id, path in MODELS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_info[model_id] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "architecture": checkpoint.get("architecture"),
            "channel": checkpoint.get("channel"),
            "target": checkpoint.get("target"),
            "fc_weight_shape": list(checkpoint["model_state_dict"]["backbone.fc.weight"].shape),
            "fc_bias": checkpoint["model_state_dict"]["backbone.fc.bias"].tolist(),
            "all_parameters_finite": all(bool(torch.isfinite(v).all()) for v in checkpoint["model_state_dict"].values()),
        }
    validation_metrics = json.loads((ROOT / "models" / "multiyear45_metrics_ir_weighted.json").read_text(encoding="utf-8"))
    c_metrics = json.loads((ROOT / "models" / "multiyear45_metrics_ir_weighted_lr1e4.json").read_text(encoding="utf-8"))
    for info, metric in [(checkpoint_info["A"], validation_metrics), (checkpoint_info["C"], c_metrics)]:
        info["saved_validation_metrics_all_numeric_finite"] = all(
            math.isfinite(float(v)) for v in pd.json_normalize(metric).select_dtypes(include="number").to_numpy().ravel()
        )
        info["training_loss_was_logged"] = "training_loss" in metric or "loss_history" in metric

    report = {
        "scope": "Raw inference reproduction only; no hyperparameter selection or training. The original 2018-2020 test split was read only from its already-saved A/C prediction CSV.",
        "prediction_count": int(len(full)),
        "negative_prediction_count": int(len(negatives)),
        "nonfinite_prediction_count": int(full["is_nonfinite"].sum()),
        "models": checkpoint_info,
        "model_a_reproduction_crosschecks": [validation_crosscheck, temporal_crosscheck],
        "negative_counts_by_model_and_partition": summary.to_dict(orient="records"),
        "negative_storms_by_model_and_partition": {
            f"{model_id}/{partition}": sorted(group["name"].dropna().astype(str).unique().tolist())
            for (model_id, partition), group in negatives.groupby(["model", "partition"], sort=True)
        },
        "interpretation": [
            "Raw head values are multiplied by 100 to convert the trained wind_kt/100 target back to knots.",
            "Raw negative regression values are preserved; no clipping was applied to prediction CSVs or any scientific metric.",
            "The train partition is in-sample. Validation and temporal predictions are diagnostic only and were not used for tuning.",
            "The saved original test predictions are included only as previously recorded negative-output diagnostics; no test score was recomputed here.",
        ],
    }
    (OUT / "negative_prediction_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"prediction_count": len(full), "negative_counts": summary.to_dict(orient="records"), "crosschecks": report["model_a_reproduction_crosschecks"]}, indent=2))
    print(f"Saved raw predictions: {raw_path}")
    print(f"Saved negative rows: {negative_path}")


if __name__ == "__main__":
    main()
