from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data.build_dataset import category
from ml.model import CycloneImageClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "kaggle_temporal_test"
CHECKPOINT = ROOT / "models" / "multiyear45_resnet18_ir_weighted.pt"
TRAIN_INDEX = ROOT / "kaggle_multiyear45" / "images_index.csv"
OUT_JSON = ROOT / "models" / "temporal_test_metrics_weighted.json"
OUT_CSV = ROOT / "models" / "temporal_test_predictions_weighted.csv"

CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
NORM = T.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)


class TemporalDataset(Dataset):
    def __init__(self, root: Path, rows: pd.DataFrame):
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.tf = T.Compose([T.Resize(224), T.ToTensor(), NORM])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.root / row["image_path"]) as image:
            x = self.tf(image.convert("RGB"))
        return x


def score(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = pred - truth
    true_c = np.array([CATS.index(category(k)) for k in truth])
    pred_c = np.array([
        CATS.index(category(max(float(k), 0.0))) for k in pred
    ])
    return {
        "n": int(len(truth)),
        "mae_kt": round(float(np.abs(err).mean()), 3),
        "rmse_kt": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "category_accuracy": round(float(np.mean(true_c == pred_c)), 4),
        "within_one_category": round(
            float(np.mean(np.abs(true_c - pred_c) <= 1)), 4
        ),
    }


def main():
    for path in (DATA / "images_index.csv", CHECKPOINT, TRAIN_INDEX):
        if not path.exists():
            raise FileNotFoundError(path)

    index = pd.read_csv(DATA / "images_index.csv")
    index = index[index["channel"] == "ir"].copy()
    index["wind_kt"] = pd.to_numeric(index["wind_kt"], errors="coerce")
    index = index.dropna(subset=["wind_kt", "sid", "image_path"])

    if index.empty:
        raise RuntimeError("No valid infrared test images found.")

    years = pd.to_datetime(index["time"]).dt.year
    if not years.isin([2021, 2022, 2023]).all():
        raise RuntimeError("Unexpected year found in temporal test data.")

    train_index = pd.read_csv(TRAIN_INDEX)
    train = train_index[
        (train_index["channel"] == "ir")
        & (train_index["split"] == "train")
    ].copy()
    train["wind_kt"] = pd.to_numeric(train["wind_kt"], errors="coerce")
    train = train.dropna(subset=["wind_kt"])
    if train.empty:
        raise RuntimeError("Could not calculate training-mean baseline.")

    # Confirm temporal test storm IDs do not overlap the original training storms.
    train_sids = set(train["sid"].astype(str))
    test_sids = set(index["sid"].astype(str))
    overlap = train_sids & test_sids
    if overlap:
        raise RuntimeError(f"Storm leakage detected: {sorted(overlap)[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    if checkpoint.get("channel") != "ir":
        raise RuntimeError(
            f"Checkpoint channel is {checkpoint.get('channel')!r}, not 'ir'."
        )
    if checkpoint.get("target") != "wind_kt/100":
        raise RuntimeError(
            f"Unexpected checkpoint target: {checkpoint.get('target')!r}"
        )

    model = CycloneImageClassifier(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        TemporalDataset(DATA, index),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch.to(device)).squeeze(1) * 100.0
            predictions.extend(output.cpu().numpy().tolist())

    truth = index["wind_kt"].to_numpy(dtype=float)
    pred = np.asarray(predictions, dtype=float)
    baseline_mean = float(train["wind_kt"].mean())
    baseline = np.full(len(truth), baseline_mean)

    result = index[["sid", "name", "time", "image_path", "wind_kt"]].copy()
    result = result.rename(columns={"wind_kt": "actual_wind_kt"})
    result["predicted_wind_kt"] = np.round(pred, 3)
    result["baseline_wind_kt"] = round(baseline_mean, 3)
    result["actual_category"] = [
        category(float(k)) for k in truth
    ]
    result["predicted_category"] = [
        category(max(float(k), 0.0)) for k in pred
    ]
    result["year"] = pd.to_datetime(result["time"]).dt.year
    result.to_csv(OUT_CSV, index=False)

    report = {
        "experiment": "Temporal holdout: train years 2018-2020; test years 2021-2023",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "channel": "ir",
        "device": str(device),
        "test_images": int(len(index)),
        "test_storms": int(index["sid"].nunique()),
        "test_years": sorted(int(y) for y in years.unique()),
        "training_images_for_baseline": int(len(train)),
        "training_storms_for_baseline": int(train["sid"].nunique()),
        "training_mean_wind_kt": round(baseline_mean, 3),
        "test_model": score(pred, truth),
        "test_baseline_predict_train_mean": score(baseline, truth),
        "by_year": {},
        "notes": [
            "Checkpoint was loaded without retraining.",
            "Temporal test storms were checked for overlap with training storms.",
            "Test results must not be used to tune this checkpoint.",
            "Images from a storm are correlated; image-level metrics are not independent-sample estimates.",
        ],
    }

    for year in sorted(result["year"].unique()):
        mask = result["year"].to_numpy() == year
        report["by_year"][str(int(year))] = {
            "images": int(mask.sum()),
            "storms": int(result.loc[mask, "sid"].nunique()),
            "model": score(pred[mask], truth[mask]),
            "baseline": score(baseline[mask], truth[mask]),
        }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTEMPORAL HOLDOUT RESULTS")
    print(json.dumps(report, indent=2))
    print(f"\nMetrics saved to: {OUT_JSON}")
    print(f"Predictions saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()

