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
DATA = ROOT / "kaggle_multiyear45"
CHECKPOINT = ROOT / "models" / "multiyear45_resnet18_ir.pt"
OUT_CSV = ROOT / "models" / "training_fit_predictions.csv"
OUT_JSON = ROOT / "models" / "training_fit_metrics.json"

CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
NORM = T.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)


class ChipDataset(Dataset):
    def __init__(self, root, rows):
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.tf = T.Compose([T.Resize(224), T.ToTensor(), NORM])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.root / row["image_path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x


def category_score(frame):
    actual = frame["actual_wind_kt"].to_numpy(dtype=float)
    pred = frame["predicted_wind_kt"].to_numpy(dtype=float)
    err = pred - actual

    return {
        "images": int(len(frame)),
        "storms": int(frame["sid"].nunique()),
        "actual_mean_kt": round(float(actual.mean()), 3),
        "predicted_mean_kt": round(float(pred.mean()), 3),
        "mean_bias_kt": round(float(err.mean()), 3),
        "mae_kt": round(float(np.abs(err).mean()), 3),
        "rmse_kt": round(float(np.sqrt(np.mean(err ** 2))), 3),
    }


def main():
    for path in (
        DATA / "images_index.csv",
        CHECKPOINT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    index = pd.read_csv(DATA / "images_index.csv")
    index = index[
        (index["channel"] == "ir")
        & (index["split"].isin(["train", "val"]))
    ].copy()
    index["wind_kt"] = pd.to_numeric(index["wind_kt"], errors="coerce")
    index = index.dropna(subset=["wind_kt", "sid", "image_path"])

    if index.empty:
        raise RuntimeError("No training/validation IR images found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)

    if checkpoint.get("channel") != "ir":
        raise RuntimeError("Checkpoint is not an infrared model.")
    if checkpoint.get("target") != "wind_kt/100":
        raise RuntimeError("Unexpected checkpoint target scaling.")

    model = CycloneImageClassifier(num_classes=1, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outputs = []
    for split in ("train", "val"):
        rows = index[index["split"] == split].copy()
        if rows.empty:
            continue

        loader = DataLoader(
            ChipDataset(DATA, rows),
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )

        preds = []
        with torch.inference_mode():
            for batch in loader:
                values = model(batch.to(device)).squeeze(1) * 100.0
                preds.extend(values.cpu().numpy().tolist())

        rows["actual_wind_kt"] = rows["wind_kt"].astype(float)
        rows["predicted_wind_kt"] = preds
        rows["actual_category"] = rows["actual_wind_kt"].map(category)
        rows["predicted_category"] = rows["predicted_wind_kt"].map(
            lambda x: category(max(float(x), 0.0))
        )
        rows["year"] = pd.to_datetime(rows["time"]).dt.year
        outputs.append(rows)

    result = pd.concat(outputs, ignore_index=True)
    result["error_kt"] = (
        result["predicted_wind_kt"] - result["actual_wind_kt"]
    )
    result["abs_error_kt"] = result["error_kt"].abs()
    result.to_csv(OUT_CSV, index=False)

    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "warning": (
            "Training metrics are in-sample because these images were used "
            "to fit the checkpoint. Validation metrics are the more useful "
            "diagnostic of generalization within 2018-2020."
        ),
        "overall": {},
        "by_category": {},
    }

    for split in ("train", "val"):
        frame = result[result["split"] == split]
        if frame.empty:
            continue

        report["overall"][split] = category_score(frame)
        report["by_category"][split] = {}

        for cat in CATS:
            subset = frame[frame["actual_category"] == cat]
            if len(subset):
                report["by_category"][split][cat] = category_score(subset)

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nTRAINING-FIT DIAGNOSTIC")
    print(json.dumps(report, indent=2))
    print(f"\nPredictions saved to: {OUT_CSV}")
    print(f"Metrics saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
