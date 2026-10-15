"""Train and honestly evaluate a storm-intensity model on the dataset made by data/build_dataset.py.

    python -m ml.train_chips --data-dir kaggle_dataset --channel truecolor --epochs 12

Works unchanged in a Kaggle notebook: --data-dir /kaggle/input/<your-dataset>.

Task: image chip -> maximum sustained wind (kt). Splits come from images_index.csv
(by storm ID). The test split is evaluated once, with the best-validation checkpoint,
and compared with a "predict the training mean" baseline so you can see whether the
model learned anything. No horizontal flips: they mirror the storm's rotation sense.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from data.build_dataset import category
from ml.model import CycloneImageClassifier

CATS = ["TD", "TS", "C1", "C2", "C3", "C4", "C5"]
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ChipDataset(Dataset):
    def __init__(self, root: Path, rows: pd.DataFrame, size: int, train: bool):
        self.root, self.rows = root, rows.reset_index(drop=True)
        crop = [T.RandomResizedCrop(size, scale=(0.8, 1.0)), T.RandomRotation(30)] if train else [T.Resize(size)]
        self.tf = T.Compose([*crop, T.ToTensor(), NORM])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        with Image.open(self.root / r["image_path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor(float(r["wind_kt"]), dtype=torch.float32)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, truth = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).squeeze(1).cpu() * 100)  # model works in units of 100 kt
            truth.append(y)
    return torch.cat(preds).numpy(), torch.cat(truth).numpy()


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    true_c = np.array([CATS.index(category(k)) for k in truth])
    pred_c = np.array([CATS.index(category(max(k, 0))) for k in pred])
    per_cat = {c: round(float(np.abs(err[true_c == i]).mean()), 2) for i, c in enumerate(CATS) if (true_c == i).any()}
    return {
        "n": int(len(truth)),
        "mae_kt": round(float(np.abs(err).mean()), 2),
        "rmse_kt": round(float(np.sqrt((err**2).mean())), 2),
        "category_accuracy": round(float((true_c == pred_c).mean()), 3),
        "within_one_category": round(float((np.abs(true_c - pred_c) <= 1).mean()), 3),
        "mae_kt_by_true_category": per_cat,
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="kaggle_dataset")
    p.add_argument("--channel", default="truecolor", choices=["truecolor", "ir"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline / tests)")
    p.add_argument("--output", default="models/intensity_resnet18.pt")
    p.add_argument("--metrics", default="metrics.json")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    root = Path(args.data_dir)
    index = pd.read_csv(root / "images_index.csv")
    index = index[index["channel"] == args.channel]
    parts = {s: index[index["split"] == s] for s in ("train", "val", "test")}
    for s, df in parts.items():
        if df.empty:
            raise SystemExit(f"No '{s}' images for channel '{args.channel}'. Fetch more data or pick another channel.")
    seen = [set(df["sid"]) for df in parts.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]), "storm leaked across splits"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        s: DataLoader(ChipDataset(root, df, args.size, train=(s == "train")), batch_size=args.batch_size,
                      shuffle=(s == "train"), num_workers=args.workers)
        for s, df in parts.items()
    }
    model = CycloneImageClassifier(num_classes=1, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss(delta=0.2)  # on wind / 100 kt

    best, out = float("inf"), Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} channel={args.channel} " + " ".join(f"{s}={len(df)} imgs/{df['sid'].nunique()} storms" for s, df in parts.items()))
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            x, y = x.to(device), (y / 100).to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x).squeeze(1), y).backward()
            opt.step()
        sched.step()
        val = score(*predict(model, loaders["val"], device))
        print(f"epoch {epoch:02d}/{args.epochs} val MAE {val['mae_kt']:.2f} kt, category acc {val['category_accuracy']:.3f}")
        if val["mae_kt"] < best:
            best = val["mae_kt"]
            torch.save({"model_state_dict": model.state_dict(), "architecture": "resnet18", "target": "wind_kt/100",
                        "channel": args.channel, "val": val}, out)

    model.load_state_dict(torch.load(out, map_location=device)["model_state_dict"])
    pred, truth = predict(model, loaders["test"], device)
    mean_wind = float(parts["train"]["wind_kt"].mean())
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "storms": {s: int(df["sid"].nunique()) for s, df in parts.items()},
        "images": {s: int(len(df)) for s, df in parts.items()},
        "test_model": score(pred, truth),
        "test_baseline_predict_train_mean": score(np.full_like(truth, mean_wind), truth),
    }
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("test_model", "test_baseline_predict_train_mean")}, indent=2))
    return metrics


if __name__ == "__main__":
    main()
