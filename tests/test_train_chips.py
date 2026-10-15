"""Smoke test: ml/train_chips.py runs end to end and writes honest metrics.
The images are synthetic fixtures used only to exercise the code, not data."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from data.build_dataset import split_of  # noqa: E402
from ml import train_chips  # noqa: E402


def make_dataset(root, n_storms=40, per_storm=3):
    rng = np.random.default_rng(0)
    rows = []
    (root / "images").mkdir(parents=True)
    for s in range(n_storms):
        sid = f"S{s:03d}"
        for k in range(per_storm):
            wind = float(rng.integers(20, 140))
            rel = f"images/{sid}_{k}.jpg"
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(root / rel)
            rows.append({"image_path": rel, "channel": "truecolor", "sid": sid, "wind_kt": wind, "split": split_of(sid)})
    pd.DataFrame(rows).to_csv(root / "images_index.csv", index=False)


def test_train_runs_and_reports_baseline(tmp_path):
    make_dataset(tmp_path / "ds")
    metrics = train_chips.main([
        "--data-dir", str(tmp_path / "ds"), "--epochs", "1", "--batch-size", "16", "--size", "64",
        "--workers", "0", "--no-pretrained", "--output", str(tmp_path / "m.pt"), "--metrics", str(tmp_path / "metrics.json"),
    ])
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["test_model"]["n"] == saved["images"]["test"]
    assert {"mae_kt", "category_accuracy", "mae_kt_by_true_category"} <= set(saved["test_model"])
    assert saved["test_baseline_predict_train_mean"]["mae_kt"] > 0
    assert (tmp_path / "m.pt").exists() and metrics["storms"]["train"] > metrics["storms"]["test"]
