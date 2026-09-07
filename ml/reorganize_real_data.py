"""
Reorganize real satellite images with stratified splitting to ensure
every pattern class has samples in train/val/test.
"""

import json
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_FILE = PROCESSED_DIR / "real_satellite_manifest.json"

PATTERNS = ["clear", "developing", "curved_band", "central_dense_overcast", "eye", "sheared", "dissipating"]


def main():
    """Reorganize with stratified split per pattern class."""
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded manifest with {len(manifest)} samples")

    # Group by pattern
    pattern_samples = {}
    for item in manifest:
        pattern = item["pattern"]
        if pattern not in pattern_samples:
            pattern_samples[pattern] = []
        # Each item has IR and VIS
        pattern_samples[pattern].append(item)

    logger.info("Samples per pattern:")
    for p, items in pattern_samples.items():
        logger.info(f"  {p}: {len(items)} samples")

    # Clean old files
    for s in ("train", "val", "test"):
        for pattern in PATTERNS:
            p = PROCESSED_DIR / s / pattern
            p.mkdir(parents=True, exist_ok=True)
            for old_f in p.glob("*.png"):
                old_f.unlink()

    counts = {"train": 0, "val": 0, "test": 0}

    # Stratified split per pattern
    random.seed(42)
    for pattern, items in pattern_samples.items():
        random.shuffle(items)
        n = len(items)

        # Ensure at least 1 per split if possible
        if n >= 3:
            n_train = max(1, int(n * 0.70))
            n_val = max(1, int(n * 0.15))
            train_items = items[:n_train]
            val_items = items[n_train:n_train + n_val]
            test_items = items[n_train + n_val:]
        elif n == 2:
            # Put 1 in train, 1 in test
            train_items = [items[0]]
            val_items = []
            test_items = [items[1]]
        elif n == 1:
            # Put in train
            train_items = [items[0]]
            val_items = []
            test_items = []
        else:
            train_items = []
            val_items = []
            test_items = []

        # Copy IR and VIS for each sample
        for item in train_items:
            src_ir = Path(item["ir_file"])
            src_vis = Path(item["vis_file"])
            if src_ir.exists():
                shutil.copy(src_ir, PROCESSED_DIR / "train" / pattern / src_ir.name)
                counts["train"] += 1
            if src_vis.exists():
                shutil.copy(src_vis, PROCESSED_DIR / "train" / pattern / src_vis.name)
                counts["train"] += 1

        for item in val_items:
            src_ir = Path(item["ir_file"])
            src_vis = Path(item["vis_file"])
            if src_ir.exists():
                shutil.copy(src_ir, PROCESSED_DIR / "val" / pattern / src_ir.name)
                counts["val"] += 1
            if src_vis.exists():
                shutil.copy(src_vis, PROCESSED_DIR / "val" / pattern / src_vis.name)
                counts["val"] += 1

        for item in test_items:
            src_ir = Path(item["ir_file"])
            src_vis = Path(item["vis_file"])
            if src_ir.exists():
                shutil.copy(src_ir, PROCESSED_DIR / "test" / pattern / src_ir.name)
                counts["test"] += 1
            if src_vis.exists():
                shutil.copy(src_vis, PROCESSED_DIR / "test" / pattern / src_vis.name)
                counts["test"] += 1

        logger.info(f"  {pattern}: train={len(train_items)*2}, val={len(val_items)*2}, test={len(test_items)*2}")

    logger.info(f"Stratified split complete: train={counts['train']}, val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
