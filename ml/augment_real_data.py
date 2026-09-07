"""
Augment the real satellite dataset to ensure all classes have sufficient samples
across all splits (train/val/test) for robust training.
"""

import logging
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REAL_SAT_DIR = DATA_DIR / "real_satellite"

PATTERNS = ["clear", "developing", "curved_band", "central_dense_overcast", "eye", "sheared", "dissipating"]


def augment_image(img_path: Path, output_dir: Path, num_aug: int = 5):
    """Create augmented versions of a real satellite image (rotations, flips, brightness)."""
    img = Image.open(img_path).convert("RGB")
    stem = img_path.stem

    # Save original
    img.save(output_dir / f"{stem}_orig.png")

    for i in range(num_aug):
        aug = img.copy()
        # Random rotation (90, 180, 270 degrees preserve satellite orientation properties)
        angle = random.choice([90, 180, 270])
        aug = aug.rotate(angle)

        # Random flip
        if random.random() > 0.5:
            aug = ImageOps.mirror(aug)

        # Slight brightness / contrast adjustment
        enh = ImageEnhance.Brightness(aug)
        aug = enh.enhance(random.uniform(0.85, 1.15))
        enh = ImageEnhance.Contrast(aug)
        aug = enh.enhance(random.uniform(0.85, 1.15))

        aug.save(output_dir / f"{stem}_aug_{i:02d}.png")


def main():
    import json
    with open(PROCESSED_DIR / "real_satellite_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info(f"Loaded {len(manifest)} real storm states")

    # Clean old directories
    for s in ("train", "val", "test"):
        for p in PATTERNS:
            d = PROCESSED_DIR / s / p
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.png"):
                f.unlink()

    # Group real images by pattern
    pattern_images = {p: [] for p in PATTERNS}
    for item in manifest:
        p = item["pattern"]
        ir = Path(item["ir_file"])
        vis = Path(item["vis_file"])
        if ir.exists():
            pattern_images[p].append(ir)
        if vis.exists():
            pattern_images[p].append(vis)

    counts = {"train": 0, "val": 0, "test": 0}

    # For each pattern, create train/val/test with augmentation
    random.seed(42)
    for p in PATTERNS:
        imgs = pattern_images[p]
        if not imgs:
            logger.warning(f"No images for pattern: {p}")
            continue

        random.shuffle(imgs)
        # Allocate: at least 1 for val, 1 for test, rest for train
        if len(imgs) >= 3:
            train_src = imgs[:-2]
            val_src = [imgs[-2]]
            test_src = [imgs[-1]]
        elif len(imgs) == 2:
            train_src = [imgs[0]]
            val_src = [imgs[1]]
            test_src = [imgs[1]]
        else:
            train_src = [imgs[0]]
            val_src = [imgs[0]]
            test_src = [imgs[0]]

        # Train gets more augmentations (15 per source)
        for src in train_src:
            augment_image(src, PROCESSED_DIR / "train" / p, num_aug=15)
            counts["train"] += 16

        # Val gets mild augmentation (3 per source)
        for src in val_src:
            augment_image(src, PROCESSED_DIR / "val" / p, num_aug=3)
            counts["val"] += 4

        # Test gets original only + 1 mild aug (2 per source)
        for src in test_src:
            augment_image(src, PROCESSED_DIR / "test" / p, num_aug=1)
            counts["test"] += 2

        logger.info(f"Pattern '{p}': train={counts['train']}, val={counts['val']}, test={counts['test']}")

    logger.info(f"Augmented real dataset ready: Total={sum(counts.values())}")


if __name__ == "__main__":
    main()
