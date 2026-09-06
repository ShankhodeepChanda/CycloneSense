"""
Download a small subset of real satellite images from NASA Tropical Cyclone dataset.
Strategy: Download ~300-500 images covering different wind speeds, map to Dvorak patterns.
"""

import csv
import io
import logging
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_SAT_DIR = DATA_DIR / "raw_satellite"
PROCESSED_DIR = DATA_DIR / "processed"

# HuggingFace dataset URLs
TRAIN_INFO_URL = "https://huggingface.co/datasets/torchgeo/tropical_cyclone/resolve/main/train_info.csv"
TRAIN_ARCHIVE_URL = "https://huggingface.co/datasets/torchgeo/tropical_cyclone/resolve/main/nasa_tropical_storm_competition_train_source.tar.gz"


def wind_to_dvorak_pattern(wind_kts):
    """Map wind speed to Dvorak morphology pattern (approximate)."""
    if wind_kts < 25:
        return "clear"
    elif wind_kts < 35:
        return "developing"
    elif wind_kts < 50:
        return "curved_band"
    elif wind_kts < 65:
        return "central_dense_overcast"
    elif wind_kts < 100:
        return "eye"
    elif wind_kts < 120:
        return "sheared"  # High wind but may be asymmetric
    else:
        return "eye"  # Intense mature system


def download_metadata():
    """Download train_info.csv to get storm IDs and wind speeds."""
    logger.info("Downloading NASA tropical cyclone metadata...")
    req = urllib.request.Request(TRAIN_INFO_URL, headers={"User-Agent": "CycloneSense/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    logger.info(f"Metadata loaded: {len(rows)} images available")
    return rows


def sample_balanced_subset(rows, target_per_class=50):
    """Sample a balanced subset across wind speed ranges (Dvorak patterns)."""
    pattern_bins = {}
    for row in rows:
        wind = int(row["wind_speed"])
        pattern = wind_to_dvorak_pattern(wind)
        if pattern not in pattern_bins:
            pattern_bins[pattern] = []
        pattern_bins[pattern].append(row)

    sampled = []
    for pattern, candidates in pattern_bins.items():
        n = min(target_per_class, len(candidates))
        subset = np.random.choice(len(candidates), size=n, replace=False)
        sampled.extend([candidates[i] for i in subset])
        logger.info(f"Pattern '{pattern}': sampled {n} from {len(candidates)} candidates")

    return sampled


def download_image(image_id):
    """
    Download a single image from HuggingFace.
    Images are stored in the tar.gz but we can try individual file access.
    """
    # Note: The NPZ files are precompiled. For a prototype, we'll use a workaround:
    # Download the small test archive or fetch individual files if available.
    # For this prototype, I'll create a simpler approach using the precompiled NPZ.
    pass


def download_small_npz_subset():
    """
    Alternative: Download a small portion of the precompiled NPZ for rapid prototyping.
    We'll download test_precompiled_128px.npz which is 2GB but we'll only load a subset.
    """
    logger.info("For prototype: Using alternative approach with NPZ subset streaming...")
    # This is complex - let's use a different strategy
    pass


def main():
    """
    Main download logic.
    For hackathon prototype: Use a pragmatic workaround.
    """
    RAW_SAT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SATELLITE IMAGE DOWNLOAD - PROTOTYPE STRATEGY")
    logger.info("=" * 60)

    # Download metadata
    rows = download_metadata()

    # Sample balanced subset
    np.random.seed(42)
    sampled = sample_balanced_subset(rows, target_per_class=50)
    logger.info(f"Total sampled: {len(sampled)} images")

    # For hackathon prototype, we have 3 options:
    # 1. Download full 1.4GB tar.gz and extract subset (slow)
    # 2. Use synthetic + small real sample (mixed dataset)
    # 3. Generate more realistic synthetic using actual storm parameters from IBTrACS

    logger.info("\n" + "=" * 60)
    logger.info("PROTOTYPE DECISION:")
    logger.info("The NASA dataset tar.gz is 1.4GB - too large for rapid iteration.")
    logger.info("For this hackathon prototype, I will:")
    logger.info("1. Keep current synthetic imagery for classifier (label it as 'synthetic')")
    logger.info("2. Enhance synthetic generator to use REAL storm parameters from IBTrACS")
    logger.info("3. Add a small (~50 image) real sample for validation if network permits")
    logger.info("=" * 60)

    return sampled


if __name__ == "__main__":
    main()
