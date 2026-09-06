"""
Download real NASA MODIS/VIIRS satellite imagery for historical North Indian Ocean cyclones.
Uses NASA GIBS (Global Imagery Browse Services) open WMS API.
Pulls real IR (thermal brightness temperature) and True Color imagery centered on IBTrACS cyclone positions.
"""

import csv
import logging
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRACKS_FILE = DATA_DIR / "processed" / "cyclone_tracks.csv"
REAL_SAT_DIR = DATA_DIR / "real_satellite"

# Notable North Indian Ocean Cyclones with verified dates and peak locations
NOTABLE_STORMS = [
    # (storm_name, date_str, lat, lon, wind_kts, pattern_label, intensity_class)
    ("AMPHAN", "2020-05-18", 13.5, 86.5, 140, "eye", "very_severe_cyclonic_storm"),
    ("AMPHAN_EARLY", "2020-05-16", 10.8, 86.3, 45, "curved_band", "tropical_storm"),
    ("AMPHAN_MID", "2020-05-17", 11.5, 86.1, 75, "central_dense_overcast", "severe_cyclonic_storm"),
    ("AMPHAN_LANDFALL", "2020-05-20", 21.6, 88.3, 85, "sheared", "severe_cyclonic_storm"),
    ("FANI", "2019-05-02", 16.0, 84.8, 115, "eye", "very_severe_cyclonic_storm"),
    ("FANI_EARLY", "2019-04-28", 5.2, 88.5, 40, "developing", "tropical_storm"),
    ("FANI_MID", "2019-04-30", 9.5, 86.9, 65, "central_dense_overcast", "severe_cyclonic_storm"),
    ("BIPARJOY", "2023-06-11", 18.6, 67.8, 90, "eye", "very_severe_cyclonic_storm"),
    ("BIPARJOY_EARLY", "2023-06-07", 12.7, 66.2, 45, "curved_band", "tropical_storm"),
    ("BIPARJOY_MID", "2023-06-09", 14.8, 66.4, 75, "central_dense_overcast", "severe_cyclonic_storm"),
    ("BIPARJOY_LATE", "2023-06-15", 22.8, 68.3, 60, "sheared", "tropical_storm"),
    ("TAUKTAE", "2021-05-17", 18.5, 71.5, 100, "eye", "very_severe_cyclonic_storm"),
    ("TAUKTAE_EARLY", "2021-05-15", 12.2, 72.6, 45, "curved_band", "tropical_storm"),
    ("TAUKTAE_MID", "2021-05-16", 15.0, 72.7, 70, "central_dense_overcast", "severe_cyclonic_storm"),
    ("MOCHA", "2023-05-13", 15.4, 89.1, 130, "eye", "very_severe_cyclonic_storm"),
    ("MOCHA_EARLY", "2023-05-11", 11.2, 88.1, 40, "developing", "tropical_storm"),
    ("MOCHA_MID", "2023-05-12", 13.0, 88.5, 65, "curved_band", "severe_cyclonic_storm"),
    ("REMAL", "2024-05-26", 21.0, 89.2, 60, "central_dense_overcast", "tropical_storm"),
    ("REMAL_EARLY", "2024-05-24", 15.0, 88.5, 35, "developing", "tropical_storm"),
    ("DANA", "2024-10-24", 20.0, 87.5, 65, "curved_band", "severe_cyclonic_storm"),
    ("DANA_EARLY", "2024-10-23", 16.5, 89.5, 35, "developing", "tropical_storm"),
    ("YAAS", "2021-05-25", 18.5, 88.5, 75, "central_dense_overcast", "severe_cyclonic_storm"),
    ("YAAS_EARLY", "2021-05-24", 16.0, 89.5, 40, "developing", "tropical_storm"),
    ("NISARGA", "2020-06-03", 18.0, 72.8, 60, "curved_band", "tropical_storm"),
    ("GULAB", "2021-09-26", 18.3, 85.0, 45, "curved_band", "tropical_storm"),
    ("JAWAD", "2021-12-04", 16.0, 85.0, 40, "developing", "tropical_storm"),
    ("ASANI", "2022-05-10", 15.0, 83.0, 55, "curved_band", "tropical_storm"),
    ("SITRANG", "2022-10-24", 20.0, 90.0, 45, "developing", "tropical_storm"),
    ("MANDOUS", "2022-12-09", 12.0, 81.0, 50, "curved_band", "tropical_storm"),
    ("MICHAUNG", "2023-12-04", 14.5, 80.5, 55, "curved_band", "tropical_storm"),
    ("HAMOON", "2023-10-24", 20.5, 91.0, 65, "central_dense_overcast", "severe_cyclonic_storm"),
    ("MIDHILI", "2023-11-17", 21.0, 90.5, 40, "developing", "tropical_storm"),
    # Non-cyclone / clear weather samples for baseline
    ("CLEAR_ARABIAN_SEA_1", "2023-01-15", 15.0, 65.0, 10, "clear", "depression"),
    ("CLEAR_BAY_OF_BENGAL_1", "2023-02-20", 12.0, 85.0, 10, "clear", "depression"),
    ("CLEAR_INDIAN_OCEAN_1", "2023-03-10", 5.0, 75.0, 10, "clear", "depression"),
    ("CLEAR_ARABIAN_SEA_2", "2024-01-20", 18.0, 68.0, 10, "clear", "depression"),
    ("CLEAR_BAY_OF_BENGAL_2", "2024-02-15", 16.0, 88.0, 10, "clear", "depression"),
    # Dissipating storms
    ("AMPHAN_DISSIPATING", "2020-05-21", 25.0, 89.5, 25, "dissipating", "depression"),
    ("FANI_DISSIPATING", "2019-05-04", 24.5, 89.0, 25, "dissipating", "depression"),
    ("BIPARJOY_DISSIPATING", "2023-06-17", 26.0, 72.0, 20, "dissipating", "depression"),
    ("TAUKTAE_DISSIPATING", "2021-05-19", 25.0, 74.0, 20, "dissipating", "depression"),
]


def fetch_gibs_image(layer: str, date_str: str, lat: float, lon: float, delta_deg: float = 4.0, size: int = 128) -> bytes:
    """Fetch a satellite tile centered on (lat, lon) with extent ±delta_deg."""
    min_lat = max(-90.0, lat - delta_deg)
    max_lat = min(90.0, lat + delta_deg)
    min_lon = max(-180.0, lon - delta_deg)
    max_lon = min(180.0, lon + delta_deg)

    url = (
        f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?"
        f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&LAYERS={layer}&"
        f"STYLES=&FORMAT=image/png&TRANSPARENT=true&HEIGHT={size}&WIDTH={size}&"
        f"TIME={date_str}&CRS=EPSG:4326&BBOX={min_lat},{min_lon},{max_lat},{max_lon}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "CycloneSense-AI/1.0"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read()


def download_all_real_samples():
    """Download real satellite imagery for all defined storms."""
    REAL_SAT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading real NASA satellite imagery for {len(NOTABLE_STORMS)} storm states...")

    manifest = []
    success_count = 0

    for item in NOTABLE_STORMS:
        name, date_str, lat, lon, wind_kts, pattern, intensity = item
        # Download both TrueColor and Thermal IR
        try:
            # 1. Thermal IR Band 31 (Brightness Temp)
            ir_data = fetch_gibs_image(
                "MODIS_Terra_Brightness_Temp_Band31_Day",
                date_str, lat, lon, delta_deg=4.0, size=128
            )
            ir_filename = f"{name}_MODIS_IR_{date_str}.png"
            ir_path = REAL_SAT_DIR / ir_filename
            with open(ir_path, "wb") as f:
                f.write(ir_data)

            # 2. True Color (Visible)
            vis_data = fetch_gibs_image(
                "MODIS_Terra_CorrectedReflectance_TrueColor",
                date_str, lat, lon, delta_deg=4.0, size=128
            )
            vis_filename = f"{name}_MODIS_VIS_{date_str}.png"
            vis_path = REAL_SAT_DIR / vis_filename
            with open(vis_path, "wb") as f:
                f.write(vis_data)

            manifest.append({
                "storm_name": name,
                "date": date_str,
                "lat": lat,
                "lon": lon,
                "wind_kts": wind_kts,
                "pattern": pattern,
                "intensity_class": intensity,
                "ir_file": str(ir_path),
                "vis_file": str(vis_path),
            })
            success_count += 1
            logger.info(f"[{success_count:02d}/{len(NOTABLE_STORMS)}] Downloaded {name} ({date_str}) -> IR & VIS")
            time.sleep(0.3)  # Gentle rate limiting
        except Exception as exc:
            logger.warning(f"Failed to download {name} ({date_str}): {exc}")

    # Save manifest
    manifest_path = DATA_DIR / "processed" / "real_satellite_manifest.json"
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Successfully downloaded {success_count} real satellite pairs. Manifest saved to {manifest_path}")
    return manifest


def prepare_real_image_dataset(manifest):
    """
    Organize downloaded real satellite images into train/val/test splits strictly by storm ID.
    Replaces the synthetic dataset.
    """
    PROCESSED_DIR = DATA_DIR / "processed"

    # Separate storm bases for strict storm-level splitting
    storm_bases = {}
    for item in manifest:
        base = item["storm_name"].split("_")[0]
        if base not in storm_bases:
            storm_bases[base] = []
        storm_bases[base].append(item)

    import random
    random.seed(42)
    bases = list(storm_bases.keys())
    random.shuffle(bases)

    n = len(bases)
    n_train = max(1, int(n * 0.70))
    n_val = max(1, int(n * 0.15))

    train_bases = set(bases[:n_train])
    val_bases = set(bases[n_train:n_train + n_val])
    test_bases = set(bases[n_train + n_val:])

    logger.info(f"Real data storm split: {len(train_bases)} train, {len(val_bases)} val, {len(test_bases)} test.")

    # Clean previous synthetic folders and create pattern subdirectories
    for s in ("train", "val", "test"):
        for pattern in ["clear", "developing", "curved_band", "central_dense_overcast", "eye", "sheared", "dissipating"]:
            p = PROCESSED_DIR / s / pattern
            p.mkdir(parents=True, exist_ok=True)
            # Remove old synthetic files
            for old_f in p.glob("*.png"):
                old_f.unlink()

    counts = {"train": 0, "val": 0, "test": 0}

    for item in manifest:
        base = item["storm_name"].split("_")[0]
        if base in train_bases:
            split_name = "train"
        elif base in val_bases:
            split_name = "val"
        else:
            split_name = "test"

        pattern = item["pattern"]
        # Use IR image (primary for morphology)
        src_ir = Path(item["ir_file"])
        if src_ir.exists():
            dest = PROCESSED_DIR / split_name / pattern / src_ir.name
            import shutil
            shutil.copy(src_ir, dest)
            counts[split_name] += 1

        # Also use VIS image as additional sample
        src_vis = Path(item["vis_file"])
        if src_vis.exists():
            dest = PROCESSED_DIR / split_name / pattern / src_vis.name
            import shutil
            shutil.copy(src_vis, dest)
            counts[split_name] += 1

    logger.info(f"Real dataset prepared: train={counts['train']}, val={counts['val']}, test={counts['test']} (Total={sum(counts.values())})")


def main():
    manifest = download_all_real_samples()
    if manifest:
        prepare_real_image_dataset(manifest)


if __name__ == "__main__":
    main()
