"""
Classical threshold-based cyclone detection and localization using scipy.ndimage and numpy.
No OpenCV dependency required.

For tropical cyclones, the defining satellite signature is a large region of
cold cloud tops (low brightness temperature in thermal IR). This detector:
1. Thresholds the IR channel on brightness temperature (cold = bright in enhanced IR)
2. Identifies the largest connected component (the cyclonic system)
3. Returns bounding box and centroid

This is a prototype detector — production would use trained object detection (YOLO, Faster R-CNN).
"""

import logging
from typing import Tuple, Optional
from math import radians, sin, cos, sqrt, atan2

import numpy as np
from PIL import Image
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_cyclone_simple(image_array: np.ndarray, threshold: int = 180, channel: str = "ir") -> Optional[dict]:
    """
    Detect cyclone in satellite image using brightness threshold with quality filters.

    Args:
        image_array: RGB image as numpy array (H, W, 3), uint8
        threshold: Brightness threshold (0-255). For IR imagery, cold cloud tops
                   appear bright (high pixel values). For visible/true color,
                   use grayscale intensity.
        channel: 'ir' for thermal infrared or 'vis' for true color visible

    Returns:
        dict with keys: bbox (x, y, w, h), centroid (cx, cy), area, confidence, channel, method
        or None if no cyclone detected
    """
    if len(image_array.shape) == 3:
        # Convert to grayscale: standard luminance formula
        gray = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
    else:
        gray = image_array.astype(float)

    H, W = gray.shape
    total_pixels = H * W
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Quality filter 1: Reject flat/uniform images (blank white, solid color, no structure)
    if std_val < 12.0:
        return None

    # Quality filter 2: Reject fully saturated or dark images
    if mean_val > 245.0 or mean_val < 15.0:
        return None

    # Channel-specific threshold
    if channel.lower() == "ir":
        threshold = 175  # Cold cloud tops in IR Band 31
    else:
        threshold = 185  # Bright clouds in visible

    # Threshold: bright pixels (cold clouds) are potential cyclone regions
    binary = gray > threshold

    # Morphological operations to clean up noise using scipy
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    binary = ndimage.binary_closing(binary, structure=structure, iterations=2)
    binary = ndimage.binary_opening(binary, structure=structure, iterations=1)

    # Label connected components
    labeled_array, num_features = ndimage.label(binary, structure=structure)

    if num_features == 0:
        return None

    # Compute areas for each component
    component_sizes = ndimage.sum(binary, labeled_array, range(1, num_features + 1))
    if len(component_sizes) == 0:
        return None

    # Find the largest component
    largest_idx = int(np.argmax(component_sizes)) + 1  # 1-indexed
    largest_mask = (labeled_array == largest_idx)
    area = int(component_sizes[largest_idx - 1])

    # Quality filter 3: Area must be between 1% (small developing system) and 85% (not full-frame artifact)
    min_area = int(0.01 * total_pixels)
    max_area = int(0.85 * total_pixels)
    if area < min_area or area > max_area:
        return None

    # Find bounding box
    rows = np.any(largest_mask, axis=1)
    cols = np.any(largest_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x = int(cmin)
    y = int(rmin)
    w = int(cmax - cmin + 1)
    h = int(rmax - rmin + 1)

    # Quality filter 4: Aspect ratio check (cyclones are not extreme 1D lines)
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))
    if aspect_ratio > 5.0:
        return None

    # Centroid (center of mass)
    cy, cx = ndimage.center_of_mass(largest_mask)

    # Quality filter 5: Contrast check - cyclone cold core or bright cluster must stand out from background
    core_mean = float(np.mean(gray[largest_mask]))
    bg_mask = ~largest_mask
    bg_mean = float(np.mean(gray[bg_mask])) if np.any(bg_mask) else 0.0
    contrast = core_mean - bg_mean

    if contrast < 20.0:
        return None

    # Compute confidence based on area, contrast, and structure quality
    rel_area = area / total_pixels
    confidence = float(np.clip(0.45 + rel_area * 1.5 + (contrast / 255.0) * 0.4, 0.50, 0.96))

    return {
        "bbox": (x, y, w, h),
        "centroid": (round(float(cx), 2), round(float(cy), 2)),
        "area": area,
        "confidence": round(confidence, 3),
        "channel": channel,
        "method": f"morphological-{channel}-localizer",
    }


def detect_cyclone_from_file(image_path: str, threshold: int = 180, channel: str = "ir") -> Optional[dict]:
    """Load image from file and detect cyclone."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    return detect_cyclone_simple(img_array, threshold=threshold, channel=channel)


def detect_cyclone_from_bytes(image_bytes: bytes, threshold: int = 180, channel: str = "ir") -> Optional[dict]:
    """Load image from bytes and detect cyclone."""
    from io import BytesIO
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)
    return detect_cyclone_simple(img_array, threshold=threshold, channel=channel)


def validate_detection_against_ibtracs(
    detection: dict, true_lat: float, true_lon: float, image_bbox_deg: Tuple[float, float, float, float], img_size: int = 128
) -> dict:
    """
    Validate detected centroid against known IBTrACS position.

    Args:
        detection: Output from detect_cyclone_simple
        true_lat, true_lon: Ground truth position from IBTrACS
        image_bbox_deg: Image extent as (min_lat, min_lon, max_lat, max_lon)
        img_size: Size of image in pixels (assumes square)

    Returns:
        dict with distance_km, pixel_error, lat_detected, lon_detected
    """
    if detection is None:
        return {"error": "No detection", "distance_km": None}

    cx_px, cy_px = detection["centroid"]
    min_lat, min_lon, max_lat, max_lon = image_bbox_deg

    # Linear mapping from pixel to lat/lon
    detected_lat = max_lat - (cy_px / img_size) * (max_lat - min_lat)
    detected_lon = min_lon + (cx_px / img_size) * (max_lon - min_lon)

    # Compute distance using haversine
    R = 6371.0  # Earth radius in km
    dlat = radians(true_lat - detected_lat)
    dlon = radians(true_lon - detected_lon)
    a = sin(dlat / 2) ** 2 + cos(radians(detected_lat)) * cos(radians(true_lat)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance_km = R * c

    pixel_error = sqrt((cx_px - img_size / 2) ** 2 + (cy_px - img_size / 2) ** 2)

    return {
        "detected_lat": round(detected_lat, 2),
        "detected_lon": round(detected_lon, 2),
        "true_lat": true_lat,
        "true_lon": true_lon,
        "distance_km": round(distance_km, 1),
        "pixel_error": round(pixel_error, 1),
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MANIFEST_FILE = DATA_DIR / "processed" / "real_satellite_manifest.json"

    if not MANIFEST_FILE.exists():
        logger.error(f"Manifest file not found: {MANIFEST_FILE}")
        import sys
        sys.exit(1)

    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info("=" * 60)
    logger.info("VALIDATING CYCLONE DETECTION ON REAL SATELLITE IMAGES")
    logger.info("=" * 60)

    results = []
    for item in manifest[:10]:  # Test on first 10 storm states
        ir_file = item["ir_file"]
        name = item["storm_name"]
        true_lat = item["lat"]
        true_lon = item["lon"]

        # Image bbox was centered on (true_lat, true_lon) with ±4.0 deg
        image_bbox = (true_lat - 4.0, true_lon - 4.0, true_lat + 4.0, true_lon + 4.0)

        detection = detect_cyclone_from_file(ir_file, threshold=180)
        val = validate_detection_against_ibtracs(detection, true_lat, true_lon, image_bbox)

        status = "DETECTED" if detection else "MISSED"
        dist_str = f"{val.get('distance_km', 'N/A')} km" if detection else "N/A"
        logger.info(f"Storm: {name:<20} | Status: {status:<8} | Offset: {dist_str}")
        results.append({"storm": name, "status": status, "validation": val})

    logger.info("=" * 60)
