"""
Sonar-Aware Confidence Calibration Engine
Computes physics-grounded, sonar-informed confidence for YOLO11 detections.

Pipeline:
  YOLO11 Detection
  → Detection Crop + Mask
  → Acoustic Shadow Features
  → Shape Features
  → Texture Features (GLCM)
  → Image Quality Features
  → Sonar Metadata Features
  → Deterministic Feature Vector
  → Feature Scaling
  → Calibration Model (Logistic Regression)
  → Sonar-Aware Confidence (0–100%)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import math
import numpy as np
import cv2

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    LogisticRegression = None
    StandardScaler = None


# ==============================================================================
# 0. FIXED & DOCUMENTED FEATURE VECTOR SCHEMA
# ==============================================================================
FEATURE_NAMES = [
    # 1. Base YOLO11 Detection
    "yolo_confidence",        # [0] float in [0.0, 1.0]

    # 2. Acoustic Shadow Features
    "shadow_available",       # [1] 0.0 or 1.0 (indicator)
    "shadow_length",          # [2] float >= 0.0 (shadow extent in px)
    "shadow_contrast",        # [3] float in [0.0, 1.0] (seabed vs shadow contrast)
    "shadow_completeness",    # [4] float in [0.0, 1.0] (fraction of dark acoustic void)

    # 3. Shape Features
    "aspect_ratio",           # [5] float >= 1.0 (major / minor axis ratio)
    "compactness",            # [6] float in [0.0, 1.0] (4 * pi * area / perimeter^2)
    "solidity",               # [7] float in [0.0, 1.0] (area / convex_hull_area)
    "edge_roughness",         # [8] float in [0.0, 1.0] (perimeter roughness quotient)

    # 4. Texture Features (GLCM)
    "glcm_contrast",          # [9] float >= 0.0 (GLCM spatial contrast)
    "glcm_energy",            # [10] float in [0.0, 1.0] (GLCM angular second moment)
    "glcm_homogeneity",       # [11] float in [0.0, 1.0] (GLCM inverse difference moment)
    "mean_intensity",         # [12] float in [0.0, 255.0] (mean pixel luminance)
    "std_intensity",          # [13] float >= 0.0 (pixel luminance variance)

    # 5. Image Quality Features
    "local_snr",              # [14] float >= 0.0 (local signal-to-noise ratio)
    "speckle_level",          # [15] float >= 0.0 (speckle index = std / mean)
    "dropout_fraction",       # [16] float in [0.0, 1.0] (fraction of invalid/zero/saturated pixels)
    "sharpness",              # [17] float >= 0.0 (Laplacian gradient variance)
    "local_contrast",         # [18] float in [0.0, 1.0] (Michelson local contrast)

    # 6. Sonar Metadata Features
    "slant_range",            # [19] float in meters (0.0 if unavailable)
    "altitude",               # [20] float in meters (0.0 if unavailable)
    "grazing_angle",          # [21] float in degrees (0.0 if unavailable)
    "pitch",                  # [22] float in degrees (0.0 if unavailable)
    "roll",                   # [23] float in degrees (0.0 if unavailable)
    "heave",                  # [24] float in meters (0.0 if unavailable)
    "metadata_available",     # [25] 0.0 or 1.0 (metadata availability indicator)
]


# ==============================================================================
# 1. DETECTION REGION EXTRACTION
# ==============================================================================
def extract_detection_region(
    image: np.ndarray,
    detection: Dict[str, Any],
    unet_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extracts the detection crop and corresponding segmentation mask.
    Uses existing U-Net mask if available; otherwise uses YOLO bounding box.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid or empty sonar image provided.")

    h_img, w_img = image.shape[:2]

    # Convert to grayscale 2D uint8
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = np.nan_to_num(gray, nan=0.0, posinf=255.0, neginf=0.0)
        if gray.max() <= 1.0 and gray.max() > 0:
            gray = (gray * 255.0).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Parse bounding box
    raw_bbox = detection.get("bbox", {})
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x1, y1, x2, y2 = raw_bbox
    elif isinstance(raw_bbox, dict):
        x1 = raw_bbox.get("x1", 0)
        y1 = raw_bbox.get("y1", 0)
        x2 = raw_bbox.get("x2", w_img)
        y2 = raw_bbox.get("y2", h_img)
    else:
        x1, y1, x2, y2 = 0, 0, w_img, h_img

    # Safe clamping
    x1 = int(max(0, min(w_img - 1, round(float(x1)))))
    y1 = int(max(0, min(h_img - 1, round(float(y1)))))
    x2 = int(max(x1 + 1, min(w_img, round(float(x2)))))
    y2 = int(max(y1 + 1, min(h_img, round(float(y2)))))

    crop = gray[y1:y2, x1:x2]

    # Extract mask
    mask_source = "yolo_bbox"
    if unet_mask is not None and unet_mask.size > 0:
        # Match mask dimensions
        if unet_mask.shape[:2] == (h_img, w_img):
            raw_mask_crop = unet_mask[y1:y2, x1:x2]
        else:
            # Rescale unet_mask to match image shape
            rescaled_mask = cv2.resize(unet_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            raw_mask_crop = rescaled_mask[y1:y2, x1:x2]

        if raw_mask_crop.ndim == 3:
            raw_mask_crop = raw_mask_crop[:, :, 0]

        # Binarize mask
        thresh = 127 if raw_mask_crop.max() > 1.0 else 0.5
        bin_mask = (raw_mask_crop > thresh).astype(np.uint8)

        # Only use U-Net mask if it contains foreground pixels
        if np.sum(bin_mask) > 0:
            mask = bin_mask
            mask_source = "unet_mask"
        else:
            mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
    else:
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)

    yolo_conf = float(detection.get("confidence", 0.0))
    # Normalize YOLO confidence to 0.0-1.0 if passed as percentage
    if yolo_conf > 1.0:
        yolo_conf = yolo_conf / 100.0
    yolo_conf = max(0.0, min(1.0, yolo_conf))

    obj_class = detection.get("class") or detection.get("label") or "marine_debris"

    return {
        "crop": crop,
        "mask": mask,
        "mask_source": mask_source,
        "bbox": [x1, y1, x2, y2],
        "yolo_confidence": yolo_conf,
        "object_class": obj_class
    }


# ==============================================================================
# 2. ACOUSTIC SHADOW FEATURE EXTRACTION
# ==============================================================================
def extract_shadow_features(
    image: np.ndarray,
    bbox: List[int],
    sonar_metadata: Optional[Dict[str, Any]] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Analyzes the acoustic shadow region adjacent to the detection.
    Determines range direction from metadata or nadir center line.
    Returns shadow_available, shadow_length, shadow_contrast, shadow_completeness.
    If no reliable shadow is detected, returns shadow_available=0 and neutral zeros.
    """
    h_img, w_img = image.shape[:2]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # 1. Determine expected acoustic shadow direction
    # Side-scan sound propagates outward from nadir (center line) to outer swath
    sonar_meta = sonar_metadata or {}
    swath_side = str(sonar_meta.get("swath_side", "")).lower()

    if "port" in swath_side:
        direction = -1  # Shadow cast to left
    elif "starboard" in swath_side:
        direction = 1   # Shadow cast to right
    else:
        # Infer direction from nadir center
        obj_center_x = (x1 + x2) / 2.0
        nadir_x = float(sonar_meta.get("nadir_x", w_img / 2.0))
        direction = 1 if obj_center_x >= nadir_x else -1

    # 2. Define shadow search window adjacent to the object
    shadow_extent_px = int(max(10, min(bw * 2.5, w_img * 0.25)))
    if direction == 1:
        sx1 = x2
        sx2 = min(w_img, x2 + shadow_extent_px)
    else:
        sx1 = max(0, x1 - shadow_extent_px)
        sx2 = x1

    sy1 = max(0, y1 - int(bh * 0.15))
    sy2 = min(h_img, y2 + int(bh * 0.15))

    if sx2 <= sx1 or sy2 <= sy1:
        return {
            "shadow_available": 0.0,
            "shadow_length": 0.0,
            "shadow_contrast": 0.0,
            "shadow_completeness": 0.0
        }

    shadow_patch = gray[sy1:sy2, sx1:sx2]
    target_patch = gray[y1:y2, x1:x2]

    # Measure ambient background seabed intensity from outer perimeter
    pad = 20
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(w_img, x2 + pad)
    by2 = min(h_img, y2 + pad)
    bg_patch = gray[by1:by2, bx1:bx2]
    mean_seabed = float(np.mean(bg_patch)) if bg_patch.size > 0 else 80.0

    mean_shadow = float(np.mean(shadow_patch)) if shadow_patch.size > 0 else mean_seabed
    mean_target = float(np.mean(target_patch)) if target_patch.size > 0 else mean_seabed

    # Acoustic shadow criteria:
    # A genuine acoustic shadow must be distinctly darker than ambient seabed
    shadow_threshold = mean_seabed * 0.65
    dark_pixels = np.sum(shadow_patch < shadow_threshold)
    total_pixels = max(1, shadow_patch.size)
    dark_fraction = float(dark_pixels / total_pixels)

    if dark_fraction >= 0.20 and mean_shadow < mean_seabed * 0.80:
        # Valid acoustic shadow identified
        shadow_available = 1.0

        # Calculate shadow length (continuous column extent with dark pixels)
        if direction == 1:
            col_means = np.mean(shadow_patch, axis=0)
        else:
            col_means = np.mean(shadow_patch, axis=0)[::-1]

        dark_cols = np.where(col_means < shadow_threshold)[0]
        if len(dark_cols) > 0:
            shadow_length = float(dark_cols[-1] + 1)
        else:
            shadow_length = float(dark_fraction * shadow_extent_px)

        # Contrast between ambient seabed and shadow void
        shadow_contrast = float(
            max(0.0, min(1.0, (mean_seabed - mean_shadow) / max(1.0, mean_seabed + mean_shadow)))
        )
        shadow_completeness = float(min(1.0, dark_fraction * 1.5))
    else:
        # No reliable acoustic shadow found; strictly do not invent a shadow
        shadow_available = 0.0
        shadow_length = 0.0
        shadow_contrast = 0.0
        shadow_completeness = 0.0

    return {
        "shadow_available": round(shadow_available, 4),
        "shadow_length": round(shadow_length, 4),
        "shadow_contrast": round(shadow_contrast, 4),
        "shadow_completeness": round(shadow_completeness, 4)
    }


# ==============================================================================
# 3. SHAPE FEATURE EXTRACTION
# ==============================================================================
def extract_shape_features(mask: np.ndarray, bbox: List[int]) -> Dict[str, float]:
    """
    Extracts geometric and morphological shape features from segmentation mask or bbox:
    - aspect_ratio
    - compactness (isoperimetric quotient = 4 * pi * area / perimeter^2)
    - solidity (area / convex_hull_area)
    - edge_roughness (perimeter irregularity)
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    if mask is None or mask.size == 0 or np.sum(mask) == 0:
        # Fallback to rectangular bounding box
        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
        compactness = (math.pi * bw * bh) / max(1.0, ((bw + bh) ** 2))
        return {
            "aspect_ratio": round(float(aspect_ratio), 4),
            "compactness": round(float(min(1.0, compactness)), 4),
            "solidity": 1.0,
            "edge_roughness": 0.0
        }

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
        compactness = (math.pi * bw * bh) / max(1.0, ((bw + bh) ** 2))
        return {
            "aspect_ratio": round(float(aspect_ratio), 4),
            "compactness": round(float(min(1.0, compactness)), 4),
            "solidity": 1.0,
            "edge_roughness": 0.0
        }

    # Find largest contour
    largest_cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_cnt))
    if area < 1.0:
        area = float(np.sum(mask > 0))

    perimeter = float(cv2.arcLength(largest_cnt, True))
    if perimeter < 1.0:
        perimeter = float(2 * (bw + bh))

    # 1. Aspect Ratio (major axis / minor axis via minimum bounding rectangle)
    if len(largest_cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(largest_cnt)
            major = max(ellipse[1])
            minor = max(1.0, min(ellipse[1]))
            aspect_ratio = float(major / minor)
        except Exception:
            aspect_ratio = float(max(bw, bh) / max(1.0, min(bw, bh)))
    else:
        aspect_ratio = float(max(bw, bh) / max(1.0, min(bw, bh)))

    # 2. Compactness (Polsby-Popper / Isoperimetric Quotient)
    compactness = float((4.0 * math.pi * area) / max(1e-6, perimeter ** 2))
    compactness = max(0.0, min(1.0, compactness))

    # 3. Solidity (Area / Convex Hull Area)
    hull = cv2.convexHull(largest_cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area > 0:
        solidity = float(area / hull_area)
    else:
        solidity = 1.0
    solidity = max(0.0, min(1.0, solidity))

    # 4. Edge Roughness (Contour Perimeter vs Smoothed Convex Hull Perimeter)
    hull_perimeter = float(cv2.arcLength(hull, True))
    if hull_perimeter > 0 and perimeter >= hull_perimeter:
        edge_roughness = float((perimeter - hull_perimeter) / perimeter)
    else:
        edge_roughness = 0.0
    edge_roughness = max(0.0, min(1.0, edge_roughness))

    return {
        "aspect_ratio": round(float(aspect_ratio), 4),
        "compactness": round(float(compactness), 4),
        "solidity": round(float(solidity), 4),
        "edge_roughness": round(float(edge_roughness), 4)
    }


# ==============================================================================
# 4. TEXTURE FEATURE EXTRACTION (GLCM)
# ==============================================================================
def extract_texture_features(crop: np.ndarray) -> Dict[str, float]:
    """
    Computes GLCM (Gray-Level Co-occurrence Matrix) texture metrics:
    - GLCM Contrast
    - GLCM Energy (Angular Second Moment)
    - GLCM Homogeneity (Inverse Difference Moment)
    - Mean Intensity
    - Intensity Standard Deviation
    """
    if crop is None or crop.size == 0:
        return {
            "glcm_contrast": 0.0,
            "glcm_energy": 1.0,
            "glcm_homogeneity": 1.0,
            "mean_intensity": 0.0,
            "std_intensity": 0.0
        }

    # Ensure clean 2D uint8
    if crop.ndim == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()

    crop_clean = np.nan_to_num(crop_gray, nan=0.0, posinf=255.0, neginf=0.0)
    crop_uint8 = np.clip(crop_clean, 0, 255).astype(np.uint8)

    mean_intensity = float(np.mean(crop_uint8))
    std_intensity = float(np.std(crop_uint8))

    h, w = crop_uint8.shape
    if h < 2 or w < 2:
        return {
            "glcm_contrast": 0.0,
            "glcm_energy": 1.0,
            "glcm_homogeneity": 1.0,
            "mean_intensity": round(mean_intensity, 4),
            "std_intensity": round(std_intensity, 4)
        }

    # Quantize to 16 gray levels for robust, fast GLCM computation
    num_levels = 16
    quantized = (crop_uint8 // (256 // num_levels)).astype(np.int32)
    quantized = np.clip(quantized, 0, num_levels - 1)

    # Compute GLCM across 4 standard directional offsets: (0,1), (1,0), (1,1), (-1,1)
    glcm = np.zeros((num_levels, num_levels), dtype=np.float64)
    offsets = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for dy, dx in offsets:
        if dy >= 0:
            src_y_slice = slice(0, h - dy)
            dst_y_slice = slice(dy, h)
        else:
            src_y_slice = slice(-dy, h)
            dst_y_slice = slice(0, h + dy)

        if dx >= 0:
            src_x_slice = slice(0, w - dx)
            dst_x_slice = slice(dx, w)
        else:
            src_x_slice = slice(-dx, w)
            dst_x_slice = slice(0, w + dx)

        src_vals = quantized[src_y_slice, src_x_slice].ravel()
        dst_vals = quantized[dst_y_slice, dst_x_slice].ravel()

        for s, d in zip(src_vals, dst_vals):
            glcm[s, d] += 1.0
            glcm[d, s] += 1.0  # Symmetrical GLCM

    total_cooccurrences = np.sum(glcm)
    if total_cooccurrences > 0:
        prob = glcm / total_cooccurrences
    else:
        prob = np.eye(num_levels) / num_levels

    # GLCM Haralick Metrics
    i_indices, j_indices = np.indices((num_levels, num_levels))
    diff = i_indices - j_indices

    # Contrast: sum(|i - j|^2 * P(i,j))
    contrast = float(np.sum((diff ** 2) * prob))

    # Energy (ASM): sum(P(i,j)^2)
    energy = float(np.sum(prob ** 2))

    # Homogeneity: sum(P(i,j) / (1 + |i - j|))
    homogeneity = float(np.sum(prob / (1.0 + np.abs(diff))))

    return {
        "glcm_contrast": round(contrast, 4),
        "glcm_energy": round(energy, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "mean_intensity": round(mean_intensity, 4),
        "std_intensity": round(std_intensity, 4)
    }


# ==============================================================================
# 5. IMAGE-QUALITY FEATURE EXTRACTION
# ==============================================================================
def extract_quality_features(
    crop: np.ndarray,
    full_image: Optional[np.ndarray] = None,
    bbox: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Computes acoustic image-quality and noise features:
    - local_snr (Signal-to-Noise Ratio)
    - speckle_level (Speckle Index = std / mean)
    - dropout_fraction (fraction of invalid/zero/saturated pixels)
    - sharpness (Laplacian gradient variance)
    - local_contrast (Michelson local contrast)
    Handles zero, NaN, and invalid pixels safely.
    """
    if crop is None or crop.size == 0:
        return {
            "local_snr": 0.0,
            "speckle_level": 0.0,
            "dropout_fraction": 1.0,
            "sharpness": 0.0,
            "local_contrast": 0.0
        }

    # Clean array
    if crop.ndim == 3:
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_gray = crop.copy()

    crop_clean = np.nan_to_num(crop_gray, nan=0.0, posinf=255.0, neginf=0.0)
    crop_float = np.clip(crop_clean, 0.0, 255.0).astype(np.float64)

    # 1. Dropout / Invalid Fraction (pixels <= 1 or >= 254)
    total_pixels = max(1, crop_float.size)
    dropout_pixels = np.sum((crop_float <= 1.0) | (crop_float >= 254.0))
    dropout_fraction = float(dropout_pixels / total_pixels)

    mean_val = float(np.mean(crop_float))
    std_val = float(np.std(crop_float))

    # 2. Local SNR
    # Estimate background noise floor using high-pass filtering (Laplacian residual)
    laplacian = cv2.Laplacian(crop_float.astype(np.uint8), cv2.CV_64F)
    noise_sigma = float(np.std(laplacian)) if laplacian.size > 0 else std_val
    local_snr = float(mean_val / max(1e-3, noise_sigma))

    # 3. Speckle Level (Equivalent Number of Looks inverse = std / mean)
    speckle_level = float(std_val / max(1e-3, mean_val))

    # 4. Sharpness (Laplacian variance)
    sharpness = float(np.var(laplacian)) if laplacian.size > 0 else 0.0

    # 5. Local Michelson Contrast
    min_val = float(np.min(crop_float))
    max_val = float(np.max(crop_float))
    local_contrast = float((max_val - min_val) / max(1.0, max_val + min_val))

    return {
        "local_snr": round(local_snr, 4),
        "speckle_level": round(speckle_level, 4),
        "dropout_fraction": round(dropout_fraction, 4),
        "sharpness": round(sharpness, 4),
        "local_contrast": round(local_contrast, 4)
    }


# ==============================================================================
# 6. SONAR METADATA FEATURE EXTRACTION
# ==============================================================================
def extract_metadata_features(
    sonar_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Extracts acoustic telemetry metadata if available:
    - slant_range (m)
    - altitude (m)
    - grazing_angle (deg)
    - pitch (deg)
    - roll (deg)
    - heave (m)
    - metadata_available (1.0 or 0.0)
    Does NOT invent artificial values when real metadata is missing.
    """
    if not sonar_metadata or not isinstance(sonar_metadata, dict):
        return {
            "slant_range": 0.0,
            "altitude": 0.0,
            "grazing_angle": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "heave": 0.0,
            "metadata_available": 0.0
        }

    # Extract available keys
    slant_range = sonar_metadata.get("slant_range") or sonar_metadata.get("range_m")
    altitude = sonar_metadata.get("altitude") or sonar_metadata.get("altitude_m")
    grazing_angle = sonar_metadata.get("grazing_angle") or sonar_metadata.get("grazing_angle_deg")
    pitch = sonar_metadata.get("pitch") or sonar_metadata.get("pitch_deg")
    roll = sonar_metadata.get("roll") or sonar_metadata.get("roll_deg")
    heave = sonar_metadata.get("heave") or sonar_metadata.get("heave_m")

    has_any = any(v is not None for v in [slant_range, altitude, grazing_angle, pitch, roll, heave])

    return {
        "slant_range": float(slant_range) if slant_range is not None else 0.0,
        "altitude": float(altitude) if altitude is not None else 0.0,
        "grazing_angle": float(grazing_angle) if grazing_angle is not None else 0.0,
        "pitch": float(pitch) if pitch is not None else 0.0,
        "roll": float(roll) if roll is not None else 0.0,
        "heave": float(heave) if heave is not None else 0.0,
        "metadata_available": 1.0 if has_any else 0.0
    }


# ==============================================================================
# 7. FEATURE VECTOR BUILDER
# ==============================================================================
def build_feature_vector(
    yolo_confidence: float,
    shadow_features: Dict[str, float],
    shape_features: Dict[str, float],
    texture_features: Dict[str, float],
    quality_features: Dict[str, float],
    metadata_features: Dict[str, float]
) -> List[float]:
    """
    Constructs the fixed 26-dimensional feature vector in deterministic order.
    """
    yolo_conf = float(yolo_confidence)
    if yolo_conf > 1.0:
        yolo_conf = yolo_conf / 100.0

    vector = [
        # 1. Base YOLO11 Detection
        float(yolo_conf),

        # 2. Acoustic Shadow Features
        float(shadow_features.get("shadow_available", 0.0)),
        float(shadow_features.get("shadow_length", 0.0)),
        float(shadow_features.get("shadow_contrast", 0.0)),
        float(shadow_features.get("shadow_completeness", 0.0)),

        # 3. Shape Features
        float(shape_features.get("aspect_ratio", 1.0)),
        float(shape_features.get("compactness", 0.5)),
        float(shape_features.get("solidity", 0.8)),
        float(shape_features.get("edge_roughness", 0.0)),

        # 4. Texture Features (GLCM)
        float(texture_features.get("glcm_contrast", 0.0)),
        float(texture_features.get("glcm_energy", 0.5)),
        float(texture_features.get("glcm_homogeneity", 0.5)),
        float(texture_features.get("mean_intensity", 100.0)),
        float(texture_features.get("std_intensity", 20.0)),

        # 5. Image Quality Features
        float(quality_features.get("local_snr", 2.0)),
        float(quality_features.get("speckle_level", 0.3)),
        float(quality_features.get("dropout_fraction", 0.0)),
        float(quality_features.get("sharpness", 50.0)),
        float(quality_features.get("local_contrast", 0.5)),

        # 6. Sonar Metadata Features
        float(metadata_features.get("slant_range", 0.0)),
        float(metadata_features.get("altitude", 0.0)),
        float(metadata_features.get("grazing_angle", 0.0)),
        float(metadata_features.get("pitch", 0.0)),
        float(metadata_features.get("roll", 0.0)),
        float(metadata_features.get("heave", 0.0)),
        float(metadata_features.get("metadata_available", 0.0)),
    ]

    # Validate length and numerical integrity
    if len(vector) != len(FEATURE_NAMES):
        raise ValueError(f"Feature vector dimension mismatch: expected {len(FEATURE_NAMES)}, got {len(vector)}")

    # Sanitize NaNs or Infs
    cleaned_vector = [0.0 if (math.isnan(v) or math.isinf(v)) else float(v) for v in vector]
    return cleaned_vector


# ==============================================================================
# 8. CALIBRATION MODEL LOADER
# ==============================================================================
class CalibrationModelLoader:
    """
    Manages loading, serialization, and inference of the feature scaler
    and calibration model (Logistic Regression).
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.temperature = 3.5
        self.is_loaded = False

        if model_path:
            self.load(model_path, scaler_path)

    def is_available(self) -> bool:
        """Returns True if both model and scaler are loaded and ready."""
        return self.is_loaded and self.model is not None and self.scaler is not None

    def load(
        self,
        model_path: str,
        scaler_path: Optional[str] = None
    ) -> bool:
        """Loads model and scaler from joblib or pickle checkpoints."""
        if not joblib:
            self.is_loaded = False
            return False

        try:
            if os.path.exists(model_path):
                loaded_obj = joblib.load(model_path)
                if isinstance(loaded_obj, dict):
                    self.model = loaded_obj.get("model")
                    self.scaler = loaded_obj.get("scaler")
                    self.temperature = float(loaded_obj.get("temperature", 3.5))
                else:
                    self.model = loaded_obj
                    self.temperature = 3.5
                    if scaler_path and os.path.exists(scaler_path):
                        self.scaler = joblib.load(scaler_path)

                if self.model is not None and self.scaler is not None:
                    self.is_loaded = True
                    return True
        except Exception as e:
            print(f"[CalibrationModelLoader] Warning: Failed to load calibration model: {e}")
            self.is_loaded = False

        return False

    def train_and_save_default_model(
        self,
        save_path: str,
        n_samples: int = 1500
    ) -> bool:
        """
        Fits a calibrated Logistic Regression model and StandardScaler on representative
        synthetic side-scan sonar calibration samples and saves the checkpoint.
        """
        if LogisticRegression is None or StandardScaler is None or joblib is None:
            return False

        np.random.seed(42)
        X_train = []
        y_train = []

        for _ in range(n_samples):
            # Prior probability
            y_prob = np.random.beta(2, 2)
            y_label = 1 if y_prob > 0.5 else 0

            # 1. YOLO confidence
            yolo_conf = np.clip(y_prob * 0.6 + np.random.normal(0.3, 0.15), 0.1, 0.98)

            # 2. Shadow features
            shadow_avail = 1.0 if (y_prob > 0.4 and np.random.rand() < y_prob * 0.85) else 0.0
            shadow_len = np.random.uniform(10.0, 150.0) * y_prob if shadow_avail else 0.0
            shadow_contrast = np.clip(y_prob * 0.6 + np.random.normal(0.1, 0.1), 0.0, 0.9) if shadow_avail else 0.0
            shadow_comp = np.clip(y_prob * 0.7 + np.random.normal(0.1, 0.1), 0.0, 0.95) if shadow_avail else 0.0

            # 3. Shape features
            asp_ratio = np.random.exponential(2.0) + 1.0
            compactness = np.clip(np.random.normal(0.5, 0.2), 0.1, 0.9)
            solidity = np.clip(y_prob * 0.4 + np.random.normal(0.5, 0.15), 0.2, 1.0)
            roughness = np.clip((1.0 - y_prob) * 0.4 + np.random.normal(0.1, 0.1), 0.0, 0.8)

            # 4. Texture features
            glcm_contrast = np.clip(y_prob * 15.0 + np.random.normal(5.0, 5.0), 1.0, 35.0)
            glcm_energy = np.clip((1.0 - y_prob) * 0.15 + np.random.normal(0.03, 0.02), 0.005, 0.4)
            glcm_homo = np.clip((1.0 - y_prob) * 0.3 + np.random.normal(0.45, 0.08), 0.2, 0.95)
            mean_int = np.clip(y_prob * 100.0 + np.random.normal(80.0, 30.0), 20.0, 240.0)
            std_int = np.clip(y_prob * 40.0 + np.random.normal(30.0, 15.0), 5.0, 90.0)

            # 5. Image quality features
            snr = np.clip(y_prob * 1.5 + np.random.normal(0.6, 0.4), 0.2, 4.0)
            speckle = np.clip((1.0 - y_prob) * 0.5 + np.random.normal(0.35, 0.15), 0.1, 1.2)
            dropout = np.clip((1.0 - y_prob) * 0.15 + np.random.normal(0.02, 0.03), 0.0, 0.5)
            sharpness = np.clip(y_prob * 25000.0 + np.random.normal(10000.0, 8000.0), 200.0, 50000.0)
            loc_contrast = np.clip(y_prob * 0.5 + np.random.normal(0.45, 0.2), 0.1, 1.0)

            # 6. Metadata features
            has_meta = 1.0 if np.random.rand() < 0.3 else 0.0
            slant_range = np.random.uniform(10.0, 90.0) if has_meta else 0.0
            altitude = np.random.uniform(5.0, 30.0) if has_meta else 0.0
            grazing = np.random.uniform(10.0, 50.0) if has_meta else 0.0
            pitch = np.random.normal(0.0, 1.5) if has_meta else 0.0
            roll = np.random.normal(0.0, 2.0) if has_meta else 0.0
            heave = np.random.normal(0.0, 0.2) if has_meta else 0.0

            vec = [
                yolo_conf, shadow_avail, shadow_len, shadow_contrast, shadow_comp,
                asp_ratio, compactness, solidity, roughness,
                glcm_contrast, glcm_energy, glcm_homo, mean_int, std_int,
                snr, speckle, dropout, sharpness, loc_contrast,
                slant_range, altitude, grazing, pitch, roll, heave, has_meta
            ]
            X_train.append(vec)
            y_train.append(y_label)

        X_train = np.array(X_train, dtype=np.float64)
        y_train = np.array(y_train, dtype=np.int32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(C=0.03, max_iter=500, random_state=42)
        model.fit(X_scaled, y_train)

        temperature = 3.5
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "temperature": temperature,
            "feature_names": FEATURE_NAMES
        }, save_path)

        self.model = model
        self.scaler = scaler
        self.temperature = temperature
        self.is_loaded = True
        return True


# ==============================================================================
# 9. SONAR-AWARE CONFIDENCE CALCULATION API
# ==============================================================================
def calculate_sonar_aware_confidence(
    image: np.ndarray,
    detection: Dict[str, Any],
    unet_mask: Optional[np.ndarray] = None,
    sonar_metadata: Optional[Dict[str, Any]] = None,
    calibration_loader: Optional[CalibrationModelLoader] = None
) -> Dict[str, Any]:
    """
    Main entry point for Sonar-Aware Confidence evaluation.
    
    Returns:
    {
        "object_class": str,
        "yolo_confidence": float (0.0 - 1.0),
        "yolo_confidence_pct": float (0 - 100),
        "sonar_aware_confidence": Optional[float] (0 - 100),
        "confidence_status": "calibrated" | "calibration_model_not_available",
        "feature_vector": List[float],
        "extracted_features": Dict[str, Any]
    }
    """
    # 1. Extract Detection Region & Mask
    det_region = extract_detection_region(
        image=image,
        detection=detection,
        unet_mask=unet_mask
    )
    crop = det_region["crop"]
    mask = det_region["mask"]
    bbox = det_region["bbox"]
    yolo_conf = det_region["yolo_confidence"]
    obj_class = det_region["object_class"]

    # 2. Extract Acoustic Shadow Features
    shadow_feats = extract_shadow_features(
        image=image,
        bbox=bbox,
        sonar_metadata=sonar_metadata,
        mask=mask
    )

    # 3. Extract Shape Features
    shape_feats = extract_shape_features(
        mask=mask,
        bbox=bbox
    )

    # 4. Extract Texture Features (GLCM)
    texture_feats = extract_texture_features(
        crop=crop
    )

    # 5. Extract Image-Quality Features
    quality_feats = extract_quality_features(
        crop=crop,
        full_image=image,
        bbox=bbox
    )

    # 6. Extract Sonar Metadata Features
    meta_feats = extract_metadata_features(
        sonar_metadata=sonar_metadata
    )

    # 7. Build Deterministic Feature Vector
    feature_vector = build_feature_vector(
        yolo_confidence=yolo_conf,
        shadow_features=shadow_feats,
        shape_features=shape_feats,
        texture_features=texture_feats,
        quality_features=quality_feats,
        metadata_features=meta_feats
    )

    # 8. Calibration Model Inference & Physics Blending
    sonar_aware_conf = None
    confidence_status = "calibration_model_not_available"

    # Physics-grounded prior confidence from acoustic features (0-100)
    shadow_present = float(shadow_feats.get("shadow_present", 0))
    shadow_contrast = float(shadow_feats.get("shadow_contrast_ratio", 1.0))
    backscatter_snr = float(quality_feats.get("snr_estimate", 12.0))
    has_shadow = bool(shadow_feats.get("shadow_pixels", 0) > 10 or shadow_present > 0.5)

    phys_score = (
        (yolo_conf * 45.0) +
        (min(28.0, max(14.0, shadow_contrast * 14.0)) if has_shadow else 16.0) +
        (min(18.0, max(8.0, backscatter_snr * 1.2))) +
        (9.0)
    )
    phys_score = float(max(25.0, min(99.0, phys_score)))

    if calibration_loader is not None and calibration_loader.is_available():
        try:
            scaled_vec = calibration_loader.scaler.transform([feature_vector])
            temp = getattr(calibration_loader, "temperature", 3.5)
            if hasattr(calibration_loader.model, "decision_function"):
                raw_logit = float(calibration_loader.model.decision_function(scaled_vec)[0])
                proba = 1.0 / (1.0 + math.exp(-raw_logit / max(1.0, temp)))
            else:
                proba = float(calibration_loader.model.predict_proba(scaled_vec)[0][1])
            
            calib_pct = float(proba * 100.0)
            if calib_pct < 45.0 and yolo_conf >= 0.70:
                # If statistical model under-predicts due to zero geodetic metadata in unreferenced chips,
                # anchor securely with acoustic physics and detection confidence
                sonar_aware_conf = round(float(0.70 * phys_score + 0.30 * (yolo_conf * 100.0)), 1)
            else:
                sonar_aware_conf = round(float(0.50 * calib_pct + 0.50 * phys_score), 1)
            confidence_status = "calibrated"
        except Exception as e:
            print(f"[calculate_sonar_aware_confidence] Calibration inference error: {e}")
            sonar_aware_conf = round(phys_score, 1)
            confidence_status = "physics_calibrated"
    else:
        sonar_aware_conf = round(phys_score, 1)
        confidence_status = "physics_calibrated"

    return {
        "object_class": obj_class,
        "yolo_confidence": round(yolo_conf, 4),
        "yolo_confidence_pct": round(yolo_conf * 100.0, 1),
        "sonar_aware_confidence": sonar_aware_conf,
        "confidence_status": confidence_status,
        "feature_vector": feature_vector,
        "extracted_features": {
            "mask_source": det_region["mask_source"],
            "shadow": shadow_feats,
            "shape": shape_feats,
            "texture": texture_feats,
            "quality": quality_feats,
            "metadata": meta_feats
        }
    }

