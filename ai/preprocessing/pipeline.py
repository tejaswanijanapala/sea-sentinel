"""
Layer 2: Sonar Preprocessing & Signal Enhancement Pipeline
Supports configurable normalization, speckle noise reduction (Lee filter, adaptive median),
contrast enhancement (CLAHE), acoustic shadow-highlight pairing, and mosaic tiling.
Preserves original images.
"""
from typing import Dict, Any, Tuple, Optional, List
import os
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

class SonarPreprocessor:
    """
    Modular acoustic image preprocessing engine specifically calibrated for Side-Scan Sonar (SSS).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "target_size": (640, 640),
            "normalize": True,
            "normalize_method": "min_max",
            "denoise_method": "lee",
            "lee_filter_size": 5,
            "lee_filter_var": 0.25,
            "contrast_method": "clahe",
            "clahe_clip_limit": 2.5,
            "clahe_grid_size": (8, 8),
            "shadow_threshold": 35,
            "highlight_threshold": 195,
            "tiling": {
                "tile_size": (640, 640),
                "overlap": 128
            }
        }

    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """
        Validates whether input is a readable image file and checks format/existence.
        """
        if not os.path.exists(image_path):
            return {"valid": False, "error": f"File not found: {image_path}"}
        
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"]:
            return {"valid": False, "error": f"Unsupported format: {ext}"}
            
        file_size_bytes = os.path.getsize(image_path)
        if file_size_bytes == 0:
            return {"valid": False, "error": "Empty file"}
            
        return {
            "valid": True,
            "path": image_path,
            "extension": ext,
            "size_bytes": file_size_bytes,
            "status": "ready_for_preprocessing"
        }

    def load_image_as_grayscale(self, image_input: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Loads image from filepath or numpy array, ensuring 8-bit single-channel acoustic backscatter.
        """
        meta = {}
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Sonar image not found: {image_input}")
            img = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image: {image_input}")
            meta["source_path"] = image_input
            meta["original_shape"] = img.shape
            meta["original_dtype"] = str(img.dtype)
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
            meta["source_path"] = "in_memory"
            meta["original_shape"] = img.shape
            meta["original_dtype"] = str(img.dtype)
        else:
            raise TypeError("Expected image path (str) or numpy ndarray.")

        # Convert to single channel grayscale if multi-channel
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize 16-bit to 8-bit if applicable
        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return img, meta

    def normalize(self, img: np.ndarray, method: str = "min_max") -> np.ndarray:
        """
        Normalizes pixel intensity:
          - "min_max": scales to [0, 1] floating point
          - "min_max_uint8": scales to full [0, 255] dynamic range
          - "z_score": zero mean, unit variance
        """
        img_f = img.astype(np.float32)
        min_val, max_val = np.min(img_f), np.max(img_f)

        if method == "min_max":
            if max_val > min_val:
                return (img_f - min_val) / (max_val - min_val)
            return np.zeros_like(img_f)

        elif method == "min_max_uint8":
            if max_val > min_val:
                normed = ((img_f - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                return normed
            return np.zeros_like(img, dtype=np.uint8)

        elif method == "z_score":
            mean, std = np.mean(img_f), np.std(img_f)
            if std > 1e-6:
                return (img_f - mean) / std
            return img_f - mean

        return img

    def denoise_speckle_lee(self, img: np.ndarray, size: int = 5, noise_var: float = 0.25) -> np.ndarray:
        """
        Lee Speckle Filter for multiplicative sonar speckle reduction.
        Preserves sharp object-shadow boundaries while smoothing homogenous seafloor patches.
        """
        img_f = img.astype(np.float32)
        mean = uniform_filter(img_f, size=size)
        mean_sq = uniform_filter(img_f ** 2, size=size)
        variance = np.maximum(mean_sq - (mean ** 2), 0.0)

        # Weighting factor W:
        # W = variance / (variance + noise_var * (mean ** 2))
        denominator = variance + (noise_var * (mean ** 2))
        weights = np.zeros_like(variance)
        mask = denominator > 1e-6
        weights[mask] = variance[mask] / denominator[mask]
        weights = np.clip(weights, 0.0, 1.0)

        filtered = mean + weights * (img_f - mean)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        return filtered

    def denoise_adaptive_median(self, img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Applies median filter for suppressing impulsive acoustic noise spikes.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(img, kernel_size)

    def enhance_contrast_clahe(
        self,
        img: np.ndarray,
        clip_limit: float = 2.5,
        grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Prevents over-amplifying uniform seabed noise while boosting low-relief debris contacts.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)

    def extract_shadow_highlight_pair(
        self,
        img: np.ndarray,
        shadow_thresh: int = 35,
        highlight_thresh: int = 195,
        min_area_px: int = 20
    ) -> Dict[str, Any]:
        """
        Extracts acoustic shadow and highlight pairs.
        In SSS physics, anthropogenic debris produces a bright acoustic backscatter highlight
        followed immediately by an acoustic shadow in the far-range direction.
        """
        shadow_mask = (img <= shadow_thresh).astype(np.uint8) * 255
        highlight_mask = (img >= highlight_thresh).astype(np.uint8) * 255

        # Morphological opening to remove isolated single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        shadow_clean = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        highlight_clean = cv2.morphologyEx(highlight_mask, cv2.MORPH_OPEN, kernel)

        num_shadow_labels, shadow_labels, shadow_stats, _ = cv2.connectedComponentsWithStats(shadow_clean)
        num_hl_labels, hl_labels, hl_stats, _ = cv2.connectedComponentsWithStats(highlight_clean)

        valid_highlights = []
        for i in range(1, num_hl_labels):
            area = hl_stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                x, y, w, h = (
                    hl_stats[i, cv2.CC_STAT_LEFT],
                    hl_stats[i, cv2.CC_STAT_TOP],
                    hl_stats[i, cv2.CC_STAT_WIDTH],
                    hl_stats[i, cv2.CC_STAT_HEIGHT]
                )
                valid_highlights.append({
                    "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
                    "area": int(area),
                    "mean_intensity": float(np.mean(img[y:y+h, x:x+w]))
                })

        return {
            "shadow_mask": shadow_clean,
            "highlight_mask": highlight_clean,
            "shadow_pixel_count": int(np.sum(shadow_clean > 0)),
            "highlight_pixel_count": int(np.sum(highlight_clean > 0)),
            "candidate_highlights": valid_highlights
        }

    def tile_mosaic(
        self,
        mosaic: np.ndarray,
        tile_size: Tuple[int, int] = (640, 640),
        overlap: int = 128
    ) -> List[Dict[str, Any]]:
        """
        Slices massive survey mosaics into overlapping tiles for model ingestion.
        Returns tile arrays with exact pixel bounding coordinates (col_start, row_start).
        """
        h, w = mosaic.shape[:2]
        tile_w, tile_h = tile_size
        step_x = max(1, tile_w - overlap)
        step_y = max(1, tile_h - overlap)

        tiles = []
        tile_idx = 0

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                # Ensure tile doesn't exceed image bounds
                x_end = min(x + tile_w, w)
                y_end = min(y + tile_h, h)
                x_start = max(0, x_end - tile_w)
                y_start = max(0, y_end - tile_h)

                tile_patch = mosaic[y_start:y_end, x_start:x_end]
                
                # Check if tile has valid content (not 100% black/white nodata border)
                mean_val = np.mean(tile_patch)
                if 1.0 <= mean_val <= 254.0:
                    tiles.append({
                        "tile_id": f"tile_{tile_idx:05d}",
                        "patch": tile_patch,
                        "bounds": {
                            "col_start": x_start,
                            "row_start": y_start,
                            "col_end": x_end,
                            "row_end": y_end
                        }
                    })
                    tile_idx += 1

        return tiles

    def compute_quality_metrics(self, raw_img: np.ndarray, processed_img: np.ndarray) -> Dict[str, float]:
        """
        Calculates quantitative enhancement metrics:
          - Equivalent Number of Looks (ENL): measures speckle noise reduction
          - Contrast Improvement Ratio (CIR): measures dynamic range expansion
        """
        raw_f = raw_img.astype(np.float32)
        proc_f = processed_img.astype(np.float32)

        # ENL = mean^2 / variance (higher = less speckle)
        raw_mean, raw_var = np.mean(raw_f), np.var(raw_f)
        proc_mean, proc_var = np.mean(proc_f), np.var(proc_f)

        raw_enl = (raw_mean ** 2) / (raw_var + 1e-6)
        proc_enl = (proc_mean ** 2) / (proc_var + 1e-6)
        enl_improvement = (proc_enl / (raw_enl + 1e-6))

        # Standard deviation (contrast) ratio
        raw_std = np.std(raw_f)
        proc_std = np.std(proc_f)
        contrast_gain = proc_std / (raw_std + 1e-6)

        return {
            "raw_enl": round(float(raw_enl), 3),
            "processed_enl": round(float(proc_enl), 3),
            "enl_improvement_factor": round(float(enl_improvement), 3),
            "raw_contrast_std": round(float(raw_std), 2),
            "processed_contrast_std": round(float(proc_std), 2),
            "contrast_gain_factor": round(float(contrast_gain), 3)
        }

    def process(self, image_input: Any) -> Dict[str, Any]:
        """
        Executes end-to-end configured preprocessing pipeline:
          Input -> Grayscale -> Dynamic Range Norm -> Lee Denoise -> CLAHE -> Shadow/Highlight Extraction
        """
        raw_gray, meta = self.load_image_as_grayscale(image_input)
        original_copy = raw_gray.copy()

        # Step 1: Dynamic Range Normalization
        norm_img = self.normalize(raw_gray, method="min_max_uint8")

        # Step 2: Sonar Speckle Denoising (Lee filter)
        lee_size = self.config.get("lee_filter_size", 5)
        lee_var = self.config.get("lee_filter_var", 0.25)
        denoised = self.denoise_speckle_lee(norm_img, size=lee_size, noise_var=lee_var)

        # Step 3: Contrast Enhancement (CLAHE)
        clip_lim = self.config.get("clahe_clip_limit", 2.5)
        grid_sz = tuple(self.config.get("clahe_grid_size", (8, 8)))
        enhanced = self.enhance_contrast_clahe(denoised, clip_limit=clip_lim, grid_size=grid_sz)

        # Step 4: Shadow-Highlight Acoustic Analysis
        sh_thresh = self.config.get("shadow_threshold", 35)
        hl_thresh = self.config.get("highlight_threshold", 195)
        sh_analysis = self.extract_shadow_highlight_pair(
            enhanced,
            shadow_thresh=sh_thresh,
            highlight_thresh=hl_thresh
        )

        # Step 5: Metrics computation
        metrics = self.compute_quality_metrics(original_copy, enhanced)

        return {
            "status": "success",
            "metadata": meta,
            "original_image": original_copy,
            "preprocessed_image": enhanced,
            "shadow_mask": sh_analysis["shadow_mask"],
            "highlight_mask": sh_analysis["highlight_mask"],
            "candidate_highlights": sh_analysis["candidate_highlights"],
            "metrics": metrics
        }
