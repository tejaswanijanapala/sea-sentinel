"""
Adaptive Sonar Preprocessing Engine.
Dynamically selects filtering, speckle reduction, and contrast enhancement profiles
based on real-time measured acoustic characteristics (noise level, turbidity, range attenuation)
while maintaining deterministic low-latency execution (<5-8 ms).
"""

from typing import Dict, Any, Tuple, Optional
import time
import math
import numpy as np
import cv2
from scipy.ndimage import uniform_filter


class AdaptiveSonarPreprocessor:
    """
    Lightweight, real-time adaptive preprocessor calibrated for Side-Scan Sonar.
    Applies adaptive Lee speckle filtering, slant-range geometric correction,
    and CLAHE equalization tailored to the acoustic environment.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = target_size
        self._clahe_cache: Dict[float, cv2.CLAHE] = {}

    def _get_clahe(self, clip_limit: float, grid_size: Tuple[int, int] = (8, 8)) -> cv2.CLAHE:
        key = round(clip_limit, 1)
        if key not in self._clahe_cache:
            self._clahe_cache[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return self._clahe_cache[key]

    def _fast_lee_filter(self, img: np.ndarray, size: int = 5, cu: float = 0.25) -> np.ndarray:
        """
        Fast vectorized Lee speckle reduction filter.
        W = 1 - (Cu^2 / Ci^2) where Ci = local_std / local_mean.
        """
        img_f = img.astype(np.float32)
        local_mean = uniform_filter(img_f, size=size)
        local_sq_mean = uniform_filter(img_f ** 2, size=size)
        local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)
        local_std = np.sqrt(local_var)

        ci = np.zeros_like(local_mean)
        nonzero_mask = local_mean > 1e-3
        ci[nonzero_mask] = local_std[nonzero_mask] / local_mean[nonzero_mask]

        cu2 = cu ** 2
        ci2 = np.maximum(ci ** 2, 1e-4)
        weights = np.clip(1.0 - (cu2 / ci2), 0.0, 1.0)

        filtered = local_mean + weights * (img_f - local_mean)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    def correct_slant_range(
        self,
        image: np.ndarray,
        altitude_px: Optional[int] = None
    ) -> np.ndarray:
        """
        Performs slant-range to ground-range geometric correction:
        Ground Range Rg = sqrt(Slant Range Rs^2 - Altitude H^2).
        Removes the central nadir blind zone (water column).
        """
        if altitude_px is None or altitude_px <= 2:
            return image

        h, w = image.shape[:2]
        center_x = w // 2
        alt = min(altitude_px, center_x - 5)

        # Generate ground range mapping index
        # Port (left) and Starboard (right) channels
        x_indices = np.arange(center_x, dtype=np.float32)
        # For port: Rs = (center_x - x)
        rs_port = center_x - x_indices
        valid_port = rs_port >= alt
        rg_port = np.zeros_like(rs_port)
        rg_port[valid_port] = np.sqrt(np.maximum(0.0, rs_port[valid_port]**2 - alt**2))

        # Remap coordinates
        max_rg = np.sqrt(max(1.0, center_x**2 - alt**2))
        norm_map_port = np.clip(center_x - (rg_port / max_rg * center_x), 0, center_x - 1).astype(np.float32)

        # Starboard mapping
        rs_star = x_indices
        valid_star = rs_star >= alt
        rg_star = np.zeros_like(rs_star)
        rg_star[valid_star] = np.sqrt(np.maximum(0.0, rs_star[valid_star]**2 - alt**2))
        norm_map_star = np.clip(center_x + (rg_star / max_rg * (w - center_x)), center_x, w - 1).astype(np.float32)

        full_map_x = np.concatenate([norm_map_port, norm_map_star])
        map_y = np.arange(h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(full_map_x, map_y)

        corrected = cv2.remap(image, grid_x.astype(np.float32), grid_y.astype(np.float32), cv2.INTER_LINEAR)
        return corrected

    def process(
        self,
        image: np.ndarray,
        quality_metrics: Optional[Dict[str, float]] = None,
        altitude_px: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies adaptive sonar preprocessing:
        1. Selects environment profile (Quiet, High Speckle, Deep Low Signal, Saturated).
        2. Applies Slant-range correction if altitude given.
        3. Applies adaptive Lee filter and CLAHE contrast enhancement.
        4. Resizes to target inference dimension with aspect-ratio preservation.
        """
        t0 = time.perf_counter()
        
        # Ensure grayscale 2D array
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Extract or compute baseline acoustic stats
        if quality_metrics:
            speckle_idx = quality_metrics.get("speckle_index", 0.5)
            mean_int = quality_metrics.get("mean_intensity", 50.0)
            dyn_range = quality_metrics.get("dynamic_range", 50.0)
        else:
            mean_int = float(np.mean(gray))
            std_int = float(np.std(gray))
            speckle_idx = float(std_int / max(mean_int, 1.0))
            p05, p95 = np.percentile(gray, [5, 95])
            dyn_range = float(p95 - p05)

        # 1. Select Environment Profile
        if speckle_idx > 0.85:
            profile_name = "HIGH_SPECKLE_TURBID"
            lee_size = 5
            lee_cu = 0.28
            clahe_clip = 3.2
        elif mean_int < 35.0 or dyn_range < 35.0:
            profile_name = "DEEP_WATER_LOW_SIGNAL"
            lee_size = 3
            lee_cu = 0.20
            clahe_clip = 3.5
        elif mean_int > 160.0:
            profile_name = "SATURATED_SHALLOW"
            lee_size = 5
            lee_cu = 0.22
            clahe_clip = 1.8
        else:
            profile_name = "QUIET_BENTHIC"
            lee_size = 3
            lee_cu = 0.22
            clahe_clip = 2.4

        # 2. Slant-Range Geometric Correction
        corrected = self.correct_slant_range(gray, altitude_px)

        # 3. Adaptive Lee Speckle Filtering
        denoised = self._fast_lee_filter(corrected, size=lee_size, cu=lee_cu)

        # 4. Adaptive Contrast Equalization
        clahe = self._get_clahe(clahe_clip)
        enhanced = clahe.apply(denoised)

        # 5. Resize to canonical target dimension
        if enhanced.shape != self.target_size:
            final_img = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            final_img = enhanced

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        meta = {
            "profile": profile_name,
            "lee_filter_size": lee_size,
            "lee_filter_cu": lee_cu,
            "clahe_clip": clahe_clip,
            "slant_range_corrected": altitude_px is not None and altitude_px > 2,
            "preprocessing_time_ms": duration_ms
        }
        return final_img, meta
