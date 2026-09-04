"""
Stage 5: Acoustic Shadow-Highlight Geometric Pairing Verifier
Validates candidate debris detections against underwater acoustic propagation physics.
In Side-Scan Sonar (SSS):
  1. Highlight: Acoustic backscatter reflection peak facing the sonar transducer.
  2. Shadow: Occluded acoustic void trailing down-range away from the nadir line.
Natural sediment ripples, sensor speckle, and gain spikes lack trailing acoustic shadows.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class AcousticShadowVerifier:
    """
    Evaluates geometric consistency between backscatter highlight and acoustic shadow trailing.
    """
    def __init__(
        self,
        highlight_percentile: float = 85.0,
        shadow_percentile: float = 15.0,
        min_contrast_ratio: float = 1.8,
        search_distance_ratio: float = 1.5
    ):
        self.highlight_percentile = highlight_percentile
        self.shadow_percentile = shadow_percentile
        self.min_contrast_ratio = min_contrast_ratio
        self.search_distance_ratio = search_distance_ratio

    def verify_patch(
        self,
        patch: np.ndarray,
        range_direction: str = "horizontal"
    ) -> Dict[str, Any]:
        """
        Analyzes a single cropped candidate ROI for shadow-highlight acoustic pairing.
        Args:
            patch: 2D grayscale sonar image patch (float [0, 1] or uint8 [0, 255])
            range_direction: "horizontal" (swath sweeps left-to-right or right-to-left) or "vertical"
        """
        if patch is None or patch.size == 0:
            return {
                "shadow_verified": False,
                "pairing_score": 0.0,
                "contrast_ratio": 1.0,
                "highlight_mean": 0.0,
                "shadow_mean": 0.0,
                "message": "Empty patch"
            }

        # Normalize to float [0, 1]
        if patch.dtype == np.uint8:
            norm_patch = patch.astype(np.float32) / 255.0
        else:
            norm_patch = np.clip(patch.astype(np.float32), 0.0, 1.0)

        h, w = norm_patch.shape[:2]
        ambient_mean = float(np.median(norm_patch))

        # Highlight threshold (top percentile)
        high_thresh = np.percentile(norm_patch, self.highlight_percentile)
        shadow_thresh = np.percentile(norm_patch, self.shadow_percentile)

        highlight_mask = norm_patch >= max(high_thresh, ambient_mean + 0.12)
        shadow_mask = norm_patch <= min(shadow_thresh, ambient_mean - 0.08)

        highlight_pixels = int(np.sum(highlight_mask))
        shadow_pixels = int(np.sum(shadow_mask))

        if highlight_pixels == 0:
            return {
                "shadow_verified": False,
                "pairing_score": 0.0,
                "contrast_ratio": 1.0,
                "highlight_mean": float(high_thresh),
                "shadow_mean": float(shadow_thresh),
                "message": "No significant acoustic highlight detected."
            }

        highlight_mean = float(np.mean(norm_patch[highlight_mask]))
        shadow_mean = float(np.mean(norm_patch[shadow_mask])) if shadow_pixels > 0 else ambient_mean

        # Contrast ratio between peak highlight and acoustic shadow
        contrast_ratio = (highlight_mean + 1e-4) / (shadow_mean + 1e-4)

        # Spatial geometric trailing check
        # Compute center of mass of highlight vs shadow
        if highlight_pixels > 0 and shadow_pixels > 0:
            y_indices, x_indices = np.where(highlight_mask)
            hl_center_y, hl_center_x = float(np.mean(y_indices)), float(np.mean(x_indices))

            sy_indices, sx_indices = np.where(shadow_mask)
            sh_center_y, sh_center_x = float(np.mean(sy_indices)), float(np.mean(sx_indices))

            spatial_dist = np.sqrt((sh_center_x - hl_center_x) ** 2 + (sh_center_y - hl_center_y) ** 2)
            has_separation = spatial_dist > 2.0  # Not completely overlapping
        else:
            has_separation = False

        # Scoring
        contrast_score = min(1.0, max(0.0, (contrast_ratio - 1.2) / (self.min_contrast_ratio - 1.2 + 1e-5)))
        presence_score = 0.5 if (highlight_pixels > 8 and shadow_pixels > 8) else 0.2
        separation_score = 0.5 if has_separation else 0.1

        pairing_score = float(np.clip(0.4 * contrast_score + 0.3 * presence_score + 0.3 * separation_score, 0.0, 1.0))
        shadow_verified = (contrast_ratio >= self.min_contrast_ratio) and (shadow_pixels >= 6) and has_separation

        return {
            "shadow_verified": bool(shadow_verified),
            "pairing_score": round(pairing_score, 4),
            "contrast_ratio": round(float(contrast_ratio), 3),
            "highlight_mean": round(highlight_mean, 4),
            "shadow_mean": round(shadow_mean, 4),
            "highlight_pixels": highlight_pixels,
            "shadow_pixels": shadow_pixels,
            "ambient_mean": round(ambient_mean, 4)
        }
