"""
Layer 5: Dimension Estimation Module
Estimates real-world physical dimensions (meters and square meters) only when
valid ground sampling/spatial resolution metadata exists.
Supports:
  1. Axis-aligned bounding box metric dimensions
  2. Rotated bounding box for oriented debris (pipelines, cables, ship hulls)
  3. Pixel-level segmentation mask contour area and perimeter
  4. Aspect ratio, elongation, and circularity morphological descriptors
"""

from typing import Dict, Any, Tuple, Optional, Union
import math
import cv2
import numpy as np


class DimensionEstimator:
    """
    Computes physical metric dimensions (meters) from bounding box, mask, and raster resolution.
    Does NOT fabricate dimensions if resolution metadata is missing.
    """
    def __init__(self, default_res: Optional[Tuple[float, float]] = None):
        self.default_res = default_res

    def estimate_dimensions(
        self,
        bbox: Dict[str, float],
        raster_res: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculates metric length, width, and area from axis-aligned bounding box:
          width_m = (x2 - x1) * x_res
          length_m = (y2 - y1) * y_res
          area_sq_m = width_m * length_m
        """
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))

        width_px = round(abs(x2 - x1), 1)
        length_px = round(abs(y2 - y1), 1)

        res = raster_res or self.default_res
        if not res or res[0] <= 0 or res[1] <= 0:
            return {
                "dimensions_metric": False,
                "length_px": length_px,
                "width_px": width_px,
                "area_px": round(length_px * width_px, 1),
                "length_m": None,
                "width_m": None,
                "area_sq_m": None,
                "aspect_ratio": round(length_px / max(1.0, width_px), 2),
                "message": "Physical dimensions unavailable: scale/resolution metadata missing."
            }

        x_res, y_res = res
        width_m = round(width_px * x_res, 2)
        length_m = round(length_px * y_res, 2)
        area_sq_m = round(width_m * length_m, 2)
        aspect_ratio = round(length_m / max(0.1, width_m), 2)

        return {
            "dimensions_metric": True,
            "length_px": length_px,
            "width_px": width_px,
            "length_m": length_m,
            "width_m": width_m,
            "area_sq_m": area_sq_m,
            "aspect_ratio": aspect_ratio,
            "spatial_resolution_m_per_px": (round(x_res, 3), round(y_res, 3))
        }

    def estimate_mask_dimensions(
        self,
        mask: np.ndarray,
        raster_res: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Extracts exact rotated minimum bounding rectangle and polygon contour metrics:
          - Accounts for oriented objects like diagonal pipelines, cables, and shipwrecks
          - Computes exact mask pixel area and contour perimeter
        """
        if mask is None or mask.size == 0 or np.sum(mask) == 0:
            return {
                "dimensions_metric": False,
                "length_m": None,
                "width_m": None,
                "area_sq_m": None,
                "message": "Empty or invalid segmentation mask."
            }

        # Ensure binary uint8 mask
        bin_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "dimensions_metric": False,
                "length_m": None,
                "width_m": None,
                "area_sq_m": None,
                "message": "No valid contours identified in mask."
            }

        # Largest contour
        largest_cnt = max(contours, key=cv2.contourArea)
        pixel_area = float(cv2.contourArea(largest_cnt))
        pixel_perimeter = float(cv2.arcLength(largest_cnt, closed=True))

        # Rotated minimum bounding rectangle
        rect = cv2.minAreaRect(largest_cnt)  # ((cx, cy), (w, h), angle)
        rect_w, rect_h = rect[1]
        major_axis_px = max(rect_w, rect_h)
        minor_axis_px = min(rect_w, rect_h)
        orientation_deg = round(rect[2], 1)

        res = raster_res or self.default_res
        if not res or res[0] <= 0 or res[1] <= 0:
            return {
                "dimensions_metric": False,
                "length_px": round(major_axis_px, 1),
                "width_px": round(minor_axis_px, 1),
                "pixel_area": pixel_area,
                "orientation_deg": orientation_deg,
                "length_m": None,
                "width_m": None,
                "area_sq_m": None,
                "message": "Physical dimensions unavailable: scale/resolution metadata missing."
            }

        x_res, y_res = res
        avg_res = (x_res + y_res) / 2.0

        length_m = round(major_axis_px * avg_res, 2)
        width_m = round(minor_axis_px * avg_res, 2)
        area_sq_m = round(pixel_area * (x_res * y_res), 2)
        perimeter_m = round(pixel_perimeter * avg_res, 2)
        circularity = round((4.0 * math.pi * pixel_area) / max(1.0, pixel_perimeter ** 2), 3)

        return {
            "dimensions_metric": True,
            "dimension_type": "oriented_contour",
            "length_m": length_m,
            "width_m": width_m,
            "area_sq_m": area_sq_m,
            "perimeter_m": perimeter_m,
            "circularity": circularity,
            "orientation_deg": orientation_deg,
            "pixel_area": int(pixel_area),
            "spatial_resolution_m_per_px": (round(x_res, 3), round(y_res, 3))
        }
