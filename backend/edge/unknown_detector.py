"""
Third Perception Branch: Unknown Object & Acoustic Anomaly Detection Engine.
Prevents forcing uncataloged marine debris or unusual anthropogenic hazards
into nearest known training classes. Dispatches uncataloged objects to Human Review.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import cv2


class UnknownObjectDetector:
    """
    Evaluates candidate detections to separate:
      1. KNOWN_OBJECT: High confidence match with verified geometric profile.
      2. UNKNOWN_OBJECT: Significant physical relief/segmentation, but novel spectral/spatial signature.
      3. NATURAL_BENTHIC_FORMATION: Rock outcropping, sand ripple, or coral structure.
      4. FALSE_POSITIVE: Transient speckle noise, bubble wake, or sonar flare.
    """
    def __init__(
        self,
        anomaly_threshold: float = 0.65,
        min_solidity_manmade: float = 0.55,
        min_shadow_relief_ratio: float = 0.20
    ):
        self.anomaly_threshold = anomaly_threshold
        self.min_solidity_manmade = min_solidity_manmade
        self.min_shadow_relief_ratio = min_shadow_relief_ratio

    def evaluate_candidate(
        self,
        patch: np.ndarray,
        yolo_conf: float,
        unet_score: float,
        class_name: str,
        mask_patch: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyzes an individual candidate region using multi-modal acoustic cues.
        """
        if patch is None or patch.size < 16:
            return {
                "category": "FALSE_POSITIVE",
                "is_unknown": False,
                "anomaly_score": 1.0,
                "relief_score": 0.0,
                "solidity": 0.0,
                "recommendation": "DISCARD"
            }

        if patch.ndim == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch

        h, w = gray.shape
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # 1. Acoustic Highlight vs Shadow Relief Analysis
        # Highlight: high intensity reflections facing sonar head
        # Shadow: zero-intensity acoustic block behind high-standing object
        highlight_mask = gray > (mean_val + 0.8 * std_val)
        shadow_mask = gray < max(15.0, mean_val - 0.9 * std_val)

        highlight_ratio = float(highlight_mask.sum()) / float(gray.size)
        shadow_ratio = float(shadow_mask.sum()) / float(gray.size)

        # Presence of both highlight and shadow confirms an elevated 3D object on seabed
        has_paired_relief = (highlight_ratio > 0.04) and (shadow_ratio > 0.06)
        relief_score = min(1.0, (highlight_ratio + 1.5 * shadow_ratio) * 2.5)

        # 2. Contour Geometry & Morphological Solidity
        # Natural rocks have low solidity and irregular fractal borders;
        # Manmade debris (containers, tires, pipes, nets) have higher compactness or defined patterns
        if mask_patch is not None and mask_patch.any():
            bin_mask = (mask_patch > 0.5).astype(np.uint8)
        else:
            bin_mask = highlight_mask.astype(np.uint8)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.5
        aspect_ratio = 1.0
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area / hull_area)
            _, _, cw, ch = cv2.boundingRect(c)
            aspect_ratio = float(max(cw, ch)) / float(max(1, min(cw, ch)))

        # 3. Anomaly Score Synthesis
        # Discrepancy between strong physical presence and low class certainty indicates an anomaly
        class_uncertainty = 1.0 - yolo_conf
        morphological_unorthodoxy = 1.0 - abs(solidity - 0.7)  # deviate from typical training clusters
        
        # High unet confidence + low yolo confidence is a prime unknown signature
        perceptual_divergence = max(0.0, unet_score - yolo_conf)

        anomaly_score = (
            0.40 * class_uncertainty +
            0.35 * perceptual_divergence +
            0.25 * (1.0 - min(1.0, yolo_conf / max(0.01, unet_score)))
        )
        anomaly_score = float(np.clip(anomaly_score, 0.0, 1.0))

        # 4. Final Classification Decision
        if yolo_conf >= 0.65 and unet_score >= 0.55:
            category = "KNOWN_OBJECT"
            is_unknown = False
            rec = "AUTOMATIC_RECORD"
        elif unet_score >= 0.40 and (anomaly_score >= self.anomaly_threshold or has_paired_relief):
            # Strong segmentation + clear relief, but unknown class
            category = "UNKNOWN_OBJECT"
            is_unknown = True
            rec = "SEND_TO_HUMAN_REVIEW"
        elif not has_paired_relief and solidity < 0.40 and relief_score < 0.15:
            # Low relief, irregular contour, diffuse shadow -> natural benthic feature
            category = "NATURAL_BENTHIC_FORMATION"
            is_unknown = False
            rec = "SUPPRESS_OR_FLAG_GEOLOGY"
        elif yolo_conf < 0.30 and unet_score < 0.30:
            category = "FALSE_POSITIVE"
            is_unknown = False
            rec = "DISCARD"
        else:
            category = "LOW_CONFIDENCE_CANDIDATE"
            is_unknown = False
            rec = "FLAG_FOR_REVIEW"

        return {
            "category": category,
            "is_unknown": is_unknown,
            "anomaly_score": round(anomaly_score, 3),
            "relief_score": round(relief_score, 3),
            "solidity": round(solidity, 3),
            "has_paired_relief": has_paired_relief,
            "aspect_ratio": round(aspect_ratio, 2),
            "recommendation": rec
        }
