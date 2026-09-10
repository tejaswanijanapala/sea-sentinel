"""
Multi-Factor Calibrated Confidence & Epistemic Uncertainty Engine.
Integrates independent evidence across YOLO, U-Net, temporal tracking, acoustic quality,
and anomaly scoring to produce calibrated confidence and actionable review tiers.
Implements HIGH_RECALL_MODE to strictly minimize catastrophic false negatives.
"""

from typing import Dict, Any, List, Optional
import math
import numpy as np


class UncertaintyCalibrationEngine:
    """
    Computes calibrated confidence and uncertainty estimates for candidate debris.
    Enforces configurable weighting and high-recall optimizations.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        high_recall_mode: bool = False
    ):
        self.high_recall_mode = high_recall_mode
        self.weights = weights or {
            "yolo_conf": 0.28,
            "unet_score": 0.28,
            "temporal_consistency": 0.18,
            "sonar_quality": 0.10,
            "shadow_relief": 0.10,
            "anomaly_penalty": 0.06
        }

    def set_high_recall_mode(self, enabled: bool):
        """Toggles High Recall Mode."""
        self.high_recall_mode = enabled

    def compute_calibrated_confidence(
        self,
        yolo_conf: float,
        unet_score: float,
        temporal_hits: int = 1,
        sonar_quality_score: float = 1.0,
        relief_score: float = 0.5,
        anomaly_score: float = 0.0,
        is_unknown: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates calibrated confidence, uncertainty spread, and assigned review tier.
        """
        # Temporal consistency term (scales logarithmically up to 5 consecutive pings)
        temporal_factor = min(1.0, 0.4 + 0.15 * min(5, temporal_hits))

        w = self.weights
        base_confidence = (
            w["yolo_conf"] * yolo_conf +
            w["unet_score"] * unet_score +
            w["temporal_consistency"] * temporal_factor +
            w["sonar_quality"] * sonar_quality_score +
            w["shadow_relief"] * relief_score -
            w["anomaly_penalty"] * anomaly_score
        )
        base_confidence = float(np.clip(base_confidence, 0.01, 0.99))

        # Epistemic uncertainty: high when models disagree or frame quality is low
        model_disagreement = abs(yolo_conf - unet_score)
        epistemic_uncertainty = (
            0.50 * model_disagreement +
            0.30 * (1.0 - sonar_quality_score) +
            0.20 * anomaly_score
        )
        epistemic_uncertainty = float(np.clip(epistemic_uncertainty, 0.0, 1.0))

        # In High Recall Mode, boost retention of weak candidates that have U-Net or temporal backing
        if self.high_recall_mode:
            if unet_score > 0.35 or temporal_hits > 1:
                # Retention boost: ensures weak acoustic echoes are not thrown out
                base_confidence = min(0.95, base_confidence * 1.18)

        # Classification Tiers
        if is_unknown:
            tier = "UNKNOWN"
            action = "SEND_TO_HUMAN_REVIEW"
        elif base_confidence >= 0.72 and epistemic_uncertainty <= 0.35:
            tier = "HIGH_CONFIDENCE"
            action = "AUTOMATIC_RECORD"
        elif base_confidence >= 0.48:
            tier = "MEDIUM_CONFIDENCE"
            action = "RECORD_AND_FLAG_FOR_REVIEW"
        elif self.high_recall_mode and (base_confidence >= 0.30 or temporal_hits >= 2):
            tier = "HIGH_RECALL_CANDIDATE"
            action = "PRESERVE_AND_FLAG_FOR_REVIEW"
        else:
            tier = "LOW_CONFIDENCE"
            action = "MARK_UNCERTAIN"

        return {
            "calibrated_confidence": round(base_confidence, 3),
            "epistemic_uncertainty": round(epistemic_uncertainty, 3),
            "tier": tier,
            "action": action,
            "high_recall_mode_active": self.high_recall_mode,
            "breakdown": {
                "yolo_contribution": round(w["yolo_conf"] * yolo_conf, 3),
                "unet_contribution": round(w["unet_score"] * unet_score, 3),
                "temporal_factor": round(temporal_factor, 3),
                "sonar_quality_factor": round(sonar_quality_score, 3),
                "relief_contribution": round(w["shadow_relief"] * relief_score, 3),
                "anomaly_penalty": round(w["anomaly_penalty"] * anomaly_score, 3)
            }
        }
