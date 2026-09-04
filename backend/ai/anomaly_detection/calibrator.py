"""
Stage 5: Multi-Factor Confidence Calibrator
Calibrates raw detector confidence by combining:
  1. YOLO / Object Detector raw confidence
  2. CNN Autoencoder reconstruction anomaly score (MSE / T)
  3. Acoustic shadow-highlight pairing score
  4. Geological rock cluster density penalty
Classifies detections into three strict operational tiers:
  - Confirmed Debris (>= 75%)
  - Suspicious Anomaly (40% - 74%)
  - Rejected Noise (< 40%)
"""

from typing import Dict, Any


class ConfidenceCalibrator:
    """
    Synthesizes multi-sensor and multi-algorithmic scores into a calibrated confidence metric.
    """
    def __init__(
        self,
        w_detector: float = 0.40,
        w_autoencoder: float = 0.35,
        w_shadow: float = 0.25,
        confirmed_threshold: float = 0.75,
        suspicious_threshold: float = 0.40
    ):
        self.w_detector = w_detector
        self.w_autoencoder = w_autoencoder
        self.w_shadow = w_shadow
        self.confirmed_threshold = confirmed_threshold
        self.suspicious_threshold = suspicious_threshold

    def calibrate(
        self,
        raw_confidence: float,
        anomaly_score: float,
        shadow_score: float,
        rock_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates calibrated confidence score and assigns status.
        Args:
            raw_confidence: Confidence from detector [0, 1]
            anomaly_score: Anomaly score from autoencoder [0, 1] (higher = more anomalous/non-seabed)
            shadow_score: Shadow-highlight pairing score [0, 1]
            rock_penalty: Penalty from DBSCAN rock cluster filter [0, 1]
        """
        base_score = (
            self.w_detector * raw_confidence +
            self.w_autoencoder * anomaly_score +
            self.w_shadow * shadow_score
        )

        calibrated = max(0.0, min(1.0, base_score - rock_penalty))

        if calibrated >= self.confirmed_threshold:
            status = "confirmed_debris"
        elif calibrated >= self.suspicious_threshold:
            status = "suspicious_anomaly"
        else:
            status = "noise_rejected"

        return {
            "raw_confidence": round(float(raw_confidence), 4),
            "calibrated_confidence": round(float(calibrated), 4),
            "anomaly_score": round(float(anomaly_score), 4),
            "shadow_score": round(float(shadow_score), 4),
            "rock_penalty": round(float(rock_penalty), 4),
            "status": status,
            "is_confirmed": (status == "confirmed_debris"),
            "is_rejected": (status == "noise_rejected")
        }
