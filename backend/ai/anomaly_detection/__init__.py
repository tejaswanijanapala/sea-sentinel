"""
Stage 5: Anomaly Detection and False Positive Filtering Package
Exports:
  - AcousticAutoencoder, CNNEncoder, CNNDecoder
  - AnomalyDetector (Algorithms 1-9)
  - AcousticShadowVerifier (Acoustic backscatter highlight-shadow pairing)
  - DBSCANRockFilter (Geological rock field suppression)
  - ConfidenceCalibrator (Multi-factor confidence scoring)
"""

from ai.anomaly_detection.models import AcousticAutoencoder, CNNEncoder, CNNDecoder
from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.anomaly_detection.shadow_verifier import AcousticShadowVerifier
from ai.anomaly_detection.rock_cluster_filter import DBSCANRockFilter
from ai.anomaly_detection.calibrator import ConfidenceCalibrator

__all__ = [
    "AcousticAutoencoder",
    "CNNEncoder",
    "CNNDecoder",
    "AnomalyDetector",
    "AcousticShadowVerifier",
    "DBSCANRockFilter",
    "ConfidenceCalibrator"
]
