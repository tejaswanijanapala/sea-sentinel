"""
Stage 5 Unit Test Suite: Anomaly Detection & False-Positive Filtering
Validates:
  1. CNN Autoencoder architecture, latent bottleneck, and reconstruction (Algorithms 1-5)
  2. Difference map and Mean Squared Error calculation (Algorithms 6-7)
  3. Threshold comparator and anomaly classification (Algorithms 8-9)
  4. Acoustic shadow-highlight geometric pairing verifier
  5. Native DBSCAN rock cluster suppression filter
  6. Multi-factor confidence calibrator across all 3 tiers
  7. Backward compatibility with pipeline orchestrator
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.anomaly_detection.models import AcousticAutoencoder, CNNEncoder, CNNDecoder
from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.anomaly_detection.shadow_verifier import AcousticShadowVerifier
from ai.anomaly_detection.rock_cluster_filter import DBSCANRockFilter
from ai.anomaly_detection.calibrator import ConfidenceCalibrator


def test_autoencoder_architecture():
    """Test CNN Encoder, Latent Bottleneck, and CNN Decoder shapes."""
    model = AcousticAutoencoder(in_channels=1, latent_dim=64, base_channels=16)
    x = torch.randn(2, 1, 128, 128)

    # Encode (Algorithm 2 & 3)
    z = model.encode(x)
    assert z.shape == (2, 64), f"Expected (2, 64), got {z.shape}"

    # Decode (Algorithm 4 & 5)
    x_recon = model.decode(z)
    assert x_recon.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {x_recon.shape}"
    assert x_recon.min() >= 0.0 and x_recon.max() <= 1.0, "Decoder output must be bounded in [0, 1]"
    assert model.count_parameters() > 0


def test_algorithms_1_to_9_pipeline():
    """Test Algorithms 1 through 9 step-by-step."""
    detector = AnomalyDetector(threshold=0.03)
    test_patch = np.random.randint(40, 200, (100, 100), dtype=np.uint8)

    # Algorithm 1: Prepare patch
    tensor, norm = detector.prepare_patch(test_patch)
    assert tensor.shape == (1, 1, 128, 128)
    assert norm.shape == (128, 128)

    # Algorithms 2-5: Reconstruct patch
    orig, recon, mse = detector.reconstruct_patch(test_patch)
    assert orig.shape == (128, 128)
    assert recon.shape == (128, 128)

    # Algorithm 6: Difference Map
    diff_map = detector.compute_difference_map(orig, recon)
    assert diff_map.shape == (128, 128)
    assert np.all(diff_map >= 0.0)

    # Algorithm 7: MSE Error
    computed_mse = detector.compute_reconstruction_error(orig, recon)
    assert abs(computed_mse - mse) < 1e-5

    # Algorithms 8 & 9: Threshold Comparator & Anomaly Classifier
    is_anomaly_low, class_low = detector.classify_anomaly(0.01)
    assert is_anomaly_low is False
    assert class_low == "NORMAL_SEAFLOOR"

    is_anomaly_high, class_high = detector.classify_anomaly(0.08)
    assert is_anomaly_high is True
    assert class_high == "ANOMALY_DEBRIS"


def test_acoustic_shadow_verifier():
    """Test highlight-shadow geometric pairing physics."""
    verifier = AcousticShadowVerifier(min_contrast_ratio=1.5)

    # 1. Valid highlight + shadow pair
    patch = np.ones((100, 100), dtype=np.float32) * 0.4  # Ambient seabed
    patch[30:50, 30:50] = 0.95  # Strong backscatter highlight
    patch[30:50, 55:75] = 0.02  # Trailing acoustic shadow
    res = verifier.verify_patch(patch)
    assert res["shadow_verified"] is True
    assert res["contrast_ratio"] > 2.0
    assert res["pairing_score"] > 0.6

    # 2. Flat ambient noise (no highlight or shadow)
    flat_patch = np.ones((100, 100), dtype=np.float32) * 0.4
    res_flat = verifier.verify_patch(flat_patch)
    assert res_flat["shadow_verified"] is False
    assert res_flat["pairing_score"] < 0.4


def test_dbscan_rock_filter():
    """Test geological rock cluster suppression."""
    rock_filter = DBSCANRockFilter(eps=50.0, min_samples=3, max_rock_area_px=600.0)

    # Clustered rock field (4 detections clustered closely together)
    rock_detections = [
        {"object_id": "ROCK_1", "bbox": {"x1": 100, "y1": 100, "x2": 115, "y2": 115}},
        {"object_id": "ROCK_2", "bbox": {"x1": 110, "y1": 105, "x2": 125, "y2": 120}},
        {"object_id": "ROCK_3", "bbox": {"x1": 105, "y1": 115, "x2": 120, "y2": 130}},
        {"object_id": "ROCK_4", "bbox": {"x1": 120, "y1": 110, "x2": 135, "y2": 125}},
        # Isolated man-made debris far away
        {"object_id": "DEBRIS_ISOLATED", "bbox": {"x1": 500, "y1": 500, "x2": 580, "y2": 560}}
    ]

    filtered = rock_filter.filter_detections(rock_detections)
    assert len(filtered) == 5

    # Rocks should be flagged as rock cluster with density penalty
    for det in filtered[:4]:
        assert det["is_rock_cluster"] is True
        assert det["rock_density_penalty"] > 0.0

    # Isolated debris should NOT be a rock cluster
    isolated = filtered[4]
    assert isolated["is_rock_cluster"] is False
    assert isolated["rock_density_penalty"] == 0.0


def test_confidence_calibrator():
    """Test multi-factor confidence scoring across all three operational tiers."""
    calibrator = ConfidenceCalibrator(confirmed_threshold=0.75, suspicious_threshold=0.40)

    # Tier 1: Confirmed Debris (>= 75%)
    res1 = calibrator.calibrate(raw_confidence=0.85, anomaly_score=0.90, shadow_score=0.80, rock_penalty=0.0)
    assert res1["status"] == "confirmed_debris"
    assert res1["calibrated_confidence"] >= 0.75
    assert res1["is_confirmed"] is True

    # Tier 2: Suspicious Anomaly (40% - 74%)
    res2 = calibrator.calibrate(raw_confidence=0.60, anomaly_score=0.50, shadow_score=0.40, rock_penalty=0.0)
    assert res2["status"] == "suspicious_anomaly"
    assert 0.40 <= res2["calibrated_confidence"] < 0.75

    # Tier 3: Noise Rejected (< 40%) - e.g. Rock cluster penalty down-weighting
    res3 = calibrator.calibrate(raw_confidence=0.55, anomaly_score=0.30, shadow_score=0.20, rock_penalty=0.50)
    assert res3["status"] == "noise_rejected"
    assert res3["calibrated_confidence"] < 0.40
    assert res3["is_rejected"] is True


def test_anomaly_detector_contract():
    """Test AnomalyDetector API backward compatibility with orchestrator."""
    detector = AnomalyDetector()

    # Simple detection without image context
    det = {"object_id": "OBJ_01", "confidence": 0.85}
    res = detector.evaluate_detection(det)

    assert "object_id" in res
    assert "raw_confidence" in res
    assert "calibrated_confidence" in res
    assert "status" in res
    assert "is_anomaly" in res
    assert "reconstruction_error" in res
    assert "shadow_verified" in res
    assert res["status"] in ["confirmed_debris", "suspicious_anomaly", "noise_rejected"]

    # Detection with mock image context
    mock_image = np.ones((300, 300), dtype=np.uint8) * 100
    mock_image[50:80, 50:70] = 230
    mock_image[50:80, 72:90] = 10
    det_with_box = {"object_id": "OBJ_02", "confidence": 0.80, "bbox": {"x1": 45, "y1": 45, "x2": 95, "y2": 85}}
    res_ctx = detector.evaluate_detection(det_with_box, image_context=mock_image)
    assert res_ctx["reconstruction_error"] >= 0.0
    assert isinstance(res_ctx["shadow_verified"], bool)


if __name__ == "__main__":
    print("Running Stage 5 Unit Tests...")
    test_autoencoder_architecture()
    print("  [PASSED] test_autoencoder_architecture")
    test_algorithms_1_to_9_pipeline()
    print("  [PASSED] test_algorithms_1_to_9_pipeline")
    test_acoustic_shadow_verifier()
    print("  [PASSED] test_acoustic_shadow_verifier")
    test_dbscan_rock_filter()
    print("  [PASSED] test_dbscan_rock_filter")
    test_confidence_calibrator()
    print("  [PASSED] test_confidence_calibrator")
    test_anomaly_detector_contract()
    print("  [PASSED] test_anomaly_detector_contract")
    print("All Stage 5 unit tests executed successfully!")
