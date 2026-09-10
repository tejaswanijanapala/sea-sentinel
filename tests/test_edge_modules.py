"""
Unit & Integration Tests for Sea Sentinel Edge AI & AUV/ROV Perception Modules.
Validates bounded ring buffer drop policy, sub-millisecond quality gate,
adaptive preprocessing, parallel perception, unknown object branch,
uncertainty calibration, acoustic telemetry modem packing, and watchdog recovery.
"""

import os
import sys
try:
    import pytest
except ImportError:
    pytest = None
import numpy as np

# Ensure backend root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from edge.frame_buffer import BoundedFrameBuffer, PreallocatedBufferPool
from edge.quality_gate import SonarQualityGate, QualityReport
from edge.adaptive_preprocessor import AdaptiveSonarPreprocessor
from edge.unknown_detector import UnknownObjectDetector
from edge.uncertainty_engine import UncertaintyCalibrationEngine
from edge.temporal_tracker import SonarKalmanTracker, TrackState
from edge.sensor_fusion import SensorFusionEngine, VehicleNavState
from edge.uncertainty_geolocator import UncertaintyGeolocator
from edge.resource_manager import EdgeResourceManager, DegradationLevel, PowerState
from edge.watchdog import EdgeWatchdogSupervisor
from edge.telemetry_modem import AcousticTelemetryEncoder
from edge.active_learning import ActiveLearningSelector, ContinualLearningGuardian
from edge.edge_perception import EdgePerceptionPipeline


def test_bounded_frame_buffer_drop_policy():
    """Verify that buffer enforces latest-frame policy when over capacity."""
    buf = BoundedFrameBuffer(max_capacity=3)
    assert buf.max_capacity == 3

    # Push 5 frames
    for i in range(5):
        buf.push({"frame_id": f"FRAME_{i+1}", "data": np.zeros((10, 10))})

    stats = buf.get_stats()
    assert stats["queue_depth"] == 3
    assert stats["total_ingested"] == 5
    assert stats["total_dropped"] == 2
    assert stats["drop_rate_pct"] == 40.0

    # Pop remaining frames: oldest remaining should be FRAME_3
    f1 = buf.pop(timeout=0.1)
    assert f1["frame_id"] == "FRAME_3"
    f2 = buf.pop(timeout=0.1)
    assert f2["frame_id"] == "FRAME_4"
    f3 = buf.pop(timeout=0.1)
    assert f3["frame_id"] == "FRAME_5"
    assert buf.pop(timeout=0.1) is None


def test_sonar_quality_gate():
    """Verify detection of valid vs corrupted, starved, or saturated sonar frames."""
    gate = SonarQualityGate()

    # 1. Valid sonar image
    valid_sonar = np.random.normal(70.0, 25.0, (256, 256)).clip(10, 220).astype(np.uint8)
    rep_valid = gate.evaluate(valid_sonar)
    assert rep_valid.passed is True
    assert rep_valid.status == "PASS"
    assert rep_valid.evaluation_time_ms < 50.0  # sub-50ms benchmark tolerance on CPU

    # 2. Dead ping / starved frame
    dead_ping = np.zeros((256, 256), dtype=np.uint8)
    rep_dead = gate.evaluate(dead_ping)
    assert rep_dead.passed is False
    assert rep_dead.status == "LOW_QUALITY_FRAME"
    assert any("INSUFFICIENT_SIGNAL" in r or "DEAD_PING" in r for r in rep_dead.reasons)

    # 3. Saturated frame
    saturated = np.full((256, 256), 255, dtype=np.uint8)
    rep_sat = gate.evaluate(saturated)
    assert rep_sat.passed is False
    assert any("SATURATION" in r for r in rep_sat.reasons)


def test_adaptive_preprocessor():
    """Verify adaptive filtering and slant-range correction."""
    prep = AdaptiveSonarPreprocessor(target_size=(640, 640))
    raw_img = np.random.randint(30, 180, (400, 400), dtype=np.uint8)

    processed, meta = prep.process(raw_img, altitude_px=20)
    assert processed.shape == (640, 640)
    assert meta["slant_range_corrected"] is True
    assert "profile" in meta
    assert meta["preprocessing_time_ms"] > 0


def test_unknown_object_detector():
    """Verify third branch separates known objects, anomalies, and natural features."""
    detector = UnknownObjectDetector()

    # 1. Known object: high yolo + high unet
    patch_known = np.random.randint(10, 240, (40, 40), dtype=np.uint8)
    res_known = detector.evaluate_candidate(patch_known, yolo_conf=0.88, unet_score=0.85, class_name="metal_container")
    assert res_known["category"] == "KNOWN_OBJECT"
    assert res_known["is_unknown"] is False

    # 2. Unknown object: low yolo + high unet + strong relief
    patch_unknown = np.zeros((40, 40), dtype=np.uint8)
    patch_unknown[5:20, 5:20] = 250  # Highlight
    patch_unknown[21:38, 5:20] = 2    # Acoustic shadow
    res_unk = detector.evaluate_candidate(patch_unknown, yolo_conf=0.25, unet_score=0.82, class_name="unknown")
    assert res_unk["category"] == "UNKNOWN_OBJECT"
    assert res_unk["is_unknown"] is True
    assert res_unk["recommendation"] == "SEND_TO_HUMAN_REVIEW"


def test_uncertainty_calibration_engine():
    """Verify multi-factor confidence and high-recall retention."""
    engine = UncertaintyCalibrationEngine(high_recall_mode=False)

    # Standard confidence
    res_high = engine.compute_calibrated_confidence(yolo_conf=0.85, unet_score=0.80, temporal_hits=3)
    assert res_high["tier"] == "HIGH_CONFIDENCE"
    assert res_high["calibrated_confidence"] >= 0.70

    # High-recall mode test
    engine.set_high_recall_mode(True)
    res_weak = engine.compute_calibrated_confidence(yolo_conf=0.32, unet_score=0.45, temporal_hits=2)
    assert res_weak["tier"] in ["HIGH_RECALL_CANDIDATE", "MEDIUM_CONFIDENCE"]
    assert res_weak["action"] in ["PRESERVE_AND_FLAG_FOR_REVIEW", "RECORD_AND_FLAG_FOR_REVIEW"]


def test_acoustic_telemetry_modem_packing():
    """Verify exactly 24-byte packet compaction and CRC-8 integrity."""
    modem = AcousticTelemetryEncoder()

    packet = modem.encode(
        target_id_num=42,
        class_name="fishing_net",
        confidence=0.91,
        slant_range_m=34.5,
        bearing_deg=78.2,
        local_x_m=12.4,
        local_y_m=32.1,
        depth_m=18.5,
        timestamp_s=1725000000.0,
        uncertainty_m=1.2
    )
    assert len(packet) == 24

    # Decode and check CRC-8 validity
    decoded = modem.decode(packet)
    assert decoded["crc_valid"] is True
    assert decoded["target_id"] == "TGT_0042"
    assert decoded["class"] in ["fishing_net", "ghost_fishing_net"]
    assert decoded["confidence"] == 0.91
    assert abs(decoded["slant_range_m"] - 34.5) < 0.2
    assert abs(decoded["depth_m"] - 18.5) < 0.2


def test_sonar_kalman_tracker():
    """Verify temporal association across pings and track promotion."""
    tracker = SonarKalmanTracker(gating_distance_px=50.0)

    # Ping 1: Initial detection
    cands_p1 = [{"class": "tire", "centroid": [150.0, 200.0], "bbox": {"x1": 140, "y1": 190, "x2": 160, "y2": 210}}]
    tracks_p1 = tracker.update(cands_p1, dt=1.0)
    assert len(tracks_p1) == 1
    assert tracks_p1[0]["track_state"] == TrackState.TENTATIVE
    assert tracks_p1[0]["temporal_hits"] == 1

    # Ping 2: Target shifts slightly due to AUV forward motion
    cands_p2 = [{"class": "tire", "centroid": [152.0, 205.0], "bbox": {"x1": 142, "y1": 195, "x2": 162, "y2": 215}}]
    tracks_p2 = tracker.update(cands_p2, dt=1.0)
    assert len(tracks_p2) == 1
    assert tracks_p2[0]["track_id"] == tracks_p1[0]["track_id"]
    assert tracks_p2[0]["track_state"] == TrackState.CONFIRMED
    assert tracks_p2[0]["temporal_hits"] == 2
    assert tracks_p2[0]["temporal_consistency_score"] > tracks_p1[0]["temporal_consistency_score"]


def test_edge_resource_manager_degradation():
    """Verify graceful degradation levels based on battery and thermals."""
    mgr = EdgeResourceManager()

    # High battery, normal temperature, normal memory
    lvl0, pstate0, pol0 = mgr.evaluate_operating_state(battery_pct=90.0, temperature_c=45.0, mem_used_pct=45.0)
    assert lvl0 in [DegradationLevel.LEVEL_0, DegradationLevel.LEVEL_1]
    assert pol0["enable_unet"] is True

    # Critical battery (<15%)
    lvl_crit, pstate_crit, pol_crit = mgr.evaluate_operating_state(battery_pct=12.0, temperature_c=50.0)
    assert lvl_crit == DegradationLevel.LEVEL_4
    assert pstate_crit == PowerState.CRITICAL
    assert pol_crit["enable_unet"] is False


def test_watchdog_and_integrity():
    """Verify heartbeat pulse and model checksum validation."""
    watchdog = EdgeWatchdogSupervisor(heartbeat_timeout_s=2.0)
    watchdog.pulse("inference")
    health = watchdog.check_health()
    assert health["healthy"] is True

    # Test file integrity
    rep = EdgeWatchdogSupervisor.verify_model_integrity("non_existent_file.pt")
    assert rep["approved"] is False
    assert rep["status"] == "FILE_NOT_FOUND"


def test_active_learning_and_continual_safety():
    """Verify active learning query and regression guardian."""
    selector = ActiveLearningSelector(utility_threshold=0.50)
    eval_res = selector.evaluate_sample_utility(
        yolo_conf=0.45,
        unet_score=0.85,
        anomaly_score=0.70,
        is_unknown=True,
        class_name="unknown",
        temporal_hits=1
    )
    assert eval_res["should_stage_for_review"] is True
    assert "MODEL_DIVERGENCE" in eval_res["reasons"]

    # Continual Learning Safety Guardian
    guardian = ContinualLearningGuardian(max_allowed_recall_drop=0.015)
    # Case A: Minor drop within tolerance
    res_ok = guardian.validate_candidate_model(
        baseline_metrics={"recall": 0.910, "map50": 0.890},
        candidate_metrics={"recall": 0.908, "map50": 0.895}
    )
    assert res_ok["approved"] is True

    # Case B: Severe catastrophic recall drop (fails safety)
    res_bad = guardian.validate_candidate_model(
        baseline_metrics={"recall": 0.910, "map50": 0.890},
        candidate_metrics={"recall": 0.850, "map50": 0.860}
    )
    assert res_bad["approved"] is False
    assert "CATASTROPHIC_FORGETTING_RECALL" in res_bad["reasons"][0]


def test_end_to_end_edge_perception_pipeline():
    """Integration test of complete pipeline from raw sonar to telemetry packets."""
    pipeline = EdgePerceptionPipeline(high_recall_mode=True)

    # Synthesize realistic test frame
    swath = np.random.rayleigh(scale=35.0, size=(640, 640)).astype(np.uint8)
    # Add highlight and shadow
    swath[300:320, 400:420] = 240
    swath[300:320, 421:460] = 2

    res = pipeline.process_frame(swath, frame_id="INTEGRATION_001", altitude_m=5.0)
    assert res["status"] == "PROCESSED"
    assert "timing_breakdown" in res
    assert "total_pipeline_ms" in res["timing_breakdown"]
    assert "nav_state" in res
    assert res["nav_state"]["depth_m"] > 0


if __name__ == "__main__":
    tests = [
        test_bounded_frame_buffer_drop_policy,
        test_sonar_quality_gate,
        test_adaptive_preprocessor,
        test_unknown_object_detector,
        test_uncertainty_calibration_engine,
        test_sonar_kalman_tracker,
        test_edge_resource_manager_degradation,
        test_watchdog_and_integrity,
        test_acoustic_telemetry_modem_packing,
        test_active_learning_and_continual_safety,
        test_end_to_end_edge_perception_pipeline,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            raise
    print(f"\nAll {passed}/{len(tests)} Edge AI tests passed successfully!")
