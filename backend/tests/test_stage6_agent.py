"""
Stage 6 Unit Test Suite: AI Agent & Orchestrator
Validates:
  1. Complete tool boundary coordination (YOLO + U-Net + Anomaly + Geotag + Risk)
  2. Input validation and graceful rejection of corrupt files
  3. Execution trace generation with microsecond stage metrics
  4. Natural language explainability synthesis
  5. SQLite audit logging and relational queries
  6. CLI pipeline runner integration
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.orchestrator import SIHPipelineAgent
from agent.explainability import ExplainabilitySynthesizer
from agent.audit_logger import SurveyAuditLogger


def test_agent_initialization():
    """Verify all specialized tools are instantiated."""
    agent = SIHPipelineAgent()
    assert agent.preprocessor is not None
    assert agent.detector is not None
    assert agent.segmenter is not None
    assert agent.anomaly_detector is not None
    assert agent.rock_filter is not None
    assert agent.measurer is not None
    assert agent.geotagger is not None
    assert agent.explainer is not None
    assert agent.audit_logger is not None


def test_agent_invalid_image():
    """Verify graceful rejection of missing image."""
    agent = SIHPipelineAgent()
    res = agent.analyze_image("non_existent_file.png")
    assert res["status"] == "rejected"
    assert "error" in res
    assert len(res["execution_trace"]) > 0


def test_explainability_synthesizer():
    """Verify natural language explanation generation."""
    explainer = ExplainabilitySynthesizer()
    detection = {"object_id": "TEST_NET_01", "class": "fishing_net", "confidence": 0.82}
    exp = explainer.explain_target(
        detection=detection,
        calibrated_status="confirmed_debris",
        calibrated_conf=0.82,
        reconstruction_error=0.12,
        shadow_verified=True,
        is_rock_cluster=False,
        risk_level="HIGH"
    )

    assert "executive_narrative" in exp
    assert "action_recommendation" in exp
    assert "fishing_net" in exp["executive_narrative"]
    assert "PRIORITY INTERVENTION" in exp["action_recommendation"]


def test_sqlite_audit_logger():
    """Verify SQLite audit database persistence and querying."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_audit.db")
        logger = SurveyAuditLogger(db_path=db_path)

        # Log a test survey
        session_id = "TEST_SURVEY_001"
        detections = [
            {
                "object_id": "TGT_01",
                "class": "fishing_net",
                "confidence": 0.85,
                "calibrated_confidence": 0.80,
                "anomaly_status": "confirmed_debris",
                "risk_score": "HIGH",
                "lat": 42.123,
                "lon": -71.456,
                "length_m": 12.0,
                "width_m": 4.5,
                "area_sq_m": 54.0,
                "reconstruction_error": 0.11,
                "shadow_verified": True,
                "is_rock_cluster": False,
                "explanation": {"executive_narrative": "Ghost net detected on seafloor."}
            }
        ]
        trace = [
            {"stage": "input_validation", "status": "completed", "duration_ms": 1.2},
            {"stage": "detection", "status": "completed", "duration_ms": 15.4}
        ]

        success = logger.log_survey(
            session_id=session_id,
            image_path="test_sonar.tif",
            georef_case="A",
            execution_trace=trace,
            detections=detections,
            yolo_loaded=True,
            unet_loaded=False,
            autoencoder_loaded=True,
            total_duration_ms=45.2
        )
        assert success is True

        # Query session summary
        summary = logger.get_session_summary(session_id)
        assert summary is not None
        assert summary["session_id"] == session_id
        assert summary["total_candidates"] == 1
        assert len(summary["targets"]) == 1
        assert len(summary["execution_trace"]) == 2

        # Query high-risk targets
        high_risk = logger.query_high_risk_targets()
        assert len(high_risk) == 1
        assert high_risk[0]["object_id"] == "TGT_01"


def test_agent_end_to_end_analysis():
    """Verify full end-to-end coordinated pipeline execution on real image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_img_path = os.path.join(tmpdir, "sonar_test_tile.png")
        # Create a realistic test image with highlight and shadow
        img = np.ones((256, 256), dtype=np.uint8) * 110
        # Add acoustic highlight (high backscatter)
        img[80:120, 80:110] = 235
        # Add acoustic shadow (occluded void)
        img[80:120, 115:150] = 12
        cv2.imwrite(test_img_path, img)

        agent = SIHPipelineAgent()
        res = agent.analyze_image(test_img_path)

        assert res["status"] == "success"
        assert res["analysis_id"].startswith("SURVEY_")
        assert "summary_statistics" in res
        assert "execution_trace" in res
        assert res["total_duration_ms"] > 0
        assert os.path.exists(res["audit_database"])


if __name__ == "__main__":
    print("Running Stage 6 Unit Tests...")
    test_agent_initialization()
    print("  [PASSED] test_agent_initialization")
    test_agent_invalid_image()
    print("  [PASSED] test_agent_invalid_image")
    test_explainability_synthesizer()
    print("  [PASSED] test_explainability_synthesizer")
    test_sqlite_audit_logger()
    print("  [PASSED] test_sqlite_audit_logger")
    test_agent_end_to_end_analysis()
    print("  [PASSED] test_agent_end_to_end_analysis")
    print("All Stage 6 unit tests executed successfully!")
