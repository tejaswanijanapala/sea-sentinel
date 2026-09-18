"""
Unit Tests for YOLO + U-Net Fusion Engine and Candidate Verifier
Validates:
  1. Multi-signal association into BOTH, YOLO_ONLY, and UNET_ONLY
  2. Bounding box and centroid fusion
  3. Recovery of U-Net only candidates (missed by YOLO)
  4. Candidate verification scoring and physics-grounded quality checks
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from inference.fusion_engine import FusionEngine
from inference.verifier import CandidateVerifier
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def test_fusion_categorization():
    fusion = FusionEngine(iou_threshold=0.25)

    # 3 YOLO candidates
    yolo_cands = [
        {"object_id": "Y1", "bbox": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}, "centroid": [100, 100], "confidence": 0.85, "class": "fishing_net"},
        {"object_id": "Y2", "bbox": {"x1": 300, "y1": 200, "x2": 450, "y2": 260}, "centroid": [375, 230], "confidence": 0.78, "class": "pipeline_or_cable"}
    ]

    # 2 U-Net candidates (one matches Y1, one is unique to U-Net)
    unet_cands = [
        {"object_id": "U1", "bbox": {"x1": 45, "y1": 48, "x2": 155, "y2": 148}, "centroid": [100, 98], "confidence": 0.88, "class": "fishing_net", "mask_area": 9800},
        {"object_id": "U2", "bbox": {"x1": 500, "y1": 350, "x2": 600, "y2": 450}, "centroid": [550, 400], "confidence": 0.80, "class": "shipwreck_fragment", "mask_area": 8500}
    ]

    out = fusion.fuse(yolo_cands, unet_cands, (640, 640))
    assert out["status"] == "success"
    assert out["total_candidates"] == 3
    assert out["confirmed_both"] == 1
    assert out["yolo_only"] == 1
    assert out["unet_only"] == 1

    objects = out["objects"]
    both_obj = next(o for o in objects if o["source_category"] == "BOTH")
    assert both_obj["agreement"] is True
    assert "yolo" in both_obj["sources"] and "unet" in both_obj["sources"]
    assert both_obj["confidence"] > 0.85 # Agreement boost

    unet_only = next(o for o in objects if o["source_category"] == "UNET_ONLY")
    assert unet_only["agreement"] is False
    assert unet_only["class"] == "shipwreck_fragment"


def test_candidate_verifier():
    verifier = CandidateVerifier()
    # Create image with bright highlight on dark background
    img = np.full((300, 300), 80, dtype=np.uint8)
    img[50:100, 50:100] = 230 # Highlight
    img[101:140, 50:100] = 20 # Trailing shadow

    candidates = [
        {
            "object_id": "T1",
            "bbox": {"x1": 50, "y1": 50, "x2": 100, "y2": 100},
            "source_category": "BOTH",
            "confidence": 0.88,
            "aspect_ratio": 1.0,
            "compactness": 0.8,
            "solidity": 0.9
        }
    ]

    verified = verifier.verify_candidates(candidates, img)
    assert len(verified) == 1
    assert verified[0]["verification_status"] == "confirmed"
    assert verified[0]["verification_score"] >= 0.65
    assert "quality_metrics" in verified[0]
    assert verified[0]["quality_metrics"]["contrast_score"] > 0.5


if __name__ == "__main__":
    test_fusion_categorization()
    test_candidate_verifier()
    logger.info("All Fusion and Verifier unit tests passed successfully!")
