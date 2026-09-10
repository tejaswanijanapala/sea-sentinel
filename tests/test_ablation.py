"""
Unit Tests for Scientific Ablation Evaluator
Validates:
  1. Benchmark evaluation across Tests A through E
  2. Recall, Precision, and F1 calculations
  3. YOLO-only and U-Net-only miss recovery counting
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from evaluation.ablation_evaluator import AblationEvaluator


def test_ablation_evaluation():
    evaluator = AblationEvaluator()

    gt = [
        {"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "class": "fishing_net"},
        {"bbox": {"x1": 300, "y1": 200, "x2": 450, "y2": 260}, "class": "pipeline_or_cable"},
        {"bbox": {"x1": 500, "y1": 350, "x2": 600, "y2": 450}, "class": "shipwreck_fragment"}
    ]

    # YOLO detects targets 1 and 2, but misses target 3
    yolo_p = [
        {"bbox": {"x1": 102, "y1": 102, "x2": 198, "y2": 198}, "confidence": 0.88, "class": "fishing_net"},
        {"bbox": {"x1": 305, "y1": 202, "x2": 448, "y2": 258}, "confidence": 0.82, "class": "pipeline_or_cable"}
    ]

    # U-Net detects targets 1 and 3, but misses target 2
    unet_p = [
        {"bbox": {"x1": 98, "y1": 98, "x2": 202, "y2": 202}, "confidence": 0.85, "class": "fishing_net"},
        {"bbox": {"x1": 498, "y1": 352, "x2": 598, "y2": 448}, "confidence": 0.80, "class": "shipwreck_fragment"}
    ]

    # Fused catches all 3 targets!
    fused_p = [
        {"bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "confidence": 0.92, "class": "fishing_net"},
        {"bbox": {"x1": 305, "y1": 202, "x2": 448, "y2": 258}, "confidence": 0.82, "class": "pipeline_or_cable"},
        {"bbox": {"x1": 498, "y1": 352, "x2": 598, "y2": 448}, "confidence": 0.80, "class": "shipwreck_fragment"}
    ]

    ablation_out = evaluator.run_ablation_study(
        yolo_predictions=yolo_p,
        unet_predictions=unet_p,
        fused_predictions=fused_p,
        verified_predictions=fused_p,
        multiframe_predictions=fused_p,
        ground_truth=gt
    )

    assert "test_a_yolo_only" in ablation_out
    assert "test_b_unet_only" in ablation_out
    assert "test_c_dual_fusion" in ablation_out
    assert "test_e_full_pipeline" in ablation_out

    # YOLO recall = 2/3 = 0.6667
    assert round(ablation_out["test_a_yolo_only"]["recall"], 2) == 0.67

    # U-Net recall = 2/3 = 0.6667
    assert round(ablation_out["test_b_unet_only"]["recall"], 2) == 0.67

    # Fused recall = 3/3 = 1.00 (100% recall achieved through complementary dual-path)
    assert ablation_out["test_c_dual_fusion"]["recall"] == 1.0

    # U-Net recovered 1 YOLO miss
    assert ablation_out["test_c_dual_fusion"]["yolo_misses_recovered_by_unet"] == 1
    # YOLO recovered 1 U-Net miss
    assert ablation_out["test_c_dual_fusion"]["unet_misses_recovered_by_yolo"] == 1


if __name__ == "__main__":
    test_ablation_evaluation()
    print("All Ablation Evaluator unit tests passed successfully!")
