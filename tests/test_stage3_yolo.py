"""
Unit Tests for Stage 3 YOLO Debris Detection
"""
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.detection.yolo_detector import YOLODetector
from training.train_yolo import validate_dataset_compatibility
from evaluation.evaluate_yolo import compute_synthetic_confusion_matrix

def test_dataset_compatibility():
    data_yaml = os.path.join(PROJECT_ROOT, "datasets", "processed", "yolo_dataset", "data.yaml")
    report = validate_dataset_compatibility(data_yaml)
    assert report["valid"] is True, f"Validation failed with errors: {report['errors']}"
    assert report["train_images"] > 0, "Train images must be non-empty"
    assert report["val_images"] > 0, "Val images must be non-empty"
    assert "fishing_net" in report["classes"].values()

def test_detector_initialization():
    # Test without weights (honest fallback)
    detector = YOLODetector(model_path=None, conf_thresh=0.40)
    assert detector.is_model_loaded is False
    assert detector.conf_thresh == 0.40

    # Test inference output when weights are not loaded
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy)
    assert res["status"] == "model_unavailable"
    assert res["model_loaded"] is False
    assert len(res["detections"]) == 0

def test_draw_detections():
    detector = YOLODetector()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_detections = [
        {
            "object_id": "DEBRIS_0001",
            "class": "fishing_net",
            "confidence": 0.88,
            "bbox": {"x1": 20, "y1": 30, "x2": 80, "y2": 90}
        }
    ]
    annotated = detector.draw_detections(img, mock_detections)
    assert annotated.shape == (200, 200, 3)
    # Check that drawing modified pixels
    assert np.any(annotated > 0)

def test_confusion_matrix_generation():
    out_cm = os.path.join(PROJECT_ROOT, "outputs", "evaluation", "test_cm.png")
    classes = ["fishing_net", "pipeline_or_cable", "shipwreck_fragment", "engineering_platform", "riprap_debris"]
    res_path = compute_synthetic_confusion_matrix(classes, out_cm)
    assert os.path.exists(res_path)
    assert os.path.getsize(res_path) > 1000

if __name__ == "__main__":
    test_dataset_compatibility()
    test_detector_initialization()
    test_draw_detections()
    test_confusion_matrix_generation()
    print("All Stage 3 YOLO unit tests passed successfully!")
