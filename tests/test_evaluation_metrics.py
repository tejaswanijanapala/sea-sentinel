"""
Unit and Integration Tests for Sea Sentinel Evaluation & Metrics Engine
========================================================================================
Validates mathematical correctness and requirements:
  - YOLOv11 detection metrics: Precision, Recall, F1, IoU, mAP@50, mAP@50-95, curves, confusion matrix.
  - U-Net segmentation metrics: Pixel Precision, Recall, F1, IoU, Dice Coefficient (per-image & aggregate).
  - Clear separation: YOLO evaluates bounding boxes; U-Net evaluates dense semantic masks (No mAP for U-Net).
  - Clean export formats: JSON & CSV structured reports.
========================================================================================
"""

import os
import sys
import json
import csv
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from evaluation.metrics_engine import MetricsEngine, calculate_bbox_iou, get_metrics_engine


class TestEvaluationMetricsModule(unittest.TestCase):

    def setUp(self):
        self.engine = MetricsEngine()

    def test_bbox_iou_calculation(self):
        """Test Bounding Box Intersection over Union (IoU) calculation."""
        # 1. Identical boxes -> IoU = 1.0
        boxA = [10, 10, 50, 50]
        boxB = [10, 10, 50, 50]
        iou_same = calculate_bbox_iou(boxA, boxB)
        self.assertAlmostEqual(iou_same, 1.0, places=4)

        # 2. Non-overlapping boxes -> IoU = 0.0
        boxC = [100, 100, 150, 150]
        iou_none = calculate_bbox_iou(boxA, boxC)
        self.assertEqual(iou_none, 0.0)

        # 3. 50% horizontal overlap
        box1 = [0, 0, 100, 100]  # area = 10,000
        box2 = [50, 0, 150, 100] # area = 10,000
        # Inter: [50, 0, 100, 100] -> area = 5,000
        # Union: 10,000 + 10,000 - 5,000 = 15,000
        # IoU = 5,000 / 15,000 = 1/3 = 0.3333
        iou_half = calculate_bbox_iou(box1, box2)
        self.assertAlmostEqual(iou_half, 1.0 / 3.0, places=4)

    def test_yolo_evaluation_metrics(self):
        """Test YOLOv11 evaluation on test split and ensure all required metrics exist."""
        yolo_res = self.engine.evaluate_yolo(split="test")
        
        self.assertIn("precision", yolo_res)
        self.assertIn("recall", yolo_res)
        self.assertIn("f1_score", yolo_res)
        self.assertIn("iou", yolo_res)
        self.assertIn("map50", yolo_res)
        self.assertIn("map50_95", yolo_res)
        self.assertIn("iou_stats", yolo_res)
        self.assertIn("per_class", yolo_res)
        self.assertIn("confusion_matrix", yolo_res)
        self.assertIn("curves", yolo_res)

        # Verify value ranges [0.0, 1.0]
        self.assertGreaterEqual(yolo_res["precision"], 0.0)
        self.assertLessEqual(yolo_res["precision"], 1.0)
        self.assertGreaterEqual(yolo_res["recall"], 0.0)
        self.assertLessEqual(yolo_res["recall"], 1.0)
        self.assertGreaterEqual(yolo_res["map50"], 0.0)
        self.assertLessEqual(yolo_res["map50"], 1.0)
        self.assertGreaterEqual(yolo_res["map50_95"], 0.0)
        self.assertLessEqual(yolo_res["map50_95"], 1.0)

        # Check curves presence
        curves = yolo_res["curves"]
        self.assertIn("precision_recall", curves)
        self.assertIn("f1_confidence", curves)

        # Check per class
        self.assertIsInstance(yolo_res["per_class"], list)
        self.assertGreater(len(yolo_res["per_class"]), 0)

    def test_unet_segmentation_metrics(self):
        """Test U-Net semantic segmentation metrics on test split."""
        unet_res = self.engine.evaluate_unet(split="test")

        self.assertIn("precision", unet_res)
        self.assertIn("recall", unet_res)
        self.assertIn("f1_score", unet_res)
        self.assertIn("iou", unet_res)
        self.assertIn("dice", unet_res)
        self.assertIn("dataset_micro_aggregate", unet_res)
        self.assertIn("per_image", unet_res)
        self.assertIn("dice_distribution", unet_res)

        # Explicit requirement: No mAP for U-Net
        self.assertNotIn("map50", unet_res)
        self.assertNotIn("map50_95", unet_res)

        # Check values
        self.assertGreater(unet_res["precision"], 0.5)
        self.assertGreater(unet_res["recall"], 0.5)
        self.assertGreater(unet_res["f1_score"], 0.5)
        self.assertGreater(unet_res["iou"], 0.5)
        self.assertGreater(unet_res["dice"], 0.5)

        # Check per-image items
        self.assertGreater(len(unet_res["per_image"]), 0)
        first_img = unet_res["per_image"][0]
        self.assertIn("image_name", first_img)
        self.assertIn("pixel_tp", first_img)
        self.assertIn("pixel_fp", first_img)
        self.assertIn("pixel_fn", first_img)
        self.assertIn("iou", first_img)
        self.assertIn("dice", first_img)

    def test_full_evaluation_workflow_and_reports(self):
        """Test complete workflow, JSON and CSV report generation."""
        report = self.engine.run_full_evaluation(split="test")

        # Verify top-level structure
        self.assertIn("yolo", report)
        self.assertIn("unet", report)
        self.assertIn("definitions", report)

        # Verify JSON report on disk
        json_path = os.path.join(self.engine.output_dir, "evaluation_report.json")
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
            self.assertIn("yolo", disk_data)
            self.assertIn("unet", disk_data)

        # Verify CSV summary on disk
        csv_path = os.path.join(self.engine.output_dir, "metrics_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertGreater(len(rows), 5)


if __name__ == "__main__":
    unittest.main()
