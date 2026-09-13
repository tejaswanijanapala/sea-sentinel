"""
YOLOv11 Evaluation & Performance Metrics Pipeline
Evaluates detection models on SSS validation/test sets, computes mAP@50, mAP@50-95,
Precision, Recall, F1-score, IoU stats, confusion matrix, and PR curves.
"""
import sys
import os
import argparse
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from evaluation.metrics_engine import MetricsEngine


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Sonar Detection Model")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate ('val' or 'test')")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("SEA SENTINEL — YOLOv11 OBJECT DETECTION EVALUATION")
    print("=" * 70)

    engine = MetricsEngine(
        yolo_weights=args.weights,
        data_yaml=args.data,
        output_dir=args.output_dir
    )

    results = engine.evaluate_yolo(split=args.split)
    print("\n" + "-" * 50)
    print("YOLOv11 EVALUATION SUMMARY:")
    print(f"  Precision:       {results['precision']:.4f}")
    print(f"  Recall:          {results['recall']:.4f}")
    print(f"  F1-Score:        {results['f1_score']:.4f}")
    print(f"  Mean BBox IoU:   {results['iou']:.4f}")
    print(f"  mAP@50:          {results['map50']:.4f}")
    print(f"  mAP@50-95:       {results['map50_95']:.4f}")
    print("-" * 50)

    out_file = os.path.join(engine.output_dir, "yolo_evaluation_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved YOLO metrics to: {out_file}")


if __name__ == "__main__":
    main()

