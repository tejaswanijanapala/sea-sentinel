"""
Stage 4: U-Net Semantic Segmentation Evaluation
Evaluates trained U-Net / Attention U-Net models on test partitions:
  - Calculates Pixel-level Precision, Recall, F1-Score, IoU (Jaccard Index), and Dice Score
  - Compares predicted segmentation masks directly with verified ground-truth masks
  - Exports per-image and dataset-wide aggregate metrics (No mAP is calculated for segmentation)
"""

import os
import sys
import argparse
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from evaluation.metrics_engine import MetricsEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net Sonar Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate ('val' or 'test')")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation reports")
    return parser.parse_args()


def evaluate(args):
    print("=" * 70)
    print("SEA SENTINEL — U-NET SEMANTIC SEGMENTATION EVALUATION")
    print("=" * 70)

    engine = MetricsEngine(
        unet_weights=args.checkpoint,
        output_dir=args.output_dir
    )

    results = engine.evaluate_unet(split=args.split)
    
    print("\n" + "-" * 50)
    print("U-NET SEGMENTATION EVALUATION RESULTS:")
    print(f"  Pixel Precision:   {results['precision']:.4f}")
    print(f"  Pixel Recall:      {results['recall']:.4f}")
    print(f"  Pixel F1-Score:    {results['f1_score']:.4f}")
    print(f"  Pixel IoU:         {results['iou']:.4f}")
    print(f"  Dice Coefficient:  {results['dice']:.4f}")
    print("-" * 50)

    out_json = os.path.join(engine.output_dir, "unet_evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved U-Net metrics to: {out_json}")
    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

