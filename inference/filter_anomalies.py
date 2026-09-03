"""
Stage 5: Anomaly Filtering & Confidence Calibration CLI
Applies Algorithms 1-9, Acoustic Shadow-Highlight Pairing, and DBSCAN Rock Cluster Suppression
to filter candidate detections and classify them into:
  - Confirmed Debris (>= 75%)
  - Suspicious Anomaly (40% - 74%)
  - Rejected Noise (< 40%)
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.anomaly_detection.autoencoder import AnomalyDetector
from ai.anomaly_detection.rock_cluster_filter import DBSCANRockFilter


def parse_args():
    parser = argparse.ArgumentParser(description="Filter false positives and calibrate detection confidence")
    parser.add_argument("--detections", type=str, required=False, default="",
                        help="Path to detections JSON file")
    parser.add_argument("--image", type=str, required=False, default="",
                        help="Path to original full-scale sonar image for patch extraction")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "checkpoints", "autoencoder", "baseline_autoencoder.pt"),
                        help="Path to trained autoencoder checkpoint")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "anomaly_detection", "inference"),
                        help="Directory to save calibrated detections JSON")
    return parser.parse_args()


def run_filtering(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("STAGE 5: ANOMALY FILTERING & CONFIDENCE CALIBRATION")
    print("=" * 70)

    detector = AnomalyDetector(checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None)
    rock_filter = DBSCANRockFilter(eps=75.0, min_samples=4)

    # 1. Load detections
    detections = []
    if args.detections and os.path.exists(args.detections):
        with open(args.detections, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                detections = data
            elif isinstance(data, dict):
                detections = data.get("detections", data.get("results", []))
    else:
        # Generate sample representative detections for validation
        print("[INFO] No input detections specified; running on benchmark test candidates.")
        detections = [
            {"object_id": "OBJ_001", "class": "fishing_net", "confidence": 0.88,
             "bbox": {"x1": 150, "y1": 200, "x2": 260, "y2": 320}},
            {"object_id": "OBJ_002", "class": "pipeline_or_cable", "confidence": 0.82,
             "bbox": {"x1": 400, "y1": 100, "x2": 450, "y2": 500}},
            # Geological rock cluster simulation (dense cluster of small highlights)
            {"object_id": "OBJ_003_ROCK", "class": "riprap_debris", "confidence": 0.58,
             "bbox": {"x1": 310, "y1": 210, "x2": 330, "y2": 230}},
            {"object_id": "OBJ_004_ROCK", "class": "riprap_debris", "confidence": 0.55,
             "bbox": {"x1": 325, "y1": 225, "x2": 345, "y2": 245}},
            {"object_id": "OBJ_005_ROCK", "class": "riprap_debris", "confidence": 0.52,
             "bbox": {"x1": 315, "y1": 240, "x2": 335, "y2": 260}},
            {"object_id": "OBJ_006_ROCK", "class": "riprap_debris", "confidence": 0.49,
             "bbox": {"x1": 330, "y1": 215, "x2": 350, "y2": 235}},
            # Weak speckle noise return
            {"object_id": "OBJ_007_NOISE", "class": "seabed_surface", "confidence": 0.38,
             "bbox": {"x1": 50, "y1": 50, "x2": 65, "y2": 65}}
        ]

    # Load image context if available
    img_context = None
    if args.image and os.path.exists(args.image):
        img_context = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    print(f"Applying DBSCAN Rock Cluster Filter across {len(detections)} detections...")
    clustered_detections = rock_filter.filter_detections(detections)

    print("Evaluating Autoencoder Reconstruction & Acoustic Shadow-Highlight Pairing...")
    calibrated_results = []
    stats = {"confirmed_debris": 0, "suspicious_anomaly": 0, "noise_rejected": 0}

    for det in clustered_detections:
        eval_res = detector.evaluate_detection(det, image_context=img_context)
        status = eval_res["status"]
        stats[status] = stats.get(status, 0) + 1

        record = {
            "object_id": det.get("object_id"),
            "class": det.get("class"),
            "bbox": det.get("bbox"),
            "raw_confidence": eval_res["raw_confidence"],
            "calibrated_confidence": eval_res["calibrated_confidence"],
            "status": status,
            "is_anomaly": eval_res["is_anomaly"],
            "reconstruction_error": eval_res["reconstruction_error"],
            "shadow_verified": eval_res["shadow_verified"],
            "shadow_score": eval_res["shadow_score"],
            "is_rock_cluster": det.get("is_rock_cluster", False),
            "rock_density_penalty": eval_res["rock_penalty"]
        }
        calibrated_results.append(record)

        print(f"  {record['object_id']:<15} | Raw: {record['raw_confidence']:.2f} -> Cal: {record['calibrated_confidence']:.2f} | "
              f"Rock Cluster: {str(record['is_rock_cluster']):<5} | Status: {status.upper()}")

    print("\n" + "=" * 50)
    print("CALIBRATION SUMMARY:")
    print(f"  Confirmed Debris (>= 75%):    {stats['confirmed_debris']}")
    print(f"  Suspicious Anomaly (40%-74%): {stats['suspicious_anomaly']}")
    print(f"  Noise Rejected (< 40%):       {stats['noise_rejected']}")
    print("=" * 50)

    out_file = os.path.join(args.output_dir, "filtered_detections.json")
    with open(out_file, "w") as f:
        json.dump({
            "stage": "Stage 5: Anomaly Detection & False-Positive Filtering",
            "total_candidates": len(calibrated_results),
            "counts": stats,
            "detections": calibrated_results
        }, f, indent=2)

    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    args = parse_args()
    run_filtering(args)
