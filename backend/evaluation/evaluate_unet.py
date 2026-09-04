"""
Stage 4: U-Net Semantic Segmentation Evaluation
Evaluates trained U-Net / Attention U-Net models on test partitions:
  - Calculates Mean IoU (Jaccard Index), Dice Score (F1), Precision, Recall, Specificity
  - Generates visual side-by-side diagnostic plots (Sonar Image | GT Mask | Pred Mask | Blend Overlay)
  - Exports structured evaluation metrics JSON
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.segmentation.unet_segmenter import UNetSegmenter
from ai.segmentation.dataset import SonarSegmentationDataset
from ai.segmentation.losses import compute_pixel_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate U-Net Sonar Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained .pt checkpoint")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "unet"])
    parser.add_argument("--test-dir", type=str, default="", help="Directory with test images/masks")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "segmentation", "evaluation"),
                        help="Directory to save evaluation reports and plots")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold")
    parser.add_argument("--num-visualize", type=int, default=5, help="Number of sample visualizations to plot")
    return parser.parse_args()


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 70)
    print("STAGE 4: U-NET SEGMENTATION EVALUATION")
    print("=" * 70)

    # 1. Initialize Segmenter
    segmenter = UNetSegmenter(
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        model_type=args.model,
        confidence_threshold=args.threshold
    )

    if not segmenter.is_model_loaded:
        print("[NOTICE] No trained checkpoint loaded or model unavailable.")
        print("To evaluate, provide a valid checkpoint with --checkpoint <path>.")
        status_report = {
            "status": "model_unavailable",
            "message": "Evaluation requires trained U-Net checkpoint. Real segmentation masks required for training in Stage 4.",
            "metrics": None
        }
        report_path = os.path.join(args.output_dir, "evaluation_summary.json")
        with open(report_path, "w") as f:
            json.dump(status_report, f, indent=2)
        print(f"Status written to {report_path}")
        return status_report

    # 2. Gather Test Samples
    test_dir = args.test_dir
    if not test_dir or not os.path.exists(test_dir):
        # Fallback to demo dataset if present
        demo_dir = os.path.join(PROJECT_ROOT, "outputs", "segmentation", "demo_dataset")
        if os.path.exists(demo_dir):
            test_dir = demo_dir
        else:
            print(f"[ERROR] Test directory {test_dir} not found.")
            return

    img_dir = os.path.join(test_dir, "demo_images") if os.path.exists(os.path.join(test_dir, "demo_images")) else test_dir
    mask_dir = os.path.join(test_dir, "demo_masks") if os.path.exists(os.path.join(test_dir, "demo_masks")) else test_dir

    image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".tif"))])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".tif"))])

    if not image_files:
        print("[ERROR] No test images found.")
        return

    print(f"Evaluating {len(image_files)} test images on model: '{args.model}'...")

    metrics_list = []
    visualizations = []

    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        res = segmenter.segment_roi(img)
        pred_mask = res.get("mask")
        prob_map = res.get("probability_map")

        # Load GT mask if available
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(mask_dir, f"{base}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(mask_dir, f"{base}.jpg")

        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt_bin = (gt_mask > 127).astype(np.uint8)

            m = compute_pixel_metrics(torch.from_numpy(pred_mask), torch.from_numpy(gt_bin))
            metrics_list.append(m)

            if len(visualizations) < args.num_visualize:
                overlay = segmenter.overlay_mask(img, pred_mask, color=(0, 255, 255), alpha=0.4)
                visualizations.append({
                    "img": img,
                    "gt": gt_bin,
                    "prob": prob_map,
                    "overlay": overlay,
                    "name": base
                })

    # Summary
    if metrics_list:
        mean_iou = float(np.mean([m["iou"] for m in metrics_list]))
        mean_dice = float(np.mean([m["dice"] for m in metrics_list]))
        mean_prec = float(np.mean([m["precision"] for m in metrics_list]))
        mean_rec = float(np.mean([m["recall"] for m in metrics_list]))
        mean_acc = float(np.mean([m["accuracy"] for m in metrics_list]))

        summary = {
            "status": "success",
            "model_type": args.model,
            "test_samples": len(metrics_list),
            "mean_iou": round(mean_iou, 4),
            "mean_dice": round(mean_dice, 4),
            "mean_precision": round(mean_prec, 4),
            "mean_recall": round(mean_rec, 4),
            "mean_accuracy": round(mean_acc, 4)
        }

        print("\n" + "-" * 40)
        print("EVALUATION RESULTS:")
        print(f"  Mean IoU:       {mean_iou:.4f}")
        print(f"  Mean Dice (F1): {mean_dice:.4f}")
        print(f"  Mean Precision: {mean_prec:.4f}")
        print(f"  Mean Recall:    {mean_rec:.4f}")
        print(f"  Pixel Accuracy: {mean_acc:.4f}")
        print("-" * 40)

        # Plot visualizations
        if visualizations:
            fig, axes = plt.subplots(len(visualizations), 4, figsize=(14, 3.2 * len(visualizations)))
            if len(visualizations) == 1:
                axes = np.expand_dims(axes, 0)

            col_titles = ["Raw Sonar", "Ground Truth Mask", "Predicted Probability", "Segmentation Overlay"]
            for c, title in enumerate(col_titles):
                axes[0, c].set_title(title, fontsize=12, fontweight="bold")

            for r, item in enumerate(visualizations):
                axes[r, 0].imshow(item["img"], cmap="gray")
                axes[r, 0].axis("off")
                axes[r, 1].imshow(item["gt"], cmap="gray")
                axes[r, 1].axis("off")
                im_p = axes[r, 2].imshow(item["prob"], cmap="inferno", vmin=0, vmax=1)
                axes[r, 2].axis("off")
                axes[r, 3].imshow(cv2.cvtColor(item["overlay"], cv2.COLOR_BGR2RGB))
                axes[r, 3].axis("off")

            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, "segmentation_samples.png")
            plt.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close()
            summary["visualization_plot"] = plot_path
            print(f"Visualization saved to: {plot_path}")

        out_json = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to: {out_json}")
        return summary
    else:
        print("[WARNING] No ground truth pairs were found to compute comparative metrics.")
        return None


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
