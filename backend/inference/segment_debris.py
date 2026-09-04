"""
Stage 4: Semantic Segmentation Inference Script
Runs U-Net / Attention U-Net inference on sonar chips or full-scale SSS mosaics.
Outputs:
  - High-contrast visual overlays (RGB PNG)
  - Raw binary masks (PNG)
  - Structured segmentation summary JSON (contours count, pixel area, mean confidence)
"""

import os
import sys
import argparse
import json
import glob
from typing import Dict, Any, List
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.segmentation.unet_segmenter import UNetSegmenter


def parse_args():
    parser = argparse.ArgumentParser(description="Segment anthropogenic debris from SSS imagery")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to single sonar image (.png, .jpg, .tif) or directory")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to trained U-Net checkpoint (.pt)")
    parser.add_argument("--model", type=str, default="attention_unet", choices=["attention_unet", "unet"],
                        help="Model architecture: 'attention_unet' or 'unet'")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "roi", "mosaic"],
                        help="Inference mode: 'roi' (single target chip), 'mosaic' (sliding window), or 'auto'")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Debris probability segmentation threshold")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs", "segmentation", "inference"),
                        help="Directory to save output masks and overlays")
    return parser.parse_args()


def run_inference(args):
    os.makedirs(args.output_dir, exist_ok=True)
    masks_dir = os.path.join(args.output_dir, "masks")
    overlays_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE 4: U-NET DEBRIS SEGMENTATION INFERENCE")
    print("=" * 70)

    # Initialize Segmenter
    segmenter = UNetSegmenter(
        checkpoint_path=args.checkpoint if args.checkpoint else None,
        model_type=args.model,
        confidence_threshold=args.threshold
    )

    # Collect input images
    if os.path.isdir(args.input):
        image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths = sorted(image_paths)
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        print(f"[ERROR] Input path does not exist: {args.input}")
        return

    if not image_paths:
        print(f"[ERROR] No valid sonar images found at {args.input}")
        return

    print(f"Found {len(image_paths)} sonar image(s) for segmentation.")
    print(f"Model loaded: {segmenter.is_model_loaded} ({segmenter.model_type if segmenter.is_model_loaded else 'None'})")

    results = []

    for idx, path in enumerate(image_paths, 1):
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[{idx}/{len(image_paths)}] Skipping unreadable: {filename}")
            continue

        h, w = img.shape[:2]

        # Determine mode
        if args.mode == "auto":
            mode = "mosaic" if (h > 512 or w > 512) else "roi"
        else:
            mode = args.mode

        if not segmenter.is_model_loaded:
            # Operational fallback: report model unavailable without fabricating fake masks
            res = segmenter.segment_roi(img)
            results.append({
                "file": filename,
                "path": path,
                "status": res["status"],
                "message": res.get("message")
            })
            continue

        if mode == "mosaic":
            seg_res = segmenter.segment_full_image(img, threshold=args.threshold)
        else:
            seg_res = segmenter.segment_roi(img, threshold=args.threshold)

        mask = seg_res.get("mask")
        prob_map = seg_res.get("probability_map")
        area_px = seg_res.get("total_area_px", 0)
        contours_count = seg_res.get("contours_count", 0)

        # Save binary mask
        mask_path = os.path.join(masks_dir, f"{stem}_mask.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        # Save colored visual overlay
        overlay = segmenter.overlay_mask(img, mask, color=(0, 255, 255), alpha=0.45, draw_contours=True)
        overlay_path = os.path.join(overlays_dir, f"{stem}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        record = {
            "file": filename,
            "dimensions": {"height": h, "width": w},
            "inference_mode": mode,
            "total_area_px": area_px,
            "contours_count": contours_count,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "status": "success"
        }
        results.append(record)
        print(f"[{idx}/{len(image_paths)}] {filename} -> Area: {area_px} px | Contours: {contours_count} | Mode: {mode}")

    # Export structured results
    summary_path = os.path.join(args.output_dir, "segmentation_results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model_type": args.model,
            "checkpoint": args.checkpoint,
            "model_loaded": segmenter.is_model_loaded,
            "total_processed": len(results),
            "results": results
        }, f, indent=2)

    print(f"\n[INFERENCE COMPLETED]: Results saved to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
