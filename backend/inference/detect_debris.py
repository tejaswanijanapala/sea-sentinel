"""
YOLO Debris Inference & Visualization Script
Runs candidate debris detection on SSS chips or survey tiles with configurable confidence,
draws bounding box overlays, and exports structured detection records to JSON.
"""
import sys
import os
import argparse
import json
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.detection.yolo_detector import YOLODetector

def main():
    parser = argparse.ArgumentParser(description="Run YOLO Inference on Sonar Imagery")
    parser.add_argument("--source", type=str, default="datasets/processed/yolo_dataset/images/test", help="Path to image or directory")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output-dir", type=str, default="outputs/detection", help="Output directory")
    parser.add_argument("--max-images", type=int, default=10, help="Max images to infer if source is directory")
    args = parser.parse_args()

    print("=" * 70)
    print("SIH26057 — YOLO DEBRIS DETECTION INFERENCE")
    print("=" * 70)

    out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.output_dir))
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    detector = YOLODetector(model_path=args.weights, conf_thresh=args.conf)
    print(f"Model loaded status: {detector.is_model_loaded} (weights: {args.weights})")
    print(f"Confidence threshold: {args.conf}")

    src_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.source))
    if not os.path.exists(src_path):
        print(f"Error: source path not found: {src_path}")
        sys.exit(1)

    image_files = []
    if os.path.isdir(src_path):
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
        for f in sorted(os.listdir(src_path)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                image_files.append(os.path.join(src_path, f))
        image_files = image_files[:args.max_images]
    else:
        image_files = [src_path]

    print(f"Running inference on {len(image_files)} images...")
    all_results = []

    for img_p in image_files:
        raw_img = cv2.imread(img_p)
        if raw_img is None:
            continue

        res = detector.detect(img_p, conf_override=args.conf)
        res["source_image"] = os.path.basename(img_p)

        # Draw detections if any
        if res.get("detections"):
            annotated = detector.draw_detections(raw_img, res["detections"])
            out_img_path = os.path.join(vis_dir, f"det_{os.path.basename(img_p)}")
            cv2.imwrite(out_img_path, annotated)
            res["visualization_path"] = out_img_path

        all_results.append(res)
        print(f"  - {os.path.basename(img_p)}: {len(res.get('detections', []))} detections found (status: {res.get('status')})")

    out_json = os.path.join(out_dir, "inference_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {out_json}")
    print(f"Visualizations saved to: {vis_dir}")
    print("\n" + "=" * 70)
    print("INFERENCE RUN COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
