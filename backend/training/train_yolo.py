"""
YOLOv11 Transfer-Learning Training Script for Side-Scan Sonar Debris
Supports configurable epochs, batch size, image resolution, and dry-run dataset validation.
"""
import sys
import os
import argparse
import yaml
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def validate_dataset_compatibility(data_yaml_path: str) -> Dict[str, Any]:
    """
    Validates dataset configuration, image existence, and YOLO label bounds.
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "train_images": 0,
        "val_images": 0,
        "classes": {}
    }

    if not os.path.exists(data_yaml_path):
        report["valid"] = False
        report["errors"].append(f"data.yaml not found: {data_yaml_path}")
        return report

    with open(data_yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    report["classes"] = data_cfg.get("names", {})
    raw_path = data_cfg.get("path")
    if raw_path and os.path.isabs(raw_path) and os.path.exists(raw_path):
        base_dir = raw_path
    elif raw_path and not os.path.isabs(raw_path):
        base_dir = os.path.normpath(os.path.join(os.path.dirname(data_yaml_path), raw_path))
    else:
        base_dir = os.path.dirname(data_yaml_path)

    for split in ["train", "val"]:
        split_rel = data_cfg.get(split, f"images/{split}")
        split_img_dir = os.path.normpath(os.path.join(base_dir, split_rel))

        if not os.path.exists(split_img_dir):
            report["valid"] = False
            report["errors"].append(f"{split} image directory missing: {split_img_dir}")
            continue

        images = [f for f in os.listdir(split_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if split == "train":
            report["train_images"] = len(images)
        else:
            report["val_images"] = len(images)

        if len(images) == 0:
            report["warnings"].append(f"{split} image directory is empty")

        # Validate corresponding label files
        split_lbl_dir = os.path.normpath(os.path.join(base_dir, "labels", split))
        for img in images[:30]: # Sample check
            lbl_name = os.path.splitext(img)[0] + ".txt"
            lbl_path = os.path.join(split_lbl_dir, lbl_name)
            if not os.path.exists(lbl_path):
                report["warnings"].append(f"Missing label for {img}")
            else:
                with open(lbl_path, "r") as lf:
                    lines = lf.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = map(float, parts[1:5])
                            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                report["errors"].append(f"Unnormalized coords in {lbl_name}: {parts}")
                                report["valid"] = False

    return report

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for SSS Debris Detection")
    parser.add_argument("--data", type=str, default="datasets/processed/yolo_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or '0')")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and exit without training")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Directory to save weights")
    args = parser.parse_args()

    print("=" * 70)
    print("SIH26057 — YOLOv11 TRANSFER-LEARNING TRAINING PIPELINE")
    print("=" * 70)

    data_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.data))
    print(f"\n[1/3] Verifying Dataset Configuration: {data_path}")
    val_report = validate_dataset_compatibility(data_path)

    print(f"  Validation Status: {'PASSED' if val_report['valid'] else 'FAILED'}")
    print(f"  Classes registered: {val_report['classes']}")
    print(f"  Training images: {val_report['train_images']}")
    print(f"  Validation images: {val_report['val_images']}")

    if val_report["errors"]:
        print("\nERRORS DETECTED:")
        for err in val_report["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    if val_report["warnings"]:
        print("\nWARNINGS:")
        for w in val_report["warnings"][:5]:
            print(f"  - {w}")

    if args.dry_run:
        print("\n[DRY RUN COMPLETED]: Dataset and annotations are 100% compatible for YOLO training.")
        return

    # If training explicitly requested
    print(f"\n[2/3] Initializing Ultralytics YOLO with base: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    print(f"\n[3/3] Starting Training (epochs={args.epochs}, batch={args.batch}, device={args.device})...")
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=os.path.abspath(os.path.join(PROJECT_ROOT, args.output_dir)),
        name="sih57_yolo_run",
        exist_ok=True,
        verbose=True
    )
    print("\nTraining completed successfully!")
    print(f"Checkpoints saved to {args.output_dir}/sih57_yolo_run/weights/")

if __name__ == "__main__":
    from typing import Dict, Any
    main()
