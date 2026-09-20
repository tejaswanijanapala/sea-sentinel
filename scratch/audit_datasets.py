import os
import json
import glob
from pathlib import Path

def audit_datasets(base_path):
    report = {
        "yolo": {"train": 0, "val": 0, "test": 0, "labels_train": 0, "labels_val": 0, "labels_test": 0},
        "unet": {"images": 0, "masks": 0}
    }
    
    # YOLO
    yolo_base = Path(base_path) / "ml_pipeline" / "datasets" / "processed" / "yolo_dataset"
    if yolo_base.exists():
        for split in ["train", "val", "test"]:
            img_dir = yolo_base / "images" / split
            if img_dir.exists():
                report["yolo"][split] = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
            
            lbl_dir = yolo_base / "labels" / split
            if lbl_dir.exists():
                report["yolo"][f"labels_{split}"] = len(list(lbl_dir.glob("*.txt")))

    # UNET
    unet_base = Path(base_path) / "ml_pipeline" / "datasets" / "processed" / "unet_dataset"
    if unet_base.exists():
        img_dir = unet_base / "images"
        if img_dir.exists():
            report["unet"]["images"] = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg")))
        
        mask_dir = unet_base / "masks"
        if mask_dir.exists():
            report["unet"]["masks"] = len(list(mask_dir.glob("*.png"))) + len(list(mask_dir.glob("*.jpg")))

    print(json.dumps(report, indent=4))
        
if __name__ == "__main__":
    audit_datasets("c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel")
