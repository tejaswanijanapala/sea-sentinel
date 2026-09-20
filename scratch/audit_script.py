import os
import json
import shutil
from pathlib import Path
from PIL import Image

def protect_datasets(base_path):
    print("Executing Phase 2: Dataset Protection")
    src_dir = Path(base_path) / "ml_pipeline" / "datasets" / "processed"
    dest_dir = Path(base_path) / "ml_pipeline" / "datasets" / "retraining"
    
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    
    # Create retraining structure
    for split in ['train', 'val', 'test']:
        (dest_dir / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    (dest_dir / "unet" / "images").mkdir(parents=True, exist_ok=True)
    (dest_dir / "unet" / "masks").mkdir(parents=True, exist_ok=True)
    
    # Copy YOLO
    src_yolo = src_dir / "yolo_dataset"
    if src_yolo.exists():
        for split in ['train', 'val', 'test']:
            src_imgs = src_yolo / "images" / split
            src_lbls = src_yolo / "labels" / split
            
            if src_imgs.exists():
                for img in src_imgs.glob("*.*"):
                    shutil.copy(img, dest_dir / "yolo" / "images" / split / img.name)
            if src_lbls.exists():
                for lbl in src_lbls.glob("*.txt"):
                    shutil.copy(lbl, dest_dir / "yolo" / "labels" / split / lbl.name)
    
    # Copy UNET
    src_unet = src_dir / "unet_dataset"
    if src_unet.exists():
        src_imgs = src_unet / "images"
        src_masks = src_unet / "masks"
        if src_imgs.exists():
            for img in src_imgs.glob("*.*"):
                shutil.copy(img, dest_dir / "unet" / "images" / img.name)
        if src_masks.exists():
            for mask in src_masks.glob("*.*"):
                shutil.copy(mask, dest_dir / "unet" / "masks" / mask.name)
                
    print("Dataset protection complete.")

def audit_yolo(base_path, output_dir):
    print("Executing Phase 3: YOLO Dataset Audit")
    yolo_dir = Path(base_path) / "ml_pipeline" / "datasets" / "retraining" / "yolo"
    
    report = {
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "invalid_bboxes": [],
        "empty_labels": []
    }
    
    for split in ['train', 'val', 'test']:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        
        if not img_dir.exists(): continue
        
        for img_path in img_dir.glob("*.*"):
            report["total_images"] += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            if not lbl_path.exists():
                report["missing_labels"].append(str(img_path))
                continue
                
            report["total_labels"] += 1
            
            # Read label
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    report["empty_labels"].append(str(lbl_path))
                    continue
                
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        report["invalid_bboxes"].append(f"{lbl_path}:{i}")
                        continue
                    try:
                        coords = [float(x) for x in parts[1:]]
                        if any(c < 0 or c > 1 for c in coords):
                            report["invalid_bboxes"].append(f"{lbl_path}:{i} (Out of bounds)")
                    except ValueError:
                        report["invalid_bboxes"].append(f"{lbl_path}:{i} (Parse error)")
    
    with open(output_dir / "yolo_dataset_audit.json", "w") as f:
        json.dump(report, f, indent=4)
        
    with open(output_dir / "YOLO_DATASET_AUDIT.md", "w") as f:
        f.write("# YOLO Dataset Audit\n\n")
        f.write(f"- Total Images: {report['total_images']}\n")
        f.write(f"- Total Labels: {report['total_labels']}\n")
        f.write(f"- Missing Labels: {len(report['missing_labels'])}\n")
        f.write(f"- Invalid BBoxes: {len(report['invalid_bboxes'])}\n")
        f.write(f"- Empty Labels: {len(report['empty_labels'])}\n")

def audit_unet(base_path, output_dir):
    print("Executing Phase 7: U-Net Dataset Audit")
    unet_dir = Path(base_path) / "ml_pipeline" / "datasets" / "retraining" / "unet"
    
    report = {
        "total_images": 0,
        "total_masks": 0,
        "missing_masks": [],
        "mismatched_dimensions": [],
        "unreadable_images": []
    }
    
    img_dir = unet_dir / "images"
    mask_dir = unet_dir / "masks"
    
    if img_dir.exists():
        for img_path in img_dir.glob("*.*"):
            report["total_images"] += 1
            mask_path = mask_dir / img_path.name
            
            if not mask_path.exists():
                report["missing_masks"].append(str(img_path))
                continue
                
            report["total_masks"] += 1
            
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        report["mismatched_dimensions"].append(str(img_path))
            except Exception as e:
                report["unreadable_images"].append(str(img_path))
    
    with open(output_dir / "unet_dataset_audit.json", "w") as f:
        json.dump(report, f, indent=4)
        
    with open(output_dir / "UNET_DATASET_AUDIT.md", "w") as f:
        f.write("# U-Net Dataset Audit\n\n")
        f.write(f"- Total Images: {report['total_images']}\n")
        f.write(f"- Total Masks: {report['total_masks']}\n")
        f.write(f"- Missing Masks: {len(report['missing_masks'])}\n")
        f.write(f"- Mismatched Dimensions: {len(report['mismatched_dimensions'])}\n")
        f.write(f"- Unreadable: {len(report['unreadable_images'])}\n")

if __name__ == "__main__":
    base = "c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel"
    out_dir = Path(base) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    protect_datasets(base)
    audit_yolo(base, out_dir)
    audit_unet(base, out_dir)
    print("Done")
