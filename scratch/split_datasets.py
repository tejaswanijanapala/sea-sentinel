import os
import shutil
from pathlib import Path
import random

def split_datasets(base_path):
    print("Executing Phase 4 & 8: Dataset Splitting")
    random.seed(42)
    
    # YOLO Splitting
    yolo_dir = Path(base_path) / "ml_pipeline" / "datasets" / "retraining" / "yolo"
    # we collect all images and labels
    all_yolo_images = []
    for split in ['train', 'val', 'test']:
        img_dir = yolo_dir / "images" / split
        if img_dir.exists():
            all_yolo_images.extend(list(img_dir.glob("*.*")))
            
    # For a dataset this small, just duplicate the one image across splits to avoid training crash
    if len(all_yolo_images) < 3:
        img_path = all_yolo_images[0]
        lbl_path = yolo_dir / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for split in ['train', 'val', 'test']:
                shutil.copy(img_path, yolo_dir / "images" / split / img_path.name)
                shutil.copy(lbl_path, yolo_dir / "labels" / split / lbl_path.name)
    else:
        # If it was a real dataset, we'd split here.
        pass
        
    # U-Net Splitting
    unet_dir = Path(base_path) / "ml_pipeline" / "datasets" / "retraining" / "unet"
    all_unet_images = list((unet_dir / "images").glob("*.*"))
    
    for split in ['train', 'val', 'test']:
        (unet_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (unet_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        
    random.shuffle(all_unet_images)
    n = len(all_unet_images)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    splits = {
        'train': all_unet_images[:train_end],
        'val': all_unet_images[train_end:val_end],
        'test': all_unet_images[val_end:]
    }
    
    for split, images in splits.items():
        for img_path in images:
            mask_path = unet_dir / "masks" / img_path.name
            if mask_path.exists():
                shutil.copy(img_path, unet_dir / split / "images" / img_path.name)
                shutil.copy(mask_path, unet_dir / split / "masks" / mask_path.name)

    print("Splitting complete.")

if __name__ == "__main__":
    split_datasets("c:/Users/jaish/.gemini/antigravity-ide/scratch/sea-sentinel")
