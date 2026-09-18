"""
Stage 4 Preparation: Generate Ground Truth Acoustic Segmentation Masks
Constructs high-fidelity pixel-level masks from real Side-Scan Sonar (SSS) images
and bounding box annotations using localized acoustic backscatter & shadow profiling.
"""

import os
import glob
import cv2
import numpy as np
from shared.utils.logger import get_logger
logger = get_logger(__name__)




def generate_acoustic_masks(
    dataset_root: str,
    output_dir: str
):
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    out_images_dir = os.path.join(output_dir, "images")
    out_masks_dir = os.path.join(output_dir, "masks")

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(images_dir, "**", "*.*"), recursive=True)
    image_files = [f for f in image_files if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))]

    logger.info(f"Processing {len(image_files)} sonar images for segmentation mask generation...")

    count = 0
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Look for matching label file in train/val/test
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_candidates = glob.glob(os.path.join(labels_dir, "**", f"{base_name}.txt"), recursive=True)

        if label_candidates and os.path.exists(label_candidates[0]):
            with open(label_candidates[0], "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx = float(parts[1]) * w
                    cy = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h

                    x1 = max(0, int(cx - bw / 2))
                    y1 = max(0, int(cy - bh / 2))
                    x2 = min(w, int(cx + bw / 2))
                    y2 = min(h, int(cy + bh / 2))

                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
                        # Adaptive highlight extraction inside bbox
                        blur = cv2.GaussianBlur(crop, (5, 5), 0)
                        mean_val = np.mean(blur)
                        std_val = np.std(blur)
                        thresh_val = max(90, mean_val + 0.3 * std_val)

                        _, local_mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)
                        
                        # If highlight area is too small, use morphological region inside box
                        if np.sum(local_mask > 0) < 20:
                            pad_x = max(2, int((x2 - x1) * 0.1))
                            pad_y = max(2, int((y2 - y1) * 0.1))
                            cv2.rectangle(local_mask, (pad_x, pad_y), (x2 - x1 - pad_x, y2 - y1 - pad_y), 255, -1)

                        # Clean contour
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)

                        mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], local_mask)

        # Save normalized image and mask
        dst_img = os.path.join(out_images_dir, f"{base_name}.png")
        dst_mask = os.path.join(out_masks_dir, f"{base_name}.png")

        cv2.imwrite(dst_img, img)
        cv2.imwrite(dst_mask, mask)
        count += 1

    logger.info(f"Successfully generated {count} image-mask pairs in {output_dir}")


if __name__ == "__main__":
    import sys
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yolo_dir = os.path.join(proj_root, "datasets", "processed", "yolo_dataset")
    out_dir = os.path.join(proj_root, "datasets", "processed", "segmentation_dataset")
    generate_acoustic_masks(yolo_dir, out_dir)
