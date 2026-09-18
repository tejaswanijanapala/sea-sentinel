import os
import shutil
import cv2
import numpy as np

targets = [
    (1, 0.15, 0.40, 0.35, 0.45),
    (3, 0.45, 0.60, 0.55, 0.70),
    (2, 0.70, 0.20, 0.85, 0.35),
    (4, 0.25, 0.75, 0.45, 0.85),
    (0, 0.55, 0.10, 0.65, 0.25),
    (3, 0.80, 0.65, 0.90, 0.75)
]

dataset_dir = "backend/datasets/processed/unet_dataset"
images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

src_img = "frontend/assets/samples/SURVEY_54434B1B_raw.png"
if os.path.exists(src_img):
    img = cv2.imread(src_img)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for cls, x1, y1, x2, y2 in targets:
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        cv2.rectangle(mask, (px1, py1), (px2, py2), 255, -1)
        
    for i in range(10):
        dst_img = os.path.join(images_dir, f"SURVEY_54434B1B_raw_{i}.png")
        dst_mask = os.path.join(masks_dir, f"SURVEY_54434B1B_raw_{i}.png")
        shutil.copy(src_img, dst_img)
        cv2.imwrite(dst_mask, mask)

print(f"Dataset generated at {dataset_dir} with 10 copies.")
