import os
import shutil

targets = [
    (1, 0.15, 0.40, 0.35, 0.45),
    (3, 0.45, 0.60, 0.55, 0.70),
    (2, 0.70, 0.20, 0.85, 0.35),
    (4, 0.25, 0.75, 0.45, 0.85),
    (0, 0.55, 0.10, 0.65, 0.25),
    (3, 0.80, 0.65, 0.90, 0.75)
]

dataset_dir = "backend/datasets/processed/yolo_dataset"
images_dir = os.path.join(dataset_dir, "images", "train")
labels_dir = os.path.join(dataset_dir, "labels", "train")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Copy the image
src_img = "frontend/assets/samples/SURVEY_54434B1B_raw.png"
dst_img = os.path.join(images_dir, "SURVEY_54434B1B_raw.png")
if os.path.exists(src_img):
    shutil.copy(src_img, dst_img)

# Generate YOLO labels
label_path = os.path.join(labels_dir, "SURVEY_54434B1B_raw.txt")
with open(label_path, "w") as f:
    for cls, x1, y1, x2, y2 in targets:
        # Convert min/max to center width height
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# Create dataset.yaml
yaml_path = os.path.join(dataset_dir, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(dataset_dir).replace(chr(92), '/')}\n")
    f.write("train: images/train\n")
    f.write("val: images/train\n")
    f.write("names:\n")
    f.write("  0: fishing_net\n")
    f.write("  1: pipeline_or_cable\n")
    f.write("  2: shipwreck_fragment\n")
    f.write("  3: engine_debris\n")
    f.write("  4: riprap_debris\n")

print(f"Dataset generated at {dataset_dir}")
