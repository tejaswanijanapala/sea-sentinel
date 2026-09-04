"""
Layer 1 Dataset Pipeline: Ingestion, Validation, Statistics, and Format Conversion
Implements Stage 1 of the SIH57 System without modifying original files.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import io
import csv
import json
import zipfile
import hashlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class DatasetPreparationPipeline:
    """
    Automated dataset inspection, validation, conversion, and statistics pipeline.
    """
    def __init__(self, raw_data_dir: str, processed_dir: str):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        # Harmonized class map for SIH57
        self.class_to_id = {
            "fishing_net": 0,
            "pipeline_or_cable": 1,
            "shipwreck_fragment": 2,
            "engineering_platform": 3,
            "riprap_debris": 4,
            "seabed_surface": 5
        }

        # Zenodo benchmark label to unified class
        self.zenodo_label_map = {
            "HN": "fishing_net",            # Hard negative / ghost fishing nets
            "POC": "pipeline_or_cable",     # Pipelines and cables
            "RO": "shipwreck_fragment",     # Rare objects (shipwrecks, buoy anchors)
            "EP": "engineering_platform",   # Engineering platforms, pile legs
            "RP": "riprap_debris",          # Riprap armor, dropped blocks
            "SS": "seabed_surface",         # Normal seafloor backscatter
            "URM": "seabed_surface",        # Underwater residual mounds (geological)
            "SM": "seabed_surface",         # Scour marks (geological)
            "SW": "seabed_surface",         # Sand waves (geological)
            "TG": "seabed_surface"          # Trench/gullies (geological)
        }

    def audit_all_sources(self) -> Dict[str, Any]:
        """
        Scans all downloaded datasets, rasters, and archives.
        """
        inventory = {
            "zip_archives": [],
            "geotiff_mosaics": [],
            "waterfall_images": [],
            "total_raw_files": 0
        }

        if not os.path.exists(self.raw_data_dir):
            return inventory

        for root, _, files in os.walk(self.raw_data_dir):
            for f in files:
                f_path = os.path.join(root, f)
                size_bytes = os.path.getsize(f_path)
                ext = os.path.splitext(f)[1].lower()

                if ext == ".zip":
                    inventory["zip_archives"].append({
                        "filename": f,
                        "path": f_path,
                        "size_mb": round(size_bytes / (1024 * 1024), 2)
                    })
                elif ext in [".tif", ".tiff"]:
                    inventory["geotiff_mosaics"].append({
                        "filename": f,
                        "path": f_path,
                        "size_mb": round(size_bytes / (1024 * 1024), 2)
                    })
                elif ext in [".png", ".jpg", ".jpeg"] and "monrovia" in f.lower():
                    inventory["waterfall_images"].append({
                        "filename": f,
                        "path": f_path,
                        "size_mb": round(size_bytes / (1024 * 1024), 2)
                    })
                inventory["total_raw_files"] += 1

        return inventory

    def inspect_zenodo_archive(self, zip_path: str, max_check: Optional[int] = None) -> Dict[str, Any]:
        """
        Deep-inspects China-Offshore-SSS-AI: validates manifest, images, hashes, formats, and corruptions.
        """
        report = {
            "archive_path": zip_path,
            "manifest_records": 0,
            "verified_images": 0,
            "corrupt_images": 0,
            "formats": {},
            "dimensions_min_max": {"min_w": 99999, "max_w": 0, "min_h": 99999, "max_h": 0},
            "class_distribution": {},
            "splits": {"train": 0, "val": 0, "test": 0, "unassigned": 0},
            "duplicate_hashes_found": 0,
            "yolo_annotations_found": False,
            "segmentation_masks_found": False
        }

        if not os.path.exists(zip_path):
            return report

        with zipfile.ZipFile(zip_path, 'r') as z:
            namelist = set(z.namelist())

            # 1. Check for YOLO and masks
            report["yolo_annotations_found"] = any(n.endswith(".txt") and "labels" in n for n in namelist)
            report["segmentation_masks_found"] = any("mask" in n.lower() for n in namelist)

            # 2. Parse train_val_test_split.csv
            splits_map = {}
            if "China-Offshore-SSS-AI/metadata/train_val_test_split.csv" in namelist:
                split_csv = z.read("China-Offshore-SSS-AI/metadata/train_val_test_split.csv").decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(split_csv))
                for row in reader:
                    img_id = row.get("image_id")
                    split = row.get("split", "unassigned").lower()
                    splits_map[img_id] = split
                    if split in report["splits"]:
                        report["splits"][split] += 1
                    else:
                        report["splits"]["unassigned"] += 1

            # 3. Parse image_manifest.csv
            seen_hashes = set()
            if "China-Offshore-SSS-AI/metadata/image_manifest.csv" in namelist:
                manifest_csv = z.read("China-Offshore-SSS-AI/metadata/image_manifest.csv").decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(manifest_csv))
                count = 0
                for row in reader:
                    report["manifest_records"] += 1
                    img_path = "China-Offshore-SSS-AI/" + row.get("release_path", "")
                    benchmark_label = row.get("benchmark_label", "")
                    abbrev = row.get("class_abbreviation", "")
                    unified_class = self.zenodo_label_map.get(abbrev, self.zenodo_label_map.get(benchmark_label, "other"))
                    
                    report["class_distribution"][unified_class] = report["class_distribution"].get(unified_class, 0) + 1

                    # Check duplicate hash
                    sha256 = row.get("sha256")
                    if sha256:
                        if sha256 in seen_hashes:
                            report["duplicate_hashes_found"] += 1
                        seen_hashes.add(sha256)

                    # Verify image stream
                    if max_check is None or count < max_check:
                        if img_path in namelist:
                            try:
                                with z.open(img_path) as img_f:
                                    with Image.open(io.BytesIO(img_f.read())) as pil_img:
                                        pil_img.verify()
                                        w, h = pil_img.size
                                        fmt = pil_img.format or "UNKNOWN"
                                        report["formats"][fmt] = report["formats"].get(fmt, 0) + 1
                                        report["dimensions_min_max"]["min_w"] = min(report["dimensions_min_max"]["min_w"], w)
                                        report["dimensions_min_max"]["max_w"] = max(report["dimensions_min_max"]["max_w"], w)
                                        report["dimensions_min_max"]["min_h"] = min(report["dimensions_min_max"]["min_h"], h)
                                        report["dimensions_min_max"]["max_h"] = max(report["dimensions_min_max"]["max_h"], h)
                                        report["verified_images"] += 1
                            except Exception:
                                report["corrupt_images"] += 1
                        count += 1

        return report

    def build_standardized_subsets(
        self,
        zip_path: str,
        sample_per_class: int = 50
    ) -> Dict[str, Any]:
        """
        Creates a clean, standardized processed dataset structure:
          1. YOLO detection dataset (images + normalized YOLO txt bounding boxes centered on chips)
          2. Anomaly baseline dataset (normal seafloor chips for Autoencoder training)
          3. Dataset manifest in JSON
        Without copying or duplicating unnecessarily.
        """
        out_yolo_dir = os.path.join(self.processed_dir, "yolo_dataset")
        out_anomaly_dir = os.path.join(self.processed_dir, "anomaly_baseline")

        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(out_yolo_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(out_yolo_dir, "labels", split), exist_ok=True)
            os.makedirs(os.path.join(out_anomaly_dir, split), exist_ok=True)

        yolo_counts = {"train": 0, "val": 0, "test": 0}
        anomaly_counts = {"train": 0, "val": 0, "test": 0}
        class_samples_extracted = {}

        if not os.path.exists(zip_path):
            return {"status": "error", "message": f"Archive not found: {zip_path}"}

        with zipfile.ZipFile(zip_path, 'r') as z:
            namelist = set(z.namelist())
            
            # Read splits
            splits_map = {}
            if "China-Offshore-SSS-AI/metadata/train_val_test_split.csv" in namelist:
                split_csv = z.read("China-Offshore-SSS-AI/metadata/train_val_test_split.csv").decode("utf-8-sig")
                for row in csv.DictReader(io.StringIO(split_csv)):
                    splits_map[row["image_id"]] = row.get("split", "train").lower()

            # Read manifest
            manifest_csv = z.read("China-Offshore-SSS-AI/metadata/image_manifest.csv").decode("utf-8-sig")
            for row in csv.DictReader(io.StringIO(manifest_csv)):
                img_id = row["image_id"]
                split = splits_map.get(img_id, "train")
                abbrev = row.get("class_abbreviation", "")
                benchmark_label = row.get("benchmark_label", "")
                unified_class = self.zenodo_label_map.get(abbrev, self.zenodo_label_map.get(benchmark_label, "seabed_surface"))

                current_extracted = class_samples_extracted.get(unified_class, 0)
                if current_extracted >= sample_per_class:
                    continue

                zip_img_path = "China-Offshore-SSS-AI/" + row["release_path"]
                if zip_img_path not in namelist:
                    continue

                img_data = z.read(zip_img_path)
                class_samples_extracted[unified_class] = current_extracted + 1

                # 1. If normal seabed, save to anomaly baseline
                if unified_class == "seabed_surface":
                    target_path = os.path.join(out_anomaly_dir, split, f"{img_id}.jpg")
                    with open(target_path, "wb") as f:
                        f.write(img_data)
                    anomaly_counts[split] += 1

                # 2. Save to YOLO dataset if anthropogenic debris class
                if unified_class in self.class_to_id and unified_class != "seabed_surface":
                    cid = self.class_to_id[unified_class]
                    target_img = os.path.join(out_yolo_dir, "images", split, f"{img_id}.jpg")
                    target_lbl = os.path.join(out_yolo_dir, "labels", split, f"{img_id}.txt")

                    with open(target_img, "wb") as f:
                        f.write(img_data)

                    # YOLO format: class_id x_center y_center width height (normalized)
                    # Since each Zenodo chip is cropped centered around the target:
                    # Normalized center = 0.5, 0.5; bounding box coverage = 0.70 width, 0.70 height
                    with open(target_lbl, "w") as f:
                        f.write(f"{cid} 0.500000 0.500000 0.700000 0.700000\n")

                    yolo_counts[split] += 1

        # Generate YOLO data.yaml
        data_yaml_path = os.path.join(out_yolo_dir, "data.yaml")
        yaml_content = f"""# YOLOv11 Sonar Debris Detection Dataset
path: {os.path.abspath(out_yolo_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: fishing_net
  1: pipeline_or_cable
  2: shipwreck_fragment
  3: engineering_platform
  4: riprap_debris
"""
        with open(data_yaml_path, "w") as f:
            f.write(yaml_content)

        return {
            "status": "success",
            "yolo_samples_extracted": yolo_counts,
            "anomaly_samples_extracted": anomaly_counts,
            "class_samples": class_samples_extracted,
            "data_yaml": data_yaml_path
        }

    def generate_visual_montage(self, zip_path: str, output_path: str) -> Optional[str]:
        """
        Creates a contact-sheet visualization showing sample chips from each target class.
        """
        if not os.path.exists(zip_path):
            return None

        samples_by_class = {}
        target_classes = ["HN", "POC", "EP", "RP", "SS", "URM"]

        with zipfile.ZipFile(zip_path, 'r') as z:
            manifest_csv = z.read("China-Offshore-SSS-AI/metadata/image_manifest.csv").decode("utf-8-sig")
            for row in csv.DictReader(io.StringIO(manifest_csv)):
                abbrev = row.get("class_abbreviation", "")
                if abbrev in target_classes and abbrev not in samples_by_class:
                    img_p = "China-Offshore-SSS-AI/" + row["release_path"]
                    if img_p in z.namelist():
                        img_bytes = z.read(img_p)
                        with Image.open(io.BytesIO(img_bytes)) as im:
                            samples_by_class[abbrev] = im.convert("RGB").resize((256, 256))
                if len(samples_by_class) == len(target_classes):
                    break

        if not samples_by_class:
            return None

        # Build 2x3 grid
        cols = 3
        rows = 2
        thumb_w, thumb_h = 256, 256
        montage = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(20, 25, 30))

        for idx, (cls_abbr, img) in enumerate(samples_by_class.items()):
            c = idx % cols
            r = idx // cols
            montage.paste(img, (c * thumb_w, r * thumb_h))

        montage.save(output_path)
        return output_path
