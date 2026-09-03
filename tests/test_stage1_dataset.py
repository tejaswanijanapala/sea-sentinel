"""
Unit Test for Stage 1 Dataset Preparation Pipeline
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_stage1_outputs():
    processed_dir = os.path.join(PROJECT_ROOT, "datasets", "processed")
    inv_file = os.path.join(processed_dir, "dataset_inventory.json")
    data_yaml = os.path.join(processed_dir, "yolo_dataset", "data.yaml")
    montage_img = os.path.join(processed_dir, "dataset_samples_montage.jpg")

    assert os.path.exists(inv_file), "dataset_inventory.json must exist"
    assert os.path.exists(data_yaml), "yolo_dataset/data.yaml must exist"
    assert os.path.exists(montage_img), "dataset_samples_montage.jpg must exist"

    with open(inv_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["stage"] == "Stage 1: Dataset Preparation"
        assert data["zenodo_inspection"]["manifest_records"] == 3255
        assert data["zenodo_inspection"]["corrupt_images"] == 0
        assert "fishing_net" in data["zenodo_inspection"]["class_distribution"]

    with open(data_yaml, "r") as f:
        content = f.read()
        assert "fishing_net" in content
        assert "pipeline_or_cable" in content

    print("test_stage1_outputs passed successfully!")

if __name__ == "__main__":
    test_stage1_outputs()
