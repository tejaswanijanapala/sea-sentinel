"""
Stage 2 Execution Script: Sonar Preprocessing & Enhancement
Runs configurable preprocessing, Lee speckle filtering, CLAHE contrast enhancement,
acoustic shadow extraction, and batch evaluation on real SSS imagery.
"""
import sys
import os
import json
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.preprocessing.pipeline import SonarPreprocessor
from ai.preprocessing.batch_processor import BatchPreprocessor

def main():
    print("=" * 70)
    print("SIH26057 — STAGE 2: SSS IMAGE PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load config
    cfg_path = os.path.join(PROJECT_ROOT, "configs", "preprocessing_config.yaml")
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
            config = raw_cfg.get("preprocessing", {})
            print(f"[1/4] Loaded preprocessing configuration from {cfg_path}")

    preprocessor = SonarPreprocessor(config)
    batch_proc = BatchPreprocessor(config)

    # 2. Setup outputs
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Process Sample Real SSS Chips (from Stage 1)
    input_chips_dir = os.path.join(PROJECT_ROOT, "datasets", "processed", "yolo_dataset", "images", "train")
    print(f"\n[2/4] Running Batch Preprocessing on SSS Debris Chips ({input_chips_dir})...")
    batch_summary = batch_proc.process_directory(
        input_dir=input_chips_dir,
        output_dir=output_dir,
        max_images=20,
        save_visualizations=True
    )
    print(f"  Processed: {batch_summary['successfully_processed']} images in {batch_summary['total_runtime_seconds']}s")
    print(f"  Average latency: {batch_summary['avg_latency_ms_per_image']} ms/image")
    print(f"  Average Equivalent Number of Looks (ENL) gain: {batch_summary['avg_enl_improvement']}x (speckle reduction)")
    print(f"  Average contrast gain: {batch_summary['avg_contrast_gain']}x")

    # 4. Process Monrovia AUV SSS Waterfall if available
    monrovia_path = r"C:\Users\santo\Downloads\monrovia-side-scan-sonar-IVER-hires.png"
    if os.path.exists(monrovia_path):
        print(f"\n[3/4] Processing High-Resolution Monrovia AUV SSS Waterfall Scan...")
        m_res = preprocessor.process(monrovia_path)
        m_out_path = os.path.join(output_dir, "monrovia_preprocessed.png")
        import cv2
        import numpy as np
        cv2.imwrite(m_out_path, m_res["preprocessed_image"])

        # Tiling test for large waterfall
        tiles = preprocessor.tile_mosaic(m_res["preprocessed_image"], tile_size=(640, 640), overlap=128)
        print(f"  Generated {len(tiles)} overlapping 640x640 inference tiles from Monrovia waterfall")
        print(f"  Monrovia preprocessed saved to: {m_out_path}")

    # 5. Summary
    print("\n[4/4] Output Visualizations & Reports:")
    print(f"  Side-by-side comparison images: {os.path.join(output_dir, 'comparisons')}")
    print(f"  Extracted acoustic shadow masks: {os.path.join(output_dir, 'shadows')}")
    print(f"  Summary report: {os.path.join(output_dir, 'batch_preprocessing_summary.json')}")

    print("\n" + "=" * 70)
    print("STAGE 2 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
