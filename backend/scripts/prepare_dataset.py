"""
Stage 1 Execution Script: Dataset Preparation & Validation
Runs dataset inspection, formats conversion, creates splits, generates visualizations,
and exports full inventory.
"""
import sys
import os
import json

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.dataset.pipeline import DatasetPreparationPipeline

def main():
    print("=" * 70)
    print("SIH26057 — STAGE 1: DATASET PREPARATION & INSPECTION")
    print("=" * 70)

    # 1. Paths
    raw_dir = r"C:\Users\santo\Downloads"
    zenodo_zip = os.path.join(raw_dir, "China-Offshore-SSS-AI_Zenodo_public_upload.zip")
    processed_dir = os.path.join(PROJECT_ROOT, "datasets", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    pipeline = DatasetPreparationPipeline(raw_data_dir=raw_dir, processed_dir=processed_dir)

    # 2. Raw File Audit
    print("\n[1/5] Auditing Raw Datasets & Mosaics in Downloads...")
    raw_audit = pipeline.audit_all_sources()
    print(f"  Total raw files scanned: {raw_audit['total_raw_files']}")
    print(f"  Found {len(raw_audit['zip_archives'])} zip archives:")
    for z in raw_audit["zip_archives"]:
        print(f"    - {z['filename']} ({z['size_mb']} MB)")
    print(f"  Found {len(raw_audit['geotiff_mosaics'])} GeoTIFF mosaics:")
    for g in raw_audit["geotiff_mosaics"]:
        print(f"    - {g['filename']} ({g['size_mb']} MB)")
    print(f"  Found {len(raw_audit['waterfall_images'])} SSS waterfall images:")
    for w in raw_audit["waterfall_images"]:
        print(f"    - {w['filename']} ({w['size_mb']} MB)")

    # 3. Deep-Inspect Zenodo SSS-AI Archive
    print("\n[2/5] Deep-Inspecting China-Offshore-SSS-AI Dataset...")
    zenodo_report = pipeline.inspect_zenodo_archive(zenodo_zip, max_check=500)
    print(f"  Manifest records: {zenodo_report['manifest_records']}")
    print(f"  Sample verified images: {zenodo_report['verified_images']}")
    print(f"  Corrupt/unreadable images: {zenodo_report['corrupt_images']}")
    print(f"  Formats found: {zenodo_report['formats']}")
    print(f"  Resolution range: W: {zenodo_report['dimensions_min_max']['min_w']} - {zenodo_report['dimensions_min_max']['max_w']} px, H: {zenodo_report['dimensions_min_max']['min_h']} - {zenodo_report['dimensions_min_max']['max_h']} px")
    print("  Class distribution (harmonized):")
    for cls_name, count in zenodo_report["class_distribution"].items():
        print(f"    - {cls_name}: {count} images")
    print(f"  Splits assigned in manifest: {zenodo_report['splits']}")
    print(f"  Duplicate hashes detected: {zenodo_report['duplicate_hashes_found']}")
    print(f"  Pre-existing YOLO annotations present: {zenodo_report['yolo_annotations_found']}")
    print(f"  Pre-existing segmentation masks present: {zenodo_report['segmentation_masks_found']}")

    # 4. Standardized Subsets & Annotations Generation
    print("\n[3/5] Building Standardized YOLO & Anomaly Subsets...")
    subset_res = pipeline.build_standardized_subsets(zenodo_zip, sample_per_class=40)
    print(f"  YOLO samples extracted: {subset_res['yolo_samples_extracted']}")
    print(f"  Anomaly baseline samples extracted: {subset_res['anomaly_samples_extracted']}")
    print(f"  Created YOLO dataset config: {subset_res.get('data_yaml')}")

    # 5. Visual Montage Generation
    print("\n[4/5] Generating Visual Contact-Sheet Summary...")
    montage_path = os.path.join(processed_dir, "dataset_samples_montage.jpg")
    saved_montage = pipeline.generate_visual_montage(zenodo_zip, montage_path)
    if saved_montage:
        print(f"  Visual montage generated: {saved_montage}")
    else:
        print("  Could not generate montage.")

    # 6. Export Full Inventory JSON
    print("\n[5/5] Exporting Full Dataset Inventory Report...")
    inventory_data = {
        "project": "SIH26057",
        "stage": "Stage 1: Dataset Preparation",
        "raw_audit": raw_audit,
        "zenodo_inspection": zenodo_report,
        "processed_subsets": subset_res,
        "missing_requirements": {
            "yolo_native_coordinates": "Zenodo provides chip-level target crops; standardized centered YOLO bboxes generated.",
            "segmentation_masks": "No per-pixel segmentation masks exist in available public releases; U-Net pipeline will be established, and training requires real masks."
        },
        "recommended_next_step": "Stage 2: SSS Image Preprocessing (Speckle filtering, CLAHE, and tiling for high-resolution mosaics)."
    }

    inv_path = os.path.join(processed_dir, "dataset_inventory.json")
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2)
    print(f"  Inventory report saved to: {inv_path}")

    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
