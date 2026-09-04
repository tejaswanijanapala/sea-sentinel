"""
Batch Preprocessor for Side-Scan Sonar Imagery
Processes folders of SSS image chips or tiles with progress tracking,
metric aggregation, and composite output generation.
"""
from typing import Dict, Any, List, Optional
import os
import time
import json
import cv2
import numpy as np

from .pipeline import SonarPreprocessor

class BatchPreprocessor:
    """
    Batch processing engine for directories of SSS images.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.pipeline = SonarPreprocessor(config)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        max_images: Optional[int] = None,
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Batch processes all sonar images in an input directory.
        Outputs:
          - preprocessed/ : Enhanced 8-bit grayscale images
          - shadows/      : Extracted acoustic shadow binary masks
          - comparisons/  : Side-by-side before/after composite visualizations
        """
        os.makedirs(os.path.join(output_dir, "preprocessed"), exist_ok=True)
        if save_visualizations:
            os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "shadows"), exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_exts:
                    image_files.append(os.path.join(root, f))

        image_files.sort()
        if max_images is not None:
            image_files = image_files[:max_images]

        results = []
        total_enl_gain = 0.0
        total_contrast_gain = 0.0
        start_time = time.time()

        for idx, img_path in enumerate(image_files):
            try:
                t0 = time.time()
                res = self.pipeline.process(img_path)
                elapsed_ms = (time.time() - t0) * 1000.0

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"{base_name}_prep.png"
                proc_img = res["preprocessed_image"]
                raw_img = res["original_image"]
                shadow_mask = res["shadow_mask"]

                # Save preprocessed image
                save_path = os.path.join(output_dir, "preprocessed", out_name)
                cv2.imwrite(save_path, proc_img)

                if save_visualizations:
                    # Save shadow mask
                    cv2.imwrite(os.path.join(output_dir, "shadows", f"{base_name}_shadow.png"), shadow_mask)

                    # Create side-by-side comparison (Raw vs Processed vs Shadow Mask)
                    # Resize to common height if needed
                    h, w = raw_img.shape[:2]
                    comparison = np.hstack([raw_img, proc_img, shadow_mask])
                    cv2.imwrite(os.path.join(output_dir, "comparisons", f"{base_name}_compare.png"), comparison)

                metrics = res["metrics"]
                total_enl_gain += metrics.get("enl_improvement_factor", 1.0)
                total_contrast_gain += metrics.get("contrast_gain_factor", 1.0)

                results.append({
                    "image": os.path.basename(img_path),
                    "status": "success",
                    "latency_ms": round(elapsed_ms, 2),
                    "enl_improvement": metrics.get("enl_improvement_factor"),
                    "contrast_gain": metrics.get("contrast_gain_factor")
                })
            except Exception as e:
                results.append({
                    "image": os.path.basename(img_path),
                    "status": "error",
                    "error": str(e)
                })

        total_elapsed = time.time() - start_time
        num_processed = len([r for r in results if r["status"] == "success"])

        summary = {
            "total_scanned": len(image_files),
            "successfully_processed": num_processed,
            "failed": len(image_files) - num_processed,
            "total_runtime_seconds": round(total_elapsed, 2),
            "avg_latency_ms_per_image": round((total_elapsed / max(1, num_processed)) * 1000.0, 2),
            "avg_enl_improvement": round(total_enl_gain / max(1, num_processed), 3) if num_processed else 0.0,
            "avg_contrast_gain": round(total_contrast_gain / max(1, num_processed), 3) if num_processed else 0.0,
            "results": results
        }

        # Save summary report JSON
        with open(os.path.join(output_dir, "batch_preprocessing_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
