"""
Unit Tests for Sonar-Aware Confidence Feature
Validates all requirements:
1. Detection region extraction (U-Net mask vs YOLO bbox fallback)
2. Acoustic shadow feature extraction (with and without shadow)
3. Shape feature extraction (aspect ratio, compactness, solidity, edge roughness)
4. Texture feature extraction (GLCM contrast, energy, homogeneity, mean, std)
5. Image-quality feature extraction (SNR, speckle, dropout, sharpness, contrast, NaN handling)
6. Sonar metadata feature extraction (with metadata and with missing metadata)
7. Feature vector fixed schema and dimension consistency
8. Calibration model & scaler loader (handling available vs unavailable model)
9. Sonar-Aware Confidence calculation & preservation of original YOLO confidence
"""

import os
import sys
import unittest
import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from inference.sonar_aware_confidence import (
    FEATURE_NAMES,
    extract_detection_region,
    extract_shadow_features,
    extract_shape_features,
    extract_texture_features,
    extract_quality_features,
    extract_metadata_features,
    build_feature_vector,
    CalibrationModelLoader,
    calculate_sonar_aware_confidence
)


class TestSonarAwareConfidence(unittest.TestCase):
    def setUp(self):
        # Create a synthetic side-scan sonar image (300 x 400)
        np.random.seed(42)
        self.sonar_img = np.random.normal(90.0, 15.0, (300, 400)).astype(np.float32)
        self.sonar_img = np.clip(self.sonar_img, 0.0, 255.0).astype(np.uint8)

        # Draw a synthetic bright acoustic highlight (50, 50 to 90, 110)
        self.sonar_img[50:90, 50:110] = 220

        # Draw an adjacent dark acoustic shadow to the right (50, 110 to 90, 160)
        self.sonar_img[50:90, 110:160] = 20

        # Detection dict
        self.detection = {
            "object_id": "TGT_001",
            "class": "pipeline_cable",
            "confidence": 0.73,  # 73% YOLO11 confidence
            "bbox": {"x1": 50, "y1": 50, "x2": 110, "y2": 90}
        }

        # U-Net segmentation mask
        self.unet_mask = np.zeros((300, 400), dtype=np.uint8)
        self.unet_mask[55:85, 55:105] = 1

        # Calibration model loader
        self.calib_path = os.path.join(BACKEND_DIR, "models", "checkpoints", "sonar_confidence_calibrator.joblib")
        self.calibrator = CalibrationModelLoader()
        if not os.path.exists(self.calib_path):
            self.calibrator.train_and_save_default_model(self.calib_path)
        else:
            self.calibrator.load(self.calib_path)

    def test_01_detection_region_with_unet_mask(self):
        """Verify U-Net mask is used when available."""
        res = extract_detection_region(self.sonar_img, self.detection, unet_mask=self.unet_mask)
        self.assertEqual(res["mask_source"], "unet_mask")
        self.assertEqual(res["crop"].shape, (40, 60))
        self.assertEqual(res["mask"].shape, (40, 60))
        self.assertAlmostEqual(res["yolo_confidence"], 0.73, places=2)
        self.assertEqual(res["object_class"], "pipeline_cable")

    def test_02_detection_region_without_unet_mask(self):
        """Verify YOLO bounding box is used when U-Net mask is unavailable."""
        res = extract_detection_region(self.sonar_img, self.detection, unet_mask=None)
        self.assertEqual(res["mask_source"], "yolo_bbox")
        self.assertEqual(res["mask"].shape, (40, 60))
        self.assertTrue(np.all(res["mask"] == 1))

    def test_03_acoustic_shadow_features_with_shadow(self):
        """Verify acoustic shadow features when a genuine acoustic shadow exists."""
        bbox = [50, 50, 110, 90]
        meta = {"swath_side": "starboard", "nadir_x": 0.0}
        shadow = extract_shadow_features(self.sonar_img, bbox, sonar_metadata=meta)

        self.assertEqual(shadow["shadow_available"], 1.0)
        self.assertGreater(shadow["shadow_length"], 0.0)
        self.assertGreater(shadow["shadow_contrast"], 0.3)
        self.assertGreater(shadow["shadow_completeness"], 0.0)

    def test_04_acoustic_shadow_features_without_shadow(self):
        """Verify acoustic shadow features when no shadow is present (do not invent shadow)."""
        flat_img = np.full((300, 400), 100, dtype=np.uint8)
        flat_img[50:90, 50:110] = 120  # Slight highlight, no shadow
        bbox = [50, 50, 110, 90]
        shadow = extract_shadow_features(flat_img, bbox)

        self.assertEqual(shadow["shadow_available"], 0.0)
        self.assertEqual(shadow["shadow_length"], 0.0)
        self.assertEqual(shadow["shadow_contrast"], 0.0)
        self.assertEqual(shadow["shadow_completeness"], 0.0)

    def test_05_shape_features(self):
        """Verify shape features (aspect ratio, compactness, solidity, roughness)."""
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[5:35, 5:55] = 1
        bbox = [50, 50, 110, 90]
        shape = extract_shape_features(mask, bbox)

        self.assertGreaterEqual(shape["aspect_ratio"], 1.0)
        self.assertGreaterEqual(shape["compactness"], 0.0)
        self.assertLessEqual(shape["compactness"], 1.0)
        self.assertGreaterEqual(shape["solidity"], 0.0)
        self.assertLessEqual(shape["solidity"], 1.0)
        self.assertGreaterEqual(shape["edge_roughness"], 0.0)
        self.assertLessEqual(shape["edge_roughness"], 1.0)

    def test_06_texture_features_glcm(self):
        """Verify GLCM texture features work on sonar image crops."""
        crop = self.sonar_img[50:90, 50:110]
        texture = extract_texture_features(crop)

        self.assertIn("glcm_contrast", texture)
        self.assertIn("glcm_energy", texture)
        self.assertIn("glcm_homogeneity", texture)
        self.assertIn("mean_intensity", texture)
        self.assertIn("std_intensity", texture)

        self.assertGreaterEqual(texture["glcm_energy"], 0.0)
        self.assertLessEqual(texture["glcm_energy"], 1.0)
        self.assertGreaterEqual(texture["glcm_homogeneity"], 0.0)
        self.assertLessEqual(texture["glcm_homogeneity"], 1.0)

    def test_07_quality_features_and_nan_safety(self):
        """Verify image quality features handle NaNs, Infs, and invalid pixels safely."""
        crop_with_nan = np.array([
            [np.nan, 50.0, 120.0],
            [255.0, 0.0, np.inf],
            [80.0, 90.0, 100.0]
        ], dtype=np.float32)

        quality = extract_quality_features(crop_with_nan)
        self.assertFalse(np.isnan(quality["local_snr"]))
        self.assertFalse(np.isnan(quality["speckle_level"]))
        self.assertFalse(np.isnan(quality["dropout_fraction"]))
        self.assertFalse(np.isnan(quality["sharpness"]))
        self.assertFalse(np.isnan(quality["local_contrast"]))

    def test_08_metadata_features(self):
        """Verify sonar metadata extraction with and without metadata."""
        meta_present = {
            "slant_range": 45.5,
            "altitude": 12.0,
            "grazing_angle": 28.5,
            "pitch": 1.2,
            "roll": -0.8,
            "heave": 0.15
        }
        res_present = extract_metadata_features(meta_present)
        self.assertEqual(res_present["metadata_available"], 1.0)
        self.assertEqual(res_present["slant_range"], 45.5)
        self.assertEqual(res_present["altitude"], 12.0)

        # Missing metadata
        res_missing = extract_metadata_features(None)
        self.assertEqual(res_missing["metadata_available"], 0.0)
        self.assertEqual(res_missing["slant_range"], 0.0)
        self.assertEqual(res_missing["altitude"], 0.0)

    def test_09_feature_vector_schema_and_dimension(self):
        """Verify feature vector length and exact schema alignment."""
        shadow = {"shadow_available": 1.0, "shadow_length": 25.0, "shadow_contrast": 0.6, "shadow_completeness": 0.8}
        shape = {"aspect_ratio": 2.1, "compactness": 0.7, "solidity": 0.9, "edge_roughness": 0.1}
        texture = {"glcm_contrast": 2.5, "glcm_energy": 0.35, "glcm_homogeneity": 0.65, "mean_intensity": 120.0, "std_intensity": 25.0}
        quality = {"local_snr": 3.2, "speckle_level": 0.25, "dropout_fraction": 0.01, "sharpness": 65.0, "local_contrast": 0.7}
        meta = {"slant_range": 50.0, "altitude": 15.0, "grazing_angle": 30.0, "pitch": 0.0, "roll": 0.0, "heave": 0.0, "metadata_available": 1.0}

        vec = build_feature_vector(
            yolo_confidence=0.73,
            shadow_features=shadow,
            shape_features=shape,
            texture_features=texture,
            quality_features=quality,
            metadata_features=meta
        )

        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertEqual(vec[0], 0.73)  # YOLO confidence at index 0

    def test_10_sonar_aware_confidence_with_calibration_model(self):
        """Verify Sonar-Aware Confidence calculation when calibration model is available."""
        self.assertTrue(self.calibrator.is_available())

        res = calculate_sonar_aware_confidence(
            image=self.sonar_img,
            detection=self.detection,
            unet_mask=self.unet_mask,
            sonar_metadata={"slant_range": 40.0, "altitude": 10.0},
            calibration_loader=self.calibrator
        )

        self.assertEqual(res["confidence_status"], "calibrated")
        self.assertEqual(res["yolo_confidence"], 0.73)
        self.assertEqual(res["yolo_confidence_pct"], 73.0)
        self.assertIsNotNone(res["sonar_aware_confidence"])
        self.assertGreaterEqual(res["sonar_aware_confidence"], 0.0)
        self.assertLessEqual(res["sonar_aware_confidence"], 100.0)
        self.assertEqual(res["object_class"], "pipeline_cable")

    def test_11_sonar_aware_confidence_without_calibration_model(self):
        """Verify graceful fallback when calibration model is not available."""
        empty_loader = CalibrationModelLoader()
        self.assertFalse(empty_loader.is_available())

        res = calculate_sonar_aware_confidence(
            image=self.sonar_img,
            detection=self.detection,
            unet_mask=self.unet_mask,
            sonar_metadata=None,
            calibration_loader=empty_loader
        )

        self.assertEqual(res["confidence_status"], "calibration_model_not_available")
        self.assertEqual(res["yolo_confidence"], 0.73)
        self.assertEqual(res["yolo_confidence_pct"], 73.0)
        self.assertIsNone(res["sonar_aware_confidence"])


if __name__ == "__main__":
    unittest.main()
