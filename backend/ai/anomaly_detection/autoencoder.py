"""
Layer 4: Anomaly Detection & False Positive Filtering Core
Implements Algorithms 1-9:
  Algorithm 1: Input Sonar Image Patch
  Algorithm 2: CNN Encoder
  Algorithm 3: Latent Representation (bottleneck z)
  Algorithm 4: CNN Decoder
  Algorithm 5: Reconstructed Image Patch
  Algorithm 6: Reconstruction Difference Map
  Algorithm 7: Mean Squared Error (MSE)
  Algorithm 8: Threshold Comparator
  Algorithm 9: Anomaly Classifier
Synthesizes Autoencoder reconstruction error, acoustic shadow-highlight pairing,
and DBSCAN rock cluster filtering into calibrated confidence scoring.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import cv2
import numpy as np
import torch

from ai.anomaly_detection.models import AcousticAutoencoder
from ai.anomaly_detection.shadow_verifier import AcousticShadowVerifier
from ai.anomaly_detection.rock_cluster_filter import DBSCANRockFilter
from ai.anomaly_detection.calibrator import ConfidenceCalibrator


class AnomalyDetector:
    """
    Distinguishes natural seabed topology (sand ripples, mud, rock clusters)
    from anomalous anthropogenic debris via Autoencoder reconstruction error and acoustic heuristics.
    """
    def __init__(
        self,
        threshold: float = 0.025,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        patch_size: int = 128
    ):
        self.threshold = threshold
        self.checkpoint_path = checkpoint_path
        self.patch_size = patch_size

        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.is_model_loaded = False

        self.shadow_verifier = AcousticShadowVerifier()
        self.rock_filter = DBSCANRockFilter()
        self.calibrator = ConfidenceCalibrator()

        self._check_model()

    def _check_model(self) -> bool:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                self.model = AcousticAutoencoder(in_channels=1, latent_dim=128, base_channels=32)
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    if "threshold" in ckpt:
                        self.threshold = ckpt["threshold"]
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                else:
                    self.model = ckpt

                self.model.to(self.device)
                self.model.eval()
                self.is_model_loaded = True
                return True
            except Exception as e:
                print(f"[AnomalyDetector] Warning: Could not load checkpoint from {self.checkpoint_path}: {e}")
                self.is_model_loaded = False
                return False
        else:
            self.is_model_loaded = False
            return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        self.checkpoint_path = checkpoint_path
        return self._check_model()

    # Algorithm 1: Preprocess Input Image Patch
    def prepare_patch(self, patch: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """Algorithm 1: Prepares standardized grayscale float32 patch [1, 1, 128, 128]."""
        if patch.ndim == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch.copy()

        resized = cv2.resize(gray, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        return tensor, norm

    # Algorithms 2-5: Forward Reconstruction
    def reconstruct_patch(self, patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes Algorithms 1 through 7:
          Algorithm 1: Input patch
          Algorithm 2: CNN Encoder
          Algorithm 3: Latent code z
          Algorithm 4: CNN Decoder
          Algorithm 5: Reconstructed patch
          Algorithm 6: Difference map
          Algorithm 7: Mean Squared Error (MSE)
        """
        tensor, original_norm = self.prepare_patch(patch)

        if not self.is_model_loaded or self.model is None:
            # Operational baseline fallback: high-pass structural variance
            reconstructed_norm = cv2.GaussianBlur(original_norm, (9, 9), 0)
            diff_map = np.abs(original_norm - reconstructed_norm)
            mse = float(np.mean((original_norm - reconstructed_norm) ** 2))
            return original_norm, reconstructed_norm, mse

        with torch.no_grad():
            x_recon_tensor = self.model(tensor)
            recon_norm = x_recon_tensor.squeeze().cpu().numpy()

        diff_map = np.abs(original_norm - recon_norm)
        mse = float(np.mean((original_norm - recon_norm) ** 2))
        return original_norm, recon_norm, mse

    # Algorithm 6: Difference Map
    def compute_difference_map(self, original_patch: np.ndarray, reconstructed_patch: np.ndarray) -> np.ndarray:
        """Algorithm 6: |x - x'|"""
        return np.abs(original_patch.astype(np.float32) - reconstructed_patch.astype(np.float32))

    # Algorithm 7: Mean Squared Error
    def compute_reconstruction_error(self, original_patch: Any, reconstructed_patch: Any) -> float:
        """
        Algorithm 7: error = (1/N) * sum((x_i - x'_i)^2) [MSE]
        """
        if original_patch is None or reconstructed_patch is None:
            return 0.0
        orig = np.array(original_patch, dtype=np.float32)
        recon = np.array(reconstructed_patch, dtype=np.float32)
        if orig.shape != recon.shape:
            recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
        mse = float(np.mean((orig - recon) ** 2))
        return mse

    # Algorithms 8 & 9: Threshold Comparison and Anomaly Classification
    def classify_anomaly(self, mse: float) -> Tuple[bool, str]:
        """
        Algorithm 8: Compare MSE with baseline threshold T
        Algorithm 9: Return Anomaly classification
        """
        is_anomaly = mse > self.threshold
        classification = "ANOMALY_DEBRIS" if is_anomaly else "NORMAL_SEAFLOOR"
        return is_anomaly, classification

    def evaluate_detection(self, detection: Dict[str, Any], image_context: Any = None) -> Dict[str, Any]:
        """
        Synthesizes Algorithms 1-9, Acoustic Shadow Verification, and Confidence Calibration.
        Preserves backward compatibility with orchestrator and test_skeletons.py.
        """
        raw_conf = float(detection.get("confidence", 0.5))
        obj_id = detection.get("object_id", "OBJ_UNKNOWN")

        # 1. Evaluate crop with Autoencoder if image context is provided
        reconstruction_error = 0.0
        is_anomaly = raw_conf > self.threshold
        anomaly_score = min(1.0, raw_conf)

        if image_context is not None and isinstance(image_context, np.ndarray) and "bbox" in detection:
            bbox = detection["bbox"]
            h_img, w_img = image_context.shape[:2]
            x1 = max(0, min(w_img - 1, int(bbox.get("x1", 0))))
            y1 = max(0, min(h_img - 1, int(bbox.get("y1", 0))))
            x2 = max(x1 + 1, min(w_img, int(bbox.get("x2", w_img))))
            y2 = max(y1 + 1, min(h_img, int(bbox.get("y2", h_img))))

            patch_crop = image_context[y1:y2, x1:x2]
            if patch_crop.size > 0:
                _, _, mse = self.reconstruct_patch(patch_crop)
                reconstruction_error = mse
                is_anomaly, _ = self.classify_anomaly(mse)
                anomaly_score = min(1.0, mse / max(self.threshold, 1e-4))

                # Acoustic shadow verification
                shadow_res = self.shadow_verifier.verify_patch(patch_crop)
                shadow_verified = shadow_res["shadow_verified"]
                shadow_score = shadow_res["pairing_score"]
            else:
                shadow_verified = True
                shadow_score = 0.5
        else:
            shadow_verified = True
            shadow_score = 0.6 if raw_conf >= 0.5 else 0.3

        # 2. DBSCAN rock cluster penalty if provided in detection
        rock_penalty = float(detection.get("rock_density_penalty", 0.0))

        # 3. Calibrated Confidence Scoring
        cal_res = self.calibrator.calibrate(
            raw_confidence=raw_conf,
            anomaly_score=anomaly_score,
            shadow_score=shadow_score,
            rock_penalty=rock_penalty
        )

        return {
            "object_id": obj_id,
            "raw_confidence": raw_conf,
            "calibrated_confidence": cal_res["calibrated_confidence"],
            "status": cal_res["status"],
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": round(float(reconstruction_error), 6),
            "shadow_verified": bool(shadow_verified),
            "shadow_score": round(float(shadow_score), 4),
            "rock_penalty": round(float(rock_penalty), 4)
        }
