"""
Layer 3b: U-Net Semantic Segmentation Core
Provides high-resolution pixel-level segmentation of irregular anthropogenic marine debris
(lost fishing nets, cables, pipelines, shipwreck fragments) from Side-Scan Sonar (SSS) imagery.
Supports both standard U-Net and Attention U-Net architectures.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import os
import cv2
import numpy as np
import torch

from ai.segmentation.models import build_unet, AttentionUNet, UNet
from ai.segmentation.dataset import PatchTiler


class UNetSegmenter:
    """
    Production-grade Semantic Segmentation engine using U-Net / Attention U-Net
    for acoustic Side-Scan Sonar pixel-level debris isolation and contour profiling.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "attention_unet",
        features: Optional[List[int]] = None,
        device: Optional[str] = None,
        img_size: int = 256,
        confidence_threshold: float = 0.5
    ):
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.features = features
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # Device selection
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.is_model_loaded = False
        self.tiler = PatchTiler(patch_size=self.img_size, stride=int(self.img_size * 0.75))

        if self.checkpoint_path:
            self._load_model()

    def _load_model(self) -> bool:
        """
        Loads trained checkpoint weights if available.
        """
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            self.is_model_loaded = False
            return False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Determine features and model_type from checkpoint if stored
            state_dict = None
            features = self.features
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if "model_type" in checkpoint and not self.model_type:
                    self.model_type = checkpoint["model_type"]
                if "features" in checkpoint and features is None:
                    features = checkpoint["features"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                self.model = checkpoint
                self.model.to(self.device)
                self.model.eval()
                self.is_model_loaded = True
                return True

            # Auto-detect base features if not specified
            if features is None and state_dict is not None:
                if "inc.double_conv.0.weight" in state_dict:
                    f0 = state_dict["inc.double_conv.0.weight"].shape[0]
                    features = [f0, f0 * 2, f0 * 4, f0 * 8]

            self.model = build_unet(
                model_type=self.model_type,
                in_channels=1,
                num_classes=1,
                features=features
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_model_loaded = True
            return True
        except Exception as e:
            print(f"[UNetSegmenter] Warning: Failed to load checkpoint from {self.checkpoint_path}: {e}")
            self.is_model_loaded = False
            return False

    def load_checkpoint(self, checkpoint_path: str, model_type: Optional[str] = None) -> bool:
        """
        Explicitly loads weights from a specific path.
        """
        self.checkpoint_path = checkpoint_path
        if model_type:
            self.model_type = model_type
        return self._load_model()

    def segment_roi(
        self,
        image_patch: Union[np.ndarray, None],
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs pixel-level segmentation on candidate detection ROI patch (e.g. from YOLO crop).
        """
        if not self.is_model_loaded:
            return {
                "status": "model_unavailable",
                "message": "Trained U-Net weights not found. Real segmentation masks required for training in Stage 4.",
                "mask_available": False,
                "mask": None
            }

        if image_patch is None or not isinstance(image_patch, np.ndarray) or image_patch.size == 0:
            return {
                "status": "error",
                "message": "Invalid or empty image patch provided.",
                "mask_available": False,
                "mask": None
            }

        thresh = threshold if threshold is not None else self.confidence_threshold
        h_orig, w_orig = image_patch.shape[:2]

        # Convert to grayscale float [0, 1]
        if image_patch.ndim == 3:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_patch.copy()

        resized = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        norm_img = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize probability back to original ROI dimensions
        orig_prob = cv2.resize(probs, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        binary_mask = (orig_prob >= thresh).astype(np.uint8)

        # Extract contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = int(np.sum(binary_mask))
        mean_conf = float(np.mean(orig_prob[binary_mask > 0])) if total_area > 0 else 0.0

        return {
            "status": "success",
            "mask_available": True,
            "mask": binary_mask,
            "probability_map": orig_prob,
            "contours_count": len(contours),
            "total_area_px": total_area,
            "mean_confidence": mean_conf,
            "model_type": self.model_type
        }

    def segment_full_image(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs seamless patch-based segmentation across large sonar image/mosaic.
        """
        if not self.is_model_loaded:
            return {
                "status": "model_unavailable",
                "message": "Trained U-Net weights not found. Real segmentation masks required for training in Stage 4.",
                "mask_available": False,
                "mask": None
            }

        thresh = threshold if threshold is not None else self.confidence_threshold
        h, w = image.shape[:2]

        # Ensure single channel grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Extract overlapping patches
        patches, coords = self.tiler.tile_image(gray)
        patch_predictions = []

        with torch.no_grad():
            for p in patches:
                p_norm = (p.astype(np.float32) / 255.0)
                tensor = torch.from_numpy(p_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                logits = self.model(tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                patch_predictions.append(probs)

        # Stitch with smooth cosine blending
        full_prob = self.tiler.stitch_patches(
            patch_predictions=patch_predictions,
            coords=coords,
            original_shape=(h, w),
            blending="cosine"
        )
        binary_mask = (full_prob >= thresh).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = int(np.sum(binary_mask))

        return {
            "status": "success",
            "mask_available": True,
            "mask": binary_mask,
            "probability_map": full_prob,
            "contours_count": len(contours),
            "total_area_px": total_area,
            "model_type": self.model_type
        }

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        alpha: float = 0.45,
        draw_contours: bool = True
    ) -> np.ndarray:
        """
        Creates a high-contrast sonar overlay with semi-transparent mask and crisp contour boundaries.
        Args:
            image: Sonar image (grayscale or BGR)
            mask: Binary mask [H, W] with values in {0, 1} or {0, 255}
            color: BGR tuple for mask overlay (default: neon cyan/yellow)
            alpha: Transparency factor
            draw_contours: Whether to outline debris boundaries
        """
        if image.ndim == 2:
            base_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = image.copy()

        bin_mask = (mask > 0).astype(np.uint8)
        
        # Smooth alpha blend with numpy
        blended = base_bgr.astype(np.float32)
        mask_idx = (bin_mask > 0)
        if np.any(mask_idx):
            color_arr = np.array(color, dtype=np.float32)
            blended[mask_idx] = (1.0 - alpha) * blended[mask_idx] + alpha * color_arr
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if draw_contours and np.any(mask_idx):
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, color, 2, lineType=cv2.LINE_AA)

        return blended
