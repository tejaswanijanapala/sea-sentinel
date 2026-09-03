"""
Stage 4: Sonar Segmentation Dataset and Patch Tiler
Provides:
  1. SonarSegmentationDataset: PyTorch Dataset with acoustic sonar data augmentations
  2. verify_segmentation_dataset: Non-destructive verification of real masks in workspace
  3. PatchTiler: Sliding-window patch extraction and seamless blended reconstruction for large SSS mosaics
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class SonarSegmentationDataset(Dataset):
    """
    PyTorch Dataset for Side-Scan Sonar (SSS) Semantic Segmentation.
    Loads single-channel acoustic imagery and corresponding pixel-level binary/multi-class masks.
    Applies sonar-specific augmentations:
      - Horizontal flip (port/starboard swath symmetry)
      - Vertical flip (along-track progression)
      - Random 90/180/270 rotations
      - Acoustic gain & speckle contrast jitter
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        img_size: int = 256,
        augment: bool = False,
        normalize_mean_std: bool = False
    ):
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        self.mask_paths = mask_paths if mask_paths else []
        self.img_size = img_size
        self.augment = augment
        self.normalize_mean_std = normalize_mean_std

        # Map mask paths by basename without extension if provided as list
        self.mask_dict = {}
        if self.mask_paths:
            for mp in self.mask_paths:
                base = os.path.splitext(os.path.basename(mp))[0]
                self.mask_dict[base] = mp

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, path: str) -> np.ndarray:
        """
        Loads image as single-channel normalized float32 [0.0, 1.0].
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback for TIFF or non-standard format
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is None:
            raise FileNotFoundError(f"Failed to load sonar image at {path}")

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_mask(self, img_path: str) -> np.ndarray:
        """
        Loads corresponding mask if exists; otherwise returns zero mask.
        """
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = self.mask_dict.get(base)

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
                return mask

        # If no matching mask found, return empty binary canvas
        return np.zeros((self.img_size, self.img_size), dtype=np.float32)

    def _apply_augmentations(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acoustic-aware data augmentation.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random 90-degree rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # Sonar gain & contrast jitter
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.25)  # Contrast
            beta = random.uniform(-0.1, 0.1)    # Brightness
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        img = self._load_image(img_path)
        mask = self._load_mask(img_path)

        if self.augment:
            img, mask = self._apply_augmentations(img, mask)

        if self.normalize_mean_std:
            mean = np.mean(img)
            std = np.std(img) + 1e-6
            img = (img - mean) / std

        # Convert to torch tensor: [1, H, W]
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor, img_path


def verify_segmentation_dataset(
    data_dir: str,
    image_subdirs: Optional[List[str]] = None,
    mask_subdirs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Exhaustively scans the dataset directory for images and corresponding ground-truth masks.
    Enforces the strict rule: never fabricate masks and report true status.
    """
    if image_subdirs is None:
        image_subdirs = ["images", "img", "chips", "sonar"]
    if mask_subdirs is None:
        mask_subdirs = ["masks", "annotations", "labels", "gt_masks"]

    found_images = []
    found_masks = []

    for root, _, files in os.walk(data_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, data_dir).lower()
                is_mask = any(m_kw in rel_path for m_kw in ["mask", "annotation", "gt_mask", "segmentation"])
                if is_mask:
                    found_masks.append(full_path)
                else:
                    found_images.append(full_path)

    # Match images with masks by filename stem
    image_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in found_images}
    mask_stems = {os.path.splitext(os.path.basename(p))[0]: p for p in found_masks}

    paired_stems = set(image_stems.keys()).intersection(set(mask_stems.keys()))

    has_real_masks = len(paired_stems) > 0

    return {
        "status": "ready" if has_real_masks else "masks_missing",
        "data_dir": data_dir,
        "total_images_found": len(found_images),
        "total_masks_found": len(found_masks),
        "paired_samples_count": len(paired_stems),
        "has_real_masks": has_real_masks,
        "sample_images": found_images[:5],
        "sample_masks": found_masks[:5],
        "message": (
            f"Found {len(paired_stems)} verified paired image-mask samples."
            if has_real_masks
            else "No per-pixel segmentation masks exist in this directory. "
                 "The complete U-Net architecture is ready for training once ground truth masks are provided."
        )
    }


class PatchTiler:
    """
    Performs sliding-window tiling of large sonar mosaics and seamless blended stitching.
    Eliminates boundary seam artifacts using 2D cosine / Gaussian weighting across overlapping patches.
    """
    def __init__(self, patch_size: int = 256, stride: int = 192):
        self.patch_size = patch_size
        self.stride = stride

    def tile_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Extracts overlapping patches from an arbitrary-sized 2D sonar image.
        Returns:
            patches: List of [patch_size, patch_size] arrays
            coords: List of (y1, x1, y2, x2) bounding coordinates
        """
        h, w = image.shape[:2]
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        else:
            padded = image

        ph_total, pw_total = padded.shape[:2]
        patches = []
        coords = []

        y_points = list(range(0, max(1, ph_total - self.patch_size + 1), self.stride))
        if y_points[-1] + self.patch_size < ph_total:
            y_points.append(ph_total - self.patch_size)

        x_points = list(range(0, max(1, pw_total - self.patch_size + 1), self.stride))
        if x_points[-1] + self.patch_size < pw_total:
            x_points.append(pw_total - self.patch_size)

        # Remove potential duplicate points
        y_points = sorted(list(set(y_points)))
        x_points = sorted(list(set(x_points)))

        for y in y_points:
            for x in x_points:
                patch = padded[y:y + self.patch_size, x:x + self.patch_size]
                patches.append(patch)
                coords.append((y, x, y + self.patch_size, x + self.patch_size))

        return patches, coords

    def stitch_patches(
        self,
        patch_predictions: List[np.ndarray],
        coords: List[Tuple[int, int, int, int]],
        original_shape: Tuple[int, int],
        blending: str = "cosine"
    ) -> np.ndarray:
        """
        Reconstructs the full-size probability map from overlapping patch predictions
        using smooth distance-based blending to prevent edge discontinuities.
        """
        h, w = original_shape[:2]
        canvas_h = max(h, self.patch_size)
        canvas_w = max(w, self.patch_size)

        canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # Build 2D blending weight kernel
        if blending == "cosine":
            ax = np.sin(np.linspace(0, np.pi, self.patch_size))
            weight_kernel = np.outer(ax, ax).astype(np.float32)
            weight_kernel = np.maximum(weight_kernel, 1e-4)
        else:
            weight_kernel = np.ones((self.patch_size, self.patch_size), dtype=np.float32)

        for patch, (y1, x1, y2, x2) in zip(patch_predictions, coords):
            ph, pw = patch.shape[:2]
            kw = weight_kernel[:ph, :pw]

            canvas[y1:y2, x1:x2] += patch * kw
            weight_sum[y1:y2, x1:x2] += kw

        # Normalize by overlapping weights
        weight_sum = np.maximum(weight_sum, 1e-6)
        full_prob_map = canvas / weight_sum
        return full_prob_map[:h, :w]
