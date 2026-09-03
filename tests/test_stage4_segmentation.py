"""
Stage 4 Unit Test Suite: U-Net Semantic Segmentation
Validates:
  1. Standard U-Net and Attention U-Net architectures
  2. Attention Gate activation maps and dimensions
  3. Loss functions (Dice, BCEDice, Focal) and gradient propagation
  4. Metric computation (IoU, Dice, Precision, Recall, Specificity)
  5. PatchTiler sliding window and blended reconstruction
  6. SonarSegmentationDataset loading and augmentations
  7. UNetSegmenter operational states (model_unavailable vs ready)
  8. Non-destructive dataset verification audit
"""

import os
import sys
import tempfile
import numpy as np
import cv2
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.segmentation.models import UNet, AttentionUNet, AttentionGate, build_unet
from ai.segmentation.losses import DiceLoss, BCEDiceLoss, FocalLoss, compute_iou, compute_dice, compute_pixel_metrics
from ai.segmentation.dataset import SonarSegmentationDataset, PatchTiler, verify_segmentation_dataset
from ai.segmentation.unet_segmenter import UNetSegmenter


def test_standard_unet():
    """Test standard U-Net architecture input/output shapes."""
    model = UNet(in_channels=1, num_classes=1, features=[16, 32, 64, 128])
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"
    assert model.count_parameters() > 0


def test_attention_unet():
    """Test Attention U-Net architecture and attention coefficient maps."""
    model = AttentionUNet(in_channels=1, num_classes=1, features=[16, 32, 64, 128])
    x = torch.randn(2, 1, 128, 128)
    
    # Standard forward
    out = model(x)
    assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128), got {out.shape}"

    # Forward with attention maps
    out, attns = model(x, return_attention=True)
    assert len(attns) == 4
    for a in attns:
        assert torch.all(a >= 0.0) and torch.all(a <= 1.0), "Attention coefficients must be bounded in [0, 1]"


def test_attention_gate():
    """Test individual Attention Gate mechanism."""
    gate = AttentionGate(f_g=64, f_l=32, f_int=16)
    g = torch.randn(2, 64, 16, 16)  # Gating signal from coarser decoder
    x = torch.randn(2, 32, 32, 32)  # Skip features from finer encoder
    
    attended_x, alpha = gate(g, x)
    assert attended_x.shape == (2, 32, 32, 32)
    assert alpha.shape == (2, 1, 32, 32)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)


def test_losses_and_gradients():
    """Test DiceLoss, BCEDiceLoss, and FocalLoss calculation and gradient propagation."""
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()

    # 1. BCEDiceLoss
    criterion_bce_dice = BCEDiceLoss(w_bce=0.5, w_dice=0.5)
    loss = criterion_bce_dice(logits, target)
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None

    # 2. DiceLoss with perfect match
    dice_loss = DiceLoss(from_logits=True)
    high_logits = torch.ones(1, 1, 32, 32) * 20.0  # Sigmoid ~ 1.0
    ones_target = torch.ones(1, 1, 32, 32)
    perfect_loss = dice_loss(high_logits, ones_target)
    assert perfect_loss.item() < 0.05, f"Expected near-zero loss for perfect match, got {perfect_loss.item()}"

    # 3. FocalLoss
    focal = FocalLoss(from_logits=True)
    f_loss = focal(torch.randn(2, 1, 32, 32), target[:2, :, :32, :32])
    assert f_loss.item() > 0


def test_metrics():
    """Test IoU, Dice, and pixel classification metrics."""
    # Identical masks
    a = torch.ones(1, 1, 64, 64)
    b = torch.ones(1, 1, 64, 64)
    assert abs(compute_iou(a, b) - 1.0) < 1e-4
    assert abs(compute_dice(a, b) - 1.0) < 1e-4

    # Half overlap
    a = torch.zeros(1, 1, 10, 10)
    b = torch.zeros(1, 1, 10, 10)
    a[:, :, :5, :] = 1.0  # 50 pixels
    b[:, :, 2:7, :] = 1.0  # 50 pixels, overlap is rows 2,3,4 = 30 pixels
    # Intersection = 30, Union = 70 -> IoU = 30/70 = 0.4285
    iou = compute_iou(a, b)
    assert abs(iou - (30.0 / 70.0)) < 0.01

    # Pixel metrics suite
    m = compute_pixel_metrics(a, b)
    assert m["true_positives"] == 30
    assert m["false_positives"] == 20
    assert m["false_negatives"] == 20
    assert m["true_negatives"] == 30
    assert abs(m["accuracy"] - 0.6) < 1e-4


def test_patch_tiler_and_blending():
    """Test sliding window tiling and cosine-weighted mosaic reconstruction."""
    tiler = PatchTiler(patch_size=128, stride=96)
    img = np.ones((300, 400), dtype=np.float32) * 100.0

    patches, coords = tiler.tile_image(img)
    assert len(patches) > 0
    for p in patches:
        assert p.shape == (128, 128)

    # Stitched reconstruction
    recon = tiler.stitch_patches(patches, coords, original_shape=img.shape, blending="cosine")
    assert recon.shape == img.shape
    # Check that reconstructed image is non-zero everywhere (no blank holes)
    assert np.all(recon > 50.0)


def test_sonar_dataset():
    """Test SonarSegmentationDataset loading and augmentations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "sonar_001.png")
        mask_path = os.path.join(tmpdir, "sonar_001_mask.png")

        # Create dummy image and mask
        test_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[20:60, 20:60] = 255

        cv2.imwrite(img_path, test_img)
        cv2.imwrite(mask_path, test_mask)

        dataset = SonarSegmentationDataset(
            image_paths=[img_path],
            mask_paths=[mask_path],
            img_size=128,
            augment=True
        )

        assert len(dataset) == 1
        img_tensor, mask_tensor, path = dataset[0]
        assert img_tensor.shape == (1, 128, 128)
        assert mask_tensor.shape == (1, 128, 128)
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0
        assert mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0


def test_unet_segmenter_contract():
    """
    Test UNetSegmenter states:
    1. Without checkpoint -> returns model_unavailable (honoring system contract)
    2. With loaded weights -> performs segmentation, returns contours, overlay
    """
    # 1. Unloaded model state
    segmenter = UNetSegmenter()
    res = segmenter.segment_roi(None)
    assert res["status"] == "model_unavailable"
    assert res["mask_available"] is False

    # 2. Loaded model state
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_attention_unet.pt")
        model = AttentionUNet(in_channels=1, num_classes=1, features=[16, 32, 64, 128])
        torch.save({"model_state_dict": model.state_dict(), "model_type": "attention_unet"}, ckpt_path)

        loaded_segmenter = UNetSegmenter(
            checkpoint_path=ckpt_path,
            model_type="attention_unet",
            img_size=128
        )
        assert loaded_segmenter.is_model_loaded is True

        dummy_chip = np.random.randint(30, 200, (120, 150), dtype=np.uint8)
        roi_res = loaded_segmenter.segment_roi(dummy_chip)
        assert roi_res["status"] == "success"
        assert roi_res["mask_available"] is True
        assert roi_res["mask"].shape == (120, 150)
        assert "contours_count" in roi_res
        assert "total_area_px" in roi_res

        # Test overlay
        overlay = loaded_segmenter.overlay_mask(dummy_chip, roi_res["mask"])
        assert overlay.shape == (120, 150, 3)

        # Test full image sliding window
        mosaic_res = loaded_segmenter.segment_full_image(dummy_chip)
        assert mosaic_res["status"] == "success"
        assert mosaic_res["mask"].shape == (120, 150)


def test_dataset_verifier():
    """Test non-destructive workspace mask audit."""
    res = verify_segmentation_dataset(os.path.join(PROJECT_ROOT, "datasets"))
    assert "status" in res
    assert "has_real_masks" in res
    assert "total_images_found" in res
    assert isinstance(res["has_real_masks"], bool)


if __name__ == "__main__":
    print("Running Stage 4 Unit Tests...")
    test_standard_unet()
    print("  [PASSED] test_standard_unet")
    test_attention_unet()
    print("  [PASSED] test_attention_unet")
    test_attention_gate()
    print("  [PASSED] test_attention_gate")
    test_losses_and_gradients()
    print("  [PASSED] test_losses_and_gradients")
    test_metrics()
    print("  [PASSED] test_metrics")
    test_patch_tiler_and_blending()
    print("  [PASSED] test_patch_tiler_and_blending")
    test_sonar_dataset()
    print("  [PASSED] test_sonar_dataset")
    test_unet_segmenter_contract()
    print("  [PASSED] test_unet_segmenter_contract")
    test_dataset_verifier()
    print("  [PASSED] test_dataset_verifier")
    print("All Stage 4 unit tests executed successfully!")
