"""
Stage 4: Semantic Segmentation Package
Exports U-Net models, losses, datasets, and UNetSegmenter interface.
"""

from ai.segmentation.models import UNet, AttentionUNet, DoubleConv, AttentionGate, build_unet
from ai.segmentation.losses import DiceLoss, BCEDiceLoss, FocalLoss, compute_iou, compute_dice, compute_pixel_metrics
from ai.segmentation.dataset import SonarSegmentationDataset, PatchTiler, verify_segmentation_dataset
from ai.segmentation.unet_segmenter import UNetSegmenter

__all__ = [
    "UNet",
    "AttentionUNet",
    "DoubleConv",
    "AttentionGate",
    "build_unet",
    "DiceLoss",
    "BCEDiceLoss",
    "FocalLoss",
    "compute_iou",
    "compute_dice",
    "compute_pixel_metrics",
    "SonarSegmentationDataset",
    "PatchTiler",
    "verify_segmentation_dataset",
    "UNetSegmenter"
]
