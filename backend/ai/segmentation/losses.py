"""
Stage 4: Segmentation Loss Functions and Evaluation Metrics
Implements loss functions designed for high class imbalance in underwater acoustic sonar imagery:
  1. Dice Loss (Milletari et al., 2016)
  2. BCE-Dice Hybrid Loss (combining smooth pixel gradient + global overlap)
  3. Focal Loss (Lin et al., 2017)
  4. Geospatial and acoustic segmentation metrics (IoU, Dice, Precision, Recall)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Differentiable Soft Dice Loss for binary or multi-class semantic segmentation.
    Addresses severe foreground-to-background class imbalance where debris occupies
    less than 5% of the acoustic seafloor mosaic.
    """
    def __init__(self, smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, 1, H, W] or probabilities
            target: Ground truth binary mask [B, 1, H, W] (values in {0, 1})
        """
        if self.from_logits:
            pred = torch.sigmoid(pred)

        # Flatten batch and spatial dimensions
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1).float()

        intersection = (pred_flat * target_flat).sum()
        cardinality = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy (BCE) and Dice Loss:
    Loss = w_bce * BCE + w_dice * Dice
    Provides stable pixel-level gradient propagation while optimizing global region overlap.
    """
    def __init__(self, w_bce: float = 0.5, w_dice: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth, from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.w_bce * loss_bce + self.w_dice * loss_dice


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) to down-weight well-classified easy seabed background pixels
    and focus gradient updates on hard debris boundaries and acoustic shadow transitions.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if self.from_logits:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            prob = torch.sigmoid(pred)
        else:
            bce_loss = F.binary_cross_entropy(pred, target, reduction="none")
            prob = pred

        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        modulating_factor = (1.0 - p_t) ** self.gamma

        focal_loss = alpha_factor * modulating_factor * bce_loss
        return focal_loss.mean()


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Computes Intersection over Union (Jaccard Index) for binary segmentation.
    """
    if pred.ndim >= 2 and pred.is_floating_point():
        pred_binary = (pred >= threshold).float()
    else:
        pred_binary = pred.float()

    target_binary = target.float()

    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target_binary.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    union = (pred_flat + target_flat).clamp(0, 1).sum().item()

    if union == 0:
        return 1.0  # Perfect agreement on empty mask
    return float((intersection + smooth) / (union + smooth))


def compute_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Computes Dice Coefficient (F1-score) for binary segmentation.
    """
    if pred.ndim >= 2 and pred.is_floating_point():
        pred_binary = (pred >= threshold).float()
    else:
        pred_binary = pred.float()

    target_binary = target.float()

    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target_binary.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    total = pred_flat.sum().item() + target_flat.sum().item()

    if total == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (total + smooth))


def compute_pixel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes complete suite of pixel-level metrics:
    - IoU (Jaccard)
    - Dice (F1-score)
    - Pixel Accuracy
    - Precision
    - Recall
    - Specificity
    """
    if pred.is_floating_point():
        pred_b = (pred >= threshold).bool()
    else:
        pred_b = (pred > 0).bool()

    target_b = (target > 0).bool()

    tp = (pred_b & target_b).sum().item()
    fp = (pred_b & ~target_b).sum().item()
    fn = (~pred_b & target_b).sum().item()
    tn = (~pred_b & ~target_b).sum().item()
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    specificity = tn / max(tn + fp, eps)
    dice = (2 * tp) / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }
