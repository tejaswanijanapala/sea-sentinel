"""
Stage 4: U-Net and Attention U-Net Architectures
Provides deep semantic segmentation networks for underwater acoustic side-scan sonar (SSS) imagery.
Implements:
  1. Standard U-Net (Ronneberger et al., 2015)
  2. Attention U-Net (Oktay et al., 2018) with gating mechanisms to suppress
     seafloor clutter and reverberation while isolating irregular debris targets.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard Double Convolution block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downscaling with MaxPool2d(2) followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling with ConvTranspose2d / Bilinear interpolation followed by DoubleConv.
    Handles spatial dimension padding for odd-dimension inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: from lower decoder level, x2: skip connection from encoder
        x1 = self.up(x1)

        # Pad x1 if x2 has slightly different dimensions due to odd sizing
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018).
    Filters features propagated through skip connections using gating signals from
    coarser decoder resolutions. Suppresses irrelevant acoustic backscatter noise and
    sand ripple reverberation, focusing on high-salience debris targets.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int):
        """
        Args:
            f_g: Channel count of gating signal from decoder
            f_l: Channel count of skip connection signal from encoder
            f_int: Intermediate channel count
        """
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: Gating signal from decoder [B, f_g, H_g, W_g]
            x: Skip connection from encoder [B, f_l, H_x, W_x]
        Returns:
            attended_x: Attended skip features [B, f_l, H_x, W_x]
            alpha: Attention coefficient map [B, 1, H_x, W_x]
        """
        # Resize gating signal if spatial resolution differs
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode="bilinear", align_corners=True)

        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        attended_x = x * alpha
        return attended_x, alpha


class AttentionUpBlock(nn.Module):
    """
    Upscaling block incorporating an Attention Gate on the skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.gate = AttentionGate(f_g=in_channels // 2, f_l=in_channels // 2, f_int=in_channels // 4)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.gate = AttentionGate(f_g=in_channels // 2, f_l=in_channels // 2, f_int=in_channels // 4)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x1: from decoder (gating signal), x2: from encoder (skip features)
        x1_up = self.up(x1)

        # Pad if odd dimensions
        diff_y = x2.size()[2] - x1_up.size()[2]
        diff_x = x2.size()[3] - x1_up.size()[3]
        if diff_x > 0 or diff_y > 0:
            x1_up = F.pad(x1_up, [diff_x // 2, diff_x - diff_x // 2,
                                 diff_y // 2, diff_y - diff_y // 2])

        attended_x2, alpha = self.gate(g=x1_up, x=x2)
        x = torch.cat([attended_x2, x1_up], dim=1)
        out = self.conv(x)
        return out, alpha


class UNet(nn.Module):
    """
    Standard U-Net architecture for semantic segmentation.
    Configurable for 1-channel grayscale SSS acoustic imagery.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = False
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]  # Optimized base for edge deployment and fast training

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor)

        self.up1 = UpBlock(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = UpBlock(features[3], features[2] // factor, bilinear)
        self.up3 = UpBlock(features[2], features[1] // factor, bilinear)
        self.up4 = UpBlock(features[1], features[0], bilinear)

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture.
    Applies Attention Gates along skip connections to isolate subtle debris structures
    (e.g., thin lost fishing cables, nets, pipeline edges) amidst severe acoustic speckle.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = False
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        factor = 2 if bilinear else 1
        self.down4 = DownBlock(features[3], features[3] * 2 // factor)

        self.up1 = AttentionUpBlock(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = AttentionUpBlock(features[3], features[2] // factor, bilinear)
        self.up3 = AttentionUpBlock(features[2], features[1] // factor, bilinear)
        self.up4 = AttentionUpBlock(features[1], features[0], bilinear)

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x, a1 = self.up1(x5, x4)
        x, a2 = self.up2(x, x3)
        x, a3 = self.up3(x, x2)
        x, a4 = self.up4(x, x1)
        logits = self.outc(x)

        if return_attention:
            return logits, [a1, a2, a3, a4]
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(
    model_type: str = "attention_unet",
    in_channels: int = 1,
    num_classes: int = 1,
    features: Optional[List[int]] = None,
    bilinear: bool = False
) -> nn.Module:
    """
    Factory function to instantiate segmentation models.
    """
    model_type_norm = model_type.lower().replace("-", "_").replace(" ", "")
    if "attention" in model_type_norm:
        return AttentionUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            bilinear=bilinear
        )
    elif "unet" in model_type_norm:
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            bilinear=bilinear
        )
    else:
        raise ValueError(f"Unknown segmentation model type: '{model_type}'. Choose 'unet' or 'attention_unet'.")
