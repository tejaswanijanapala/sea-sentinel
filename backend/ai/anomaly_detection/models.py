"""
Stage 5: Deep Convolutional Autoencoder for Acoustic Anomaly Detection
Implements:
  Algorithm 2: CNN Encoder
  Algorithm 3: Latent Representation (bottleneck z)
  Algorithm 4: CNN Decoder
  Algorithm 5: Reconstructed Image Patch x'
Learns the manifold of normal seabed backscatter (sand ripples, flat mud, soft sediment).
When anomalous anthropogenic debris (cables, nets, metal structures) is passed through,
the network cannot reconstruct the out-of-distribution geometry, producing high reconstruction error.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Algorithm 2: CNN Encoder
    Compresses high-dimensional acoustic sonar patch [B, 1, 128, 128]
    into low-dimensional latent bottleneck z [B, latent_dim].
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.conv = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Bottleneck projection to Algorithm 3 Latent Representation
        self.fc = nn.Linear(base_channels * 8 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h_flat = h.contiguous().view(h.size(0), -1)
        z = self.fc(h_flat)
        return z


class CNNDecoder(nn.Module):
    """
    Algorithm 4: CNN Decoder
    Reconstructs acoustic patch from latent representation z:
    [B, latent_dim] -> [B, 1, 128, 128]
    """
    def __init__(self, out_channels: int = 1, latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, base_channels * 8 * 8 * 8)

        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid()  # Normalize output to [0.0, 1.0]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.size(0), self.base_channels * 8, 8, 8)
        x_recon = self.deconv(h)
        return x_recon


class AcousticAutoencoder(nn.Module):
    """
    End-to-End CNN Autoencoder integrating Algorithms 1-5:
    x (Algorithm 1)
      -> CNN Encoder (Algorithm 2)
      -> Latent Code z (Algorithm 3)
      -> CNN Decoder (Algorithm 4)
      -> Reconstructed Patch x' (Algorithm 5)
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encoder = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)
        self.decoder = CNNDecoder(out_channels=in_channels, latent_dim=latent_dim, base_channels=base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
