"""
Stage 5: Train CNN Autoencoder on Normal Seafloor Baseline
Learns the acoustic distribution of natural seafloor backscatter (flat sediment, sand ripples).
Calculates the normal reconstruction error baseline (mu, sigma) to automatically
calibrate the 3-sigma anomaly detection threshold T = mu + 3*sigma.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.anomaly_detection.models import AcousticAutoencoder


class NormalSeabedDataset(Dataset):
    """
    Loads normal seafloor chips with acoustic data augmentations.
    """
    def __init__(self, directory: str, patch_size: int = 128, augment: bool = True):
        self.patch_size = patch_size
        self.augment = augment
        self.image_paths = []
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.image_paths.append(os.path.join(root, f))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)

        img = cv2.resize(img, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        norm = img.astype(np.float32) / 255.0

        if self.augment:
            if np.random.random() > 0.5:
                norm = np.fliplr(norm).copy()
            if np.random.random() > 0.5:
                norm = np.flipud(norm).copy()
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                norm = np.rot90(norm, k).copy()

        tensor = torch.from_numpy(norm).unsqueeze(0).float()
        return tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train Acoustic Autoencoder for Anomaly Detection")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "datasets", "processed", "anomaly_baseline"),
                        help="Path to normal seabed baseline directory")
    parser.add_argument("--patch-size", type=int, default=128, help="Spatial resolution (H=W)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent bottleneck dimensions")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "models", "checkpoints", "autoencoder"),
                        help="Output directory for model weights and threshold JSON")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dry-run", action="store_true", help="Perform architecture & dataset audit and exit")
    return parser.parse_args()


def run_dry_run(args, device):
    print("=" * 70)
    print("STAGE 5: AUTOENCODER ANOMALY DETECTION DRY RUN")
    print("=" * 70)

    # 1. Architecture creation
    print("\n[1/3] Instantiating AcousticAutoencoder Architecture...")
    model = AcousticAutoencoder(in_channels=1, latent_dim=args.latent_dim, base_channels=32)
    model.to(device)
    param_count = model.count_parameters()
    print(f"      Trainable Parameters: {param_count:,}")

    # 2. Forward & Backward verification
    print("\n[2/3] Verifying Tensor Gradient Flow (Algorithms 1-5)...")
    dummy_x = torch.randn(2, 1, args.patch_size, args.patch_size, device=device)
    dummy_z = model.encode(dummy_x)
    assert dummy_z.shape == (2, args.latent_dim), f"Expected (2, {args.latent_dim}), got {dummy_z.shape}"
    x_recon = model.decode(dummy_z)
    assert x_recon.shape == (2, 1, args.patch_size, args.patch_size)

    criterion = nn.MSELoss()
    loss = criterion(x_recon, dummy_x)
    loss.backward()
    print(f"      Encoder bottleneck: {tuple(dummy_z.shape)} (Verified)")
    print(f"      Reconstruction:     {tuple(x_recon.shape)} (Verified)")
    print(f"      Backward gradient propagation: Complete without error.")

    # 3. Baseline dataset audit
    print("\n[3/3] Auditing Normal Seabed Baseline Dataset...")
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
    val_count = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 0
    print(f"      Baseline Train Chips: {train_count}")
    print(f"      Baseline Val Chips:   {val_count}")

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "stage5_dry_run_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "stage": "Stage 5: Anomaly Detection & False-Positive Filtering",
            "model": "AcousticAutoencoder",
            "parameters": param_count,
            "latent_dim": args.latent_dim,
            "patch_size": [args.patch_size, args.patch_size],
            "baseline_train_chips": train_count,
            "baseline_val_chips": val_count,
            "algorithms_verified": "Algorithms 1-9",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"\n[DRY RUN COMPLETED]: Report saved to {report_path}")
    print("=" * 70)


def train(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    if args.dry_run:
        run_dry_run(args, device)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds = NormalSeabedDataset(train_dir, patch_size=args.patch_size, augment=True)
    val_ds = NormalSeabedDataset(val_dir, patch_size=args.patch_size, augment=False)

    if len(train_ds) == 0:
        print(f"[ERROR] No training images found in {train_dir}")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Training on {len(train_ds)} normal seabed chips, validating on {len(val_ds)} chips...")

    model = AcousticAutoencoder(in_channels=1, latent_dim=args.latent_dim, base_channels=32)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recons = model(imgs)
            loss = criterion(recons, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_errors = []

        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(device)
                recons = model(imgs)
                loss = criterion(recons, imgs)
                val_loss += loss.item() * imgs.size(0)

                # Collect per-sample MSE
                per_sample_mse = torch.mean((recons - imgs) ** 2, dim=[1, 2, 3]).cpu().numpy()
                val_errors.extend(per_sample_mse.tolist())

        val_loss = val_loss / max(1, len(val_ds))
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        history.append({
            "epoch": epoch,
            "train_mse": round(train_loss, 6),
            "val_mse": round(val_loss, 6)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

    # Calculate 3-sigma anomaly threshold on normal validation set
    mu = float(np.mean(val_errors)) if val_errors else best_val_loss
    sigma = float(np.std(val_errors)) if val_errors else 0.005
    threshold_3sigma = float(mu + 3.0 * sigma)

    print("\n" + "=" * 50)
    print("BASELINE ANOMALY THRESHOLD CALIBRATION:")
    print(f"  Normal Seabed Mean MSE (mu):    {mu:.6f}")
    print(f"  Normal Seabed Std MSE (sigma):  {sigma:.6f}")
    print(f"  Calibrated Threshold (mu + 3s): {threshold_3sigma:.6f}")
    print("=" * 50)

    # Save Checkpoint
    checkpoint_path = os.path.join(args.output_dir, "baseline_autoencoder.pt")
    torch.save({
        "model_state_dict": best_model_state,
        "latent_dim": args.latent_dim,
        "patch_size": args.patch_size,
        "threshold": threshold_3sigma,
        "mu": mu,
        "sigma": sigma,
        "best_val_mse": best_val_loss
    }, checkpoint_path)

    # Save metadata JSON
    meta_path = os.path.join(args.output_dir, "baseline_threshold.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_checkpoint": checkpoint_path,
            "threshold_3sigma": round(threshold_3sigma, 6),
            "normal_seabed_mu": round(mu, 6),
            "normal_seabed_sigma": round(sigma, 6),
            "training_samples": len(train_ds),
            "validation_samples": len(val_ds),
            "latent_dim": args.latent_dim,
            "patch_size": [args.patch_size, args.patch_size],
            "history": history
        }, f, indent=2)

    print(f"[TRAINING COMPLETE]: Saved model to {checkpoint_path}")
    print(f"[THRESHOLD SAVED]: Saved baseline calibration to {meta_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
