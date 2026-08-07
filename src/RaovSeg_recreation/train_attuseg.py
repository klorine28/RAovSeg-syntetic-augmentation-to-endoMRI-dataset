"""
Stage 2: AttUSeg — Attention U-Net for ovary segmentation.
4-layer Attention U-Net (MONAI) with features [16, 32, 64, 128].

Paper specs:
  - Loss: Focal Tversky Loss (alpha=0.8, beta=0.2, gamma=1.33)
  - Input: only ovary-containing slices (filtered by ResClass)
  - Train: 594 slices, Val: 136 slices
  - Augmentation: translations ±25px, rotations ±25°, 5x multiplier
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import AttentionUnet
from monai.transforms import RandAffined, Compose

FEATURES = (16, 32, 64, 128)
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EPOCHS = 100
AUGMENT_MULTIPLIER = 5
TRAIN_RATIO = 0.8  # paper reports 594/136 ovary slices ≈ 81/19 for AttUSeg

# Focal Tversky Loss params
ALPHA = 0.8
BETA = 0.2
GAMMA = 1.33

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss as described in the paper."""

    def __init__(self, alpha=0.8, beta=0.2, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky


class SegmentationDataset(Dataset):
    """Dataset of 2D slices with segmentation masks (only ovary-containing slices)."""

    def __init__(self, slices: np.ndarray, masks: np.ndarray, augment: bool = False, multiplier: int = 1):
        self.slices = slices  # (N, H, W)
        self.masks = masks    # (N, H, W)
        self.augment = augment
        self.multiplier = multiplier if augment else 1

        # Paper: augmentation "applied to increase the training dataset size by a
        # factor of five" — read as deterministic per replica, so prob=1.0.
        self.transform = Compose([
            RandAffined(
                keys=["image", "label"],
                prob=1.0,
                rotate_range=[np.deg2rad(25)],
                translate_range=[25, 25],
                padding_mode="zeros",
                mode=["bilinear", "nearest"],
            ),
        ]) if augment else None

    def __len__(self):
        return len(self.slices) * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.slices)
        img = self.slices[real_idx].copy()[np.newaxis, ...]   # (1, H, W)
        mask = self.masks[real_idx].copy()[np.newaxis, ...]   # (1, H, W)

        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).float()

        if self.transform is not None:
            data = self.transform({"image": img_tensor, "label": mask_tensor})
            img_tensor = data["image"]
            mask_tensor = data["label"]

        return img_tensor, mask_tensor


def load_target_slices(processed_dir: Path, subjects: list[str] | None = None,
                       target: str = "ov"):
    """Load only slices that contain ovary annotations.

    If `subjects` is given, only those subject IDs are loaded.
    """
    all_slices: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    if subjects is None:
        candidates = [d for d in sorted(processed_dir.iterdir()) if d.is_dir()]
    else:
        candidates = [processed_dir / sid for sid in subjects]

    for subj_dir in candidates:
        if not subj_dir.is_dir():
            continue
        img_path = subj_dir / "image.npy"
        lbl_path = subj_dir / f"{target}_label.npy"

        if not img_path.exists() or not lbl_path.exists():
            continue

        img = np.load(img_path)
        lbl = np.load(lbl_path)

        for s in range(img.shape[0]):
            if np.sum(lbl[s] > 0) > 0:  # only ovary-containing slices
                all_slices.append(img[s])
                all_masks.append((lbl[s] > 0).astype(np.float32))

    return np.array(all_slices), np.array(all_masks)


def split_subjects(processed_dir: Path, train_ratio: float, seed: int = 42):
    """Subject-level train/val split (no leakage between subjects)."""
    subjects = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(subjects))
    n_train = max(1, int(len(subjects) * train_ratio))
    train_subjects = [subjects[i] for i in indices[:n_train]]
    val_subjects = [subjects[i] for i in indices[n_train:]]
    return train_subjects, val_subjects


def build_model() -> nn.Module:
    """Build 4-layer Attention U-Net using MONAI."""
    model = AttentionUnet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=FEATURES,
        strides=(2, 2, 2),
        dropout=0.2,
    )
    return model


def dice_score(pred, target, smooth=1e-5):
    """Compute Dice score for a batch."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + target.sum() + smooth)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total_dice = 0
    n = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_score(outputs, masks).item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    n = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, masks)

        total_loss += loss.item() * imgs.size(0)
        total_dice += dice_score(outputs, masks).item() * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_dice / n


def main():
    parser = argparse.ArgumentParser(description="Train AttUSeg segmentation network")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "processed" / "train_val")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split, model init, augmentation, "
                             "and DataLoader shuffle. Default 42 reproduces the original "
                             "baseline; multi-seed runs pass different values.")
    parser.add_argument("--target", type=str, default="ov",
                        help="Target organ label channel: ov | ut | em | cy. "
                             "Uses <target>_label.npy in each subject dir and "
                             "saves the checkpoint as attuseg_best_<target>.pth.")
    args = parser.parse_args()
    ckpt_name = f"attuseg_best_{args.target}.pth"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Seed every source of randomness so 3-seed augmentation runs produce
    # genuinely independent models.
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Subject-level split (paper: "subject-level independence across subsets")
    train_subjects, val_subjects = split_subjects(args.data_dir, TRAIN_RATIO, seed=args.seed)
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")

    print(f"Target: {args.target} (label file: {args.target}_label.npy, "
          f"checkpoint: {ckpt_name})")
    print(f"\nLoading train {args.target} slices...")
    train_slices, train_masks = load_target_slices(args.data_dir, train_subjects, target=args.target)
    print(f"Loading val {args.target} slices...")
    val_slices, val_masks = load_target_slices(args.data_dir, val_subjects, target=args.target)
    print(f"Train {args.target} slices: {len(train_slices)} | Val {args.target} slices: {len(val_slices)}")

    train_ds = SegmentationDataset(train_slices, train_masks,
                                   augment=True, multiplier=AUGMENT_MULTIPLIER)
    val_ds = SegmentationDataset(val_slices, val_masks, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model — paper specifies Focal Tversky (α=0.8, β=0.2, γ=1.33). No LR
    # scheduler, optimizer, LR, or weight_decay is published; defaults below.
    model = build_model().to(DEVICE)
    criterion = FocalTverskyLoss(alpha=ALPHA, beta=BETA, gamma=GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    print(f"Device: {DEVICE}")
    print(f"Training for {args.epochs} epochs...\n")

    # Paper does not specify a checkpoint-selection criterion. AttUSeg overfits
    # late in training, so we keep the best val-DSC checkpoint.
    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} DSC: {train_dice:.3f} | "
              f"Val Loss: {val_loss:.4f} DSC: {val_dice:.3f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), args.output_dir / ckpt_name)
            print(f"  -> Saved best (val_dice={val_dice:.3f})")

    print(f"\nBest validation DSC: {best_val_dice:.3f}")
    print(f"Model saved to: {args.output_dir / ckpt_name}")


if __name__ == "__main__":
    main()
