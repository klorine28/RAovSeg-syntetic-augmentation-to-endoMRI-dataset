"""
Stage 1: ResClass — Binary slice classifier (ovary-containing vs not).
2-block ResNet18-style classifier with 8 and 16 features per the paper Fig 3.

Paper specs:
  - Loss: BCEWithLogitsLoss
  - Dropout: 0.2, L2 regularization
  - Augmentation: translations ±25px, rotations ±25°, 5x multiplier
  - Train: 3,252 slices, Val: 2,168 slices

Paper does not specify epoch count or stopping criterion. We train for 50
epochs and save the best-validation-F1 checkpoint. Early stopping (patience
configurable) is disabled by default; experiments showed it stops too early
on this small dataset and selects an under-trained checkpoint.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import Compose, RandAffined

# --- Config ---
FEATURES = (8, 16)
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4  # L2 regularization
LR = 1e-3            # not specified in paper, reasonable default
BATCH_SIZE = 32
EPOCHS_MAX = 50     # cap — early stopping usually fires earlier
PATIENCE = 999        # early-stop after N epochs without F1 improvement (but does not help)
AUGMENT_MULTIPLIER = 5
TRAIN_RATIO = 0.6    # paper: 3,252/(3,252+2,168) = 60/40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class SliceDataset(Dataset):
    """Dataset of 2D slices with binary labels (has ovary or not)."""

    def __init__(self, slices: np.ndarray, labels: np.ndarray, augment: bool = False, multiplier: int = 1):
        self.slices = slices
        self.labels = labels
        self.augment = augment
        self.multiplier = multiplier if augment else 1

        self.transform = Compose([
            RandAffined(
                keys=["image"],
                prob=1.0,
                rotate_range=[np.deg2rad(25)],
                translate_range=[25, 25],
                padding_mode="zeros",
            ),
        ]) if augment else None

    def __len__(self):
        return len(self.slices) * self.multiplier

    def __getitem__(self, idx):
        real_idx = idx % len(self.slices)
        img = self.slices[real_idx].copy()[np.newaxis, ...]
        label = self.labels[real_idx]
        img_tensor = torch.from_numpy(img).float()
        if self.transform is not None:
            data = {"image": img_tensor}
            data = self.transform(data)
            img_tensor = data["image"]
        return img_tensor, torch.tensor(label, dtype=torch.float32)


def load_slices(processed_dir: Path, subjects: list[str] | None = None):
    all_slices: list[np.ndarray] = []
    all_labels: list[int] = []
    candidates = ([processed_dir / sid for sid in subjects] if subjects is not None
                  else [d for d in sorted(processed_dir.iterdir()) if d.is_dir()])
    for subj_dir in candidates:
        if not subj_dir.is_dir():
            continue
        img_path = subj_dir / "image.npy"
        lbl_path = subj_dir / "ov_label.npy"
        if not img_path.exists() or not lbl_path.exists():
            print(f"  Skip {subj_dir.name}: missing image or label")
            continue
        img = np.load(img_path)
        lbl = np.load(lbl_path)
        for s in range(img.shape[0]):
            all_slices.append(img[s])
            all_labels.append(1 if np.sum(lbl[s] > 0) > 0 else 0)
    return np.array(all_slices), np.array(all_labels)


def split_subjects(processed_dir: Path, train_ratio: float, seed: int = 42):
    """Subject-level train/val split (no leakage between subjects)."""
    subjects = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(subjects))
    n_train = max(1, int(len(subjects) * train_ratio))
    train_subjects = [subjects[i] for i in indices[:n_train]]
    val_subjects = [subjects[i] for i in indices[n_train:]]
    return train_subjects, val_subjects


class TwoBlockResNet(nn.Module):
    """Two-BasicBlock ResNet classifier per Liang et al. Fig. 3.

    Architecture: Stem -> ResNetBlock(8) -> ResNetBlock(8->16, stride=2) ->
    GAP -> Dropout(0.2) -> Linear(16, 1).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block1 = ResNetBlock(in_planes=8, planes=8, spatial_dims=2)
        downsample = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(16),
        )
        self.block2 = ResNetBlock(
            in_planes=8, planes=16, spatial_dims=2, stride=2, downsample=downsample,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(16, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def build_model() -> nn.Module:
    return TwoBlockResNet(in_channels=1, num_classes=1, dropout=DROPOUT)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = fn = tn = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs).squeeze(-1)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
    acc = correct / total
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return total_loss / total, acc, precision, recall


def main():
    parser = argparse.ArgumentParser(description="Train ResClass with early stopping on val F1")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "train_val")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "models")
    parser.add_argument("--max-epochs", type=int, default=EPOCHS_MAX,
                        help="Maximum number of epochs (early stop usually fires earlier)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early-stop patience: stop after N epochs with no val F1 improvement")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_subjects, val_subjects = split_subjects(args.data_dir, TRAIN_RATIO)
    print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Val subjects ({len(val_subjects)}): {val_subjects}")

    print("\nLoading train slices...")
    train_slices, train_labels = load_slices(args.data_dir, train_subjects)
    print("Loading val slices...")
    val_slices, val_labels = load_slices(args.data_dir, val_subjects)

    n_pos_train = int(np.sum(train_labels == 1))
    n_neg_train = int(np.sum(train_labels == 0))
    print(f"Train slices: {len(train_slices)} (pos={n_pos_train}, neg={n_neg_train})")
    print(f"Val   slices: {len(val_slices)} (pos={int(np.sum(val_labels == 1))}, "
          f"neg={int(np.sum(val_labels == 0))})")

    train_ds = SliceDataset(train_slices, train_labels, augment=True, multiplier=AUGMENT_MULTIPLIER)
    val_ds = SliceDataset(val_slices, val_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    print(f"Device: {DEVICE}")
    print(f"Max epochs: {args.max_epochs} | Early-stop patience: {args.patience} epochs\n")

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_prec, val_rec = evaluate(model, val_loader, criterion)
        val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec + 1e-8)

        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} "
              f"Prec: {val_prec:.3f} Rec: {val_rec:.3f} F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.output_dir / "resclass_best.pth")
            print(f"  -> Saved best (val_f1={val_f1:.3f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}: "
                      f"no improvement in val F1 for {args.patience} epochs.")
                break

    print(f"\nBest validation F1: {best_val_f1:.3f} (epoch {best_epoch})")
    print(f"Trained for {epoch} epochs total")
    print(f"Model saved to: {args.output_dir / 'resclass_best.pth'}")


if __name__ == "__main__":
    main()