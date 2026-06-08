"""
Visualise RAovSeg predictions vs ground truth, overlaid on the source slice.

For each test subject:
  - Load image.npy (preprocessed, enhanced), ov_label.npy (GT), and the
    prediction .npy from data/predictions/.
  - Pick every z-slice where GT or prediction has any ovary voxels.
  - Render one figure per subject with one row per such slice and four columns:
       image | image + GT (red) | image + Pred (green) | combined overlay
                                                          (R=GT only,
                                                           G=Pred only,
                                                           Y=both)
  - Title each row with the slice's 2D DSC, and the whole figure with the
    full-volume DSC.

Headless matplotlib (Agg backend), so it runs inside a SLURM job or under
`srun --pty bash` on a Stanage compute node without a display.

Usage on Stanage:
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python src/RaovSeg_recreation/visualize_predictions.py \\
        --predictions-dir data/predictions \\
        --test-dir        data/processed/test \\
        --output-dir      data/predictions/figures

Then rsync the PNGs to your laptop:
    rsync -avhP \\
        ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/\\
EndometriosisDataset/data/predictions/figures/ \\
        ./figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-5


def dsc(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = pred > 0
    gb = gt > 0
    inter = int((pb & gb).sum())
    denom = int(pb.sum()) + int(gb.sum())
    return (2.0 * inter) / (denom + EPS) if denom > 0 else 0.0


def combined_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Grayscale image with GT-only in red, pred-only in green, overlap in yellow."""
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    rgb = np.stack([img, img, img], axis=-1)

    gb = gt > 0
    pb = pred > 0
    only_gt = gb & ~pb
    only_pred = pb & ~gb
    both = gb & pb
    rgb[only_gt] = [1.0, 0.0, 0.0]
    rgb[only_pred] = [0.0, 1.0, 0.0]
    rgb[both] = [1.0, 1.0, 0.0]
    return rgb


def visualise_subject(
    subject_id: str,
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
) -> str:
    nonzero = [z for z in range(image.shape[0])
               if (gt[z] > 0).any() or (pred[z] > 0).any()]
    if not nonzero:
        return f"{subject_id}: no GT or pred voxels — skipped"

    n = len(nonzero)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, z in enumerate(nonzero):
        img_s = image[z]
        gt_s = gt[z]
        pr_s = pred[z]

        d2 = dsc(pr_s, gt_s)

        # Col 0: image
        axes[row, 0].imshow(img_s, cmap="gray")
        axes[row, 0].set_title(f"slice {z}")
        axes[row, 0].axis("off")

        # Col 1: image + GT (red, semi-transparent)
        axes[row, 1].imshow(img_s, cmap="gray")
        gt_mask = np.ma.masked_where(gt_s == 0, gt_s)
        axes[row, 1].imshow(gt_mask, cmap="autumn", alpha=0.5, vmin=0, vmax=1)
        axes[row, 1].set_title("GT")
        axes[row, 1].axis("off")

        # Col 2: image + pred (green)
        axes[row, 2].imshow(img_s, cmap="gray")
        pr_mask = np.ma.masked_where(pr_s == 0, pr_s)
        axes[row, 2].imshow(pr_mask, cmap="summer", alpha=0.5, vmin=0, vmax=1)
        axes[row, 2].set_title(f"Pred (2D DSC={d2:.3f})")
        axes[row, 2].axis("off")

        # Col 3: combined RGB overlay
        axes[row, 3].imshow(combined_overlay(img_s, gt_s, pr_s))
        axes[row, 3].set_title("R=GT, G=Pred, Y=both")
        axes[row, 3].axis("off")

    full = dsc(pred, gt)
    fig.suptitle(f"{subject_id}  full-volume DSC = {full:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return f"{subject_id}: full DSC={full:.4f}, slices={n}, wrote {out_path}"


def main():
    parser = argparse.ArgumentParser(description="Visualise RAovSeg predictions vs GT")
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory containing *_pred.npy")
    parser.add_argument("--test-dir", type=Path, required=True,
                        help="data/processed/test (has D2-XXX/{image,ov_label}.npy)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write per-subject PNGs")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subset of subject IDs (default: all in predictions-dir)")
    args = parser.parse_args()

    if args.subjects:
        sids = sorted(args.subjects)
    else:
        sids = sorted(p.stem.removesuffix("_pred")
                      for p in args.predictions_dir.glob("*_pred.npy"))

    if not sids:
        print(f"No predictions found in {args.predictions_dir}")
        return

    for sid in sids:
        pred_path = args.predictions_dir / f"{sid}_pred.npy"
        img_path = args.test_dir / sid / "image.npy"
        gt_path = args.test_dir / sid / "ov_label.npy"

        missing = [p for p in (pred_path, img_path, gt_path) if not p.exists()]
        if missing:
            print(f"{sid}: skipped, missing {[str(p) for p in missing]}")
            continue

        image = np.load(img_path)
        gt = np.load(gt_path)
        pred = np.load(pred_path)

        if not (image.shape == gt.shape == pred.shape):
            print(f"{sid}: shape mismatch image={image.shape} gt={gt.shape} pred={pred.shape}")
            continue

        msg = visualise_subject(sid, image, gt, pred, args.output_dir / f"{sid}.png")
        print(msg)


if __name__ == "__main__":
    main()
