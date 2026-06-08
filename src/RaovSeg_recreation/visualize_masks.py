"""
visualize_masks.py — overlay every available anatomical mask on raw MRI slices.

For each subject (one per dir in UT-EndoMRI/D2_TCPW/), this loads the raw
T2FS NIfTI and every mask file present:

    D2-XXX_ov.nii.gz   ovary       (red)
    D2-XXX_ut.nii.gz   uterus      (yellow)
    D2-XXX_em.nii.gz   endometrioma (green)
    D2-XXX_cy.nii.gz   cyst        (cyan)

Only slices where *any* mask has voxels are rendered (skips empty slices to
keep the figure tight). For each such slice the figure has columns:

    [raw image]  [image + all masks overlaid]  [one column per mask present]

Optionally overlays the RAovSeg ovary prediction (loaded from .npy) as a
fifth color (magenta) if --predictions-dir is given. The prediction is
on the preprocessed grid (0.35×0.35 in-plane, native z), so we resample it
to the raw grid using nearest-neighbour — fine for binary masks.

Headless matplotlib (Agg backend), so this runs on a Stanage login or
compute node without a display.

Usage on Stanage:
    module load Anaconda3/2024.02-1
    source activate synth_mri
    export MKL_THREADING_LAYER=GNU
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python src/visualize_masks.py \\
        --raw-root      UT-EndoMRI/D2_TCPW \\
        --output-dir    data/predictions/mask_figures \\
        --predictions-dir data/predictions          # optional

Pull figures down:
    rsync -avhP \\
        ijp25lg@stanage.shef.ac.uk:/mnt/parscratch/users/ijp25lg/synth_mri/\\
EndometriosisDataset/data/predictions/mask_figures/ \\
        ~/Documents/EndometriosisDataset/data/predictions/mask_figures/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SUBJECT_RE = re.compile(r"^D2-\d{3}$")
SEQUENCES = ["T2FS", "T2", "T1FS", "T1"]   # T2FS preferred, fallback chain

# Mask suffix -> (display label, RGB color)
MASK_COLORS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "ov": ("ovary",        (1.0, 0.0, 0.0)),   # red
    "ut": ("uterus",       (1.0, 1.0, 0.0)),   # yellow
    "em": ("endometrioma", (0.0, 1.0, 0.0)),   # green
    "cy": ("cyst",         (0.0, 1.0, 1.0)),   # cyan
}
PRED_COLOR = (1.0, 0.0, 1.0)                    # magenta — RAovSeg ovary pred


def find_mri(subject_dir: Path, subject_id: str) -> Optional[Path]:
    for seq in SEQUENCES:
        p = subject_dir / f"{subject_id}_{seq}.nii.gz"
        if p.exists():
            return p
    return None


def load_volume(path: Path) -> np.ndarray:
    """Return numpy array (Z, Y, X) from a NIfTI."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def normalise_to_unit(arr: np.ndarray) -> np.ndarray:
    """Percentile-clip + min-max to [0,1] for display only."""
    a = arr.astype(np.float32)
    p1, p99 = np.percentile(a, [1, 99])
    a = np.clip(a, p1, p99)
    return (a - p1) / max(p99 - p1, 1e-6)


def resample_pred_to_image(
    pred: np.ndarray, raw_image_sitk: sitk.Image
) -> np.ndarray:
    """Resample a (Z, H, W) prediction (RAovSeg preprocessed grid: 512×512 at
    0.35 mm in-plane, Z spacing 6.0 mm) onto the raw image's native grid.

    RAovSeg_tools.ImgResample preserves the image *center* in physical space
    while changing spacing/size, so the preprocessed grid's origin differs
    from the raw image's origin. We must reconstruct that origin shift here,
    otherwise the prediction lands at the corner of the raw grid instead of
    the centre — manifesting as the Z and XY mis-registration we saw.
    """
    # Preprocessed grid constants — must match src/preprocess.py
    PREP_XY = 512
    PREP_SPACING = (0.35, 0.35, 6.0)

    raw_size = np.array(raw_image_sitk.GetSize(), dtype=float)         # (X, Y, Z)
    raw_spacing = np.array(raw_image_sitk.GetSpacing(), dtype=float)
    direction = np.array(raw_image_sitk.GetDirection()).reshape(3, 3)
    raw_origin = np.array(raw_image_sitk.GetOrigin())

    # Prediction shape is (Z, H, W) in numpy; SITK size is (X, Y, Z) = (W, H, Z).
    prep_size = np.array([PREP_XY, PREP_XY, pred.shape[0]], dtype=float)
    prep_spacing = np.array(PREP_SPACING, dtype=float)

    # Replicate ImgResample's center-preserving origin shift.
    raw_center = (raw_size - 1.0) / 2.0 * raw_spacing
    prep_center = (prep_size - 1.0) / 2.0 * prep_spacing
    prep_origin = raw_origin + direction @ (raw_center - prep_center)

    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_img.SetSpacing(tuple(prep_spacing.tolist()))
    pred_img.SetDirection(raw_image_sitk.GetDirection())
    pred_img.SetOrigin(tuple(prep_origin.tolist()))

    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(raw_image_sitk)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetDefaultPixelValue(0)
    return sitk.GetArrayFromImage(res.Execute(pred_img))


def color_overlay(
    image_2d: np.ndarray,
    layers: list[tuple[np.ndarray, tuple[float, float, float]]],
    alpha: float = 0.3,
) -> np.ndarray:
    """Return an RGB image: grayscale base with each binary layer alpha-blended in its color."""
    base = np.stack([image_2d] * 3, axis=-1).astype(np.float32)
    out = base.copy()
    for mask, color in layers:
        m = mask > 0
        if not m.any():
            continue
        out[m] = alpha * np.array(color, dtype=np.float32) + (1.0 - alpha) * base[m]
    return np.clip(out, 0.0, 1.0)


def visualise_subject(
    subject_id: str,
    raw_image_path: Path,
    mask_paths: dict[str, Path],
    pred_path: Optional[Path],
    out_path: Path,
) -> str:
    raw_img_sitk = sitk.ReadImage(str(raw_image_path), sitk.sitkFloat32)
    image = normalise_to_unit(load_volume(raw_image_path))

    masks: dict[str, np.ndarray] = {}
    for suffix, mp in mask_paths.items():
        m = (load_volume(mp) > 0).astype(np.uint8)
        if m.shape != image.shape:
            # Resample mask to raw image grid using NN
            m_img = sitk.ReadImage(str(mp), sitk.sitkUInt8)
            res = sitk.ResampleImageFilter()
            res.SetReferenceImage(raw_img_sitk)
            res.SetInterpolator(sitk.sitkNearestNeighbor)
            res.SetDefaultPixelValue(0)
            m = sitk.GetArrayFromImage(res.Execute(m_img)).astype(np.uint8)
        masks[suffix] = m

    pred: Optional[np.ndarray] = None
    if pred_path is not None and pred_path.exists():
        pred_arr = np.load(pred_path).astype(np.uint8)
        if pred_arr.shape != image.shape:
            pred = resample_pred_to_image(pred_arr, raw_img_sitk)
        else:
            pred = pred_arr

    # Pick slices: any voxel in any mask (or prediction)
    has_voxel = np.zeros(image.shape[0], dtype=bool)
    for m in masks.values():
        has_voxel |= m.any(axis=(1, 2))
    if pred is not None:
        has_voxel |= pred.any(axis=(1, 2))
    nonzero = np.where(has_voxel)[0].tolist()

    if not nonzero:
        return f"{subject_id}: no mask voxels in any slice — skipped"

    # Active mask types for this subject (only render columns for masks present)
    active_suffixes = [s for s in MASK_COLORS if s in masks]
    show_pred = pred is not None and pred.any()

    n_extra_cols = len(active_suffixes) + (1 if show_pred else 0)
    n_cols = 2 + n_extra_cols   # raw | all | per-mask cols
    n_rows = len(nonzero)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, z in enumerate(nonzero):
        img_s = image[z]

        # Col 0: raw image
        axes[row, 0].imshow(img_s, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"slice {z}")
        axes[row, 0].axis("off")

        # Col 1: all masks together
        layers = [(masks[s][z], MASK_COLORS[s][1]) for s in active_suffixes]
        if show_pred:
            layers.append((pred[z], PRED_COLOR))
        axes[row, 1].imshow(color_overlay(img_s, layers))
        axes[row, 1].set_title("all overlays")
        axes[row, 1].axis("off")

        # One column per mask, then prediction (if any)
        col = 2
        for s in active_suffixes:
            axes[row, col].imshow(img_s, cmap="gray", vmin=0, vmax=1)
            mask_z = np.ma.masked_where(masks[s][z] == 0, masks[s][z])
            color = MASK_COLORS[s][1]
            cmap = matplotlib.colors.ListedColormap([color])
            axes[row, col].imshow(mask_z, cmap=cmap, alpha=0.3)
            axes[row, col].set_title(MASK_COLORS[s][0])
            axes[row, col].axis("off")
            col += 1
        if show_pred:
            axes[row, col].imshow(img_s, cmap="gray", vmin=0, vmax=1)
            pred_z = np.ma.masked_where(pred[z] == 0, pred[z])
            cmap = matplotlib.colors.ListedColormap([PRED_COLOR])
            axes[row, col].imshow(pred_z, cmap=cmap, alpha=0.3)
            axes[row, col].set_title("RAovSeg pred (ovary)")
            axes[row, col].axis("off")

    fig.suptitle(f"{subject_id}  ({len(nonzero)} slices)", fontsize=14)

    # Figure-level legend listing only the classes present in this subject.
    handles = [Patch(facecolor=MASK_COLORS[s][1], edgecolor="black",
                     label=MASK_COLORS[s][0]) for s in active_suffixes]
    if show_pred:
        handles.append(Patch(facecolor=PRED_COLOR, edgecolor="black",
                             label="RAovSeg pred (ovary)"))
    if handles:
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.965), ncol=len(handles),
                   frameon=False, fontsize=11)
    # Leave headroom for both suptitle and legend
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)

    summary = ", ".join(f"{MASK_COLORS[s][0]}={int(masks[s].sum())}vx" for s in active_suffixes)
    if show_pred:
        summary += f", pred={int(pred.sum())}vx"
    return f"{subject_id}: {summary}, wrote {out_path}"


def main():
    parser = argparse.ArgumentParser(description="Overlay every available mask on raw MRI slices")
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="UT-EndoMRI/D2_TCPW (one dir per subject)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write per-subject PNGs")
    parser.add_argument("--predictions-dir", type=Path, default=None,
                        help="Optional: directory with D2-XXX_pred.npy for ovary overlay")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subset of subject IDs (default: every D2-XXX in raw-root)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep existing PNGs in --output-dir instead of removing them at start")
    args = parser.parse_args()

    # Clean stale PNGs so re-runs don't leave orphan figures for subjects that
    # no longer match (e.g. after a split or naming change). Only *.png files
    # are touched; anything else in the dir is left alone.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clean:
        stale = list(args.output_dir.glob("*.png"))
        if stale:
            print(f"Removing {len(stale)} stale PNG(s) from {args.output_dir}")
            for f in stale:
                f.unlink()

    raw_root = args.raw_root
    if args.subjects:
        sids = sorted(args.subjects)
    else:
        sids = sorted(p.name for p in raw_root.iterdir()
                      if p.is_dir() and SUBJECT_RE.match(p.name))

    if not sids:
        print(f"No D2-* subject directories found in {raw_root}")
        return

    for sid in sids:
        subj_dir = raw_root / sid
        if not subj_dir.is_dir():
            print(f"{sid}: skipped, no directory at {subj_dir}")
            continue

        image_path = find_mri(subj_dir, sid)
        if image_path is None:
            print(f"{sid}: skipped, no MRI sequence found")
            continue

        mask_paths: dict[str, Path] = {}
        for suffix in MASK_COLORS:
            p = subj_dir / f"{sid}_{suffix}.nii.gz"
            if p.exists():
                mask_paths[suffix] = p

        if not mask_paths:
            print(f"{sid}: skipped, no masks of any kind")
            continue

        pred_path: Optional[Path] = None
        if args.predictions_dir is not None:
            pred_path = args.predictions_dir / f"{sid}_pred.npy"
            if not pred_path.exists():
                pred_path = None

        msg = visualise_subject(sid, image_path, mask_paths, pred_path,
                                args.output_dir / f"{sid}.png")
        print(msg)


if __name__ == "__main__":
    main()
