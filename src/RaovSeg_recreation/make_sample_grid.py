#!/usr/bin/env python3
"""
Sample-slice grid: real vs each fixed synth variant.

Picks an ovary-containing slice from a real subject + the same slice
number from each synth variant, arranges them side-by-side. Great
for the dissertation's "here's what the synth actually looks like"
figure.

Usage:
    python -m src.RaovSeg_recreation.make_sample_grid \\
        --real-dir UT-EndoMRI/D2_TCPW \\
        --real-subject D2-005 \\
        --synth-root /mnt/parscratch/users/$USER/synth_mri/synth_volumes \\
        --synth-subject D2-900 \\
        --variants exp1c_concat_fixed exp1c_spade_fixed exp2_fixed exp2_lam05_fixed exp2_lam50_fixed \\
        --n-slices 3 \\
        --out figures_fixed/samples/sample_grid.png
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def _load_norm(path: Path) -> np.ndarray:
    """Load a NIfTI, percentile-normalise (1st/99th) to [0, 1]."""
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return (sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0)


def _pick_ovary_slices(mask: np.ndarray, n: int) -> list[int]:
    """Pick n slices with the most ovary voxels (well-spaced)."""
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    top = np.argsort(counts)[::-1]
    ordered = sorted(top[:max(n * 3, n)].tolist())  # 3× candidates, then spread
    if len(ordered) <= n:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, n).astype(int)
    return [ordered[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--real-subject", type=str, required=True)
    ap.add_argument("--synth-root", type=Path, required=True)
    ap.add_argument("--synth-subject", type=str, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Real volume + mask
    rs = args.real_subject
    real_img = _load_norm(args.real_dir / rs / f"{rs}_T2FS.nii.gz")
    real_ov = _load_mask(args.real_dir / rs / f"{rs}_ov.nii.gz")
    if real_ov is None:
        raise RuntimeError(f"No ovary mask for {rs}; pick a subject with _ov.nii.gz")
    slices = _pick_ovary_slices(real_ov, args.n_slices)
    print(f"real slices with ovary: {slices}")

    # Load all synth variants
    synth = {}
    for v in args.variants:
        img_p = args.synth_root / v / args.synth_subject / f"{args.synth_subject}_T2FS.nii.gz"
        mask_p = args.synth_root / v / args.synth_subject / f"{args.synth_subject}_ov.nii.gz"
        if not img_p.exists():
            print(f"  MISSING {v}: {img_p}")
            continue
        synth[v] = {
            "img": _load_norm(img_p),
            "ov":  _load_mask(mask_p),
        }
        print(f"  loaded {v}: shape {synth[v]['img'].shape}")

    # Grid: n_slices rows × (1 + n_variants) cols
    n_cols = 1 + len(synth)
    fig, axes = plt.subplots(len(slices), n_cols,
                             figsize=(3 * n_cols, 3 * len(slices)),
                             squeeze=False)

    for r, z in enumerate(slices):
        # Real slice
        ax = axes[r, 0]
        ax.imshow(real_img[z], cmap="gray", vmin=0, vmax=1)
        if real_ov is not None:
            _draw_contour(ax, real_ov[z], color="#ff3333")
        ax.set_title(f"Real {rs}\nslice {z}", fontsize=9) if r == 0 else ax.set_title(f"slice {z}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Synth variants — use SYNTH's own slices near its ovary
        for c, (v, d) in enumerate(synth.items(), 1):
            ax = axes[r, c]
            # If synth has different shape, pick a valid slice
            if d["ov"] is not None and d["ov"].any():
                sy_slices = _pick_ovary_slices(d["ov"], args.n_slices)
                sy_z = sy_slices[min(r, len(sy_slices) - 1)]
            else:
                sy_z = min(z, d["img"].shape[0] - 1)
            ax.imshow(d["img"][sy_z], cmap="gray", vmin=0, vmax=1)
            if d["ov"] is not None:
                _draw_contour(ax, d["ov"][sy_z], color="#33ff33")
            if r == 0:
                ax.set_title(f"{v}\nslice {sy_z}", fontsize=9)
            else:
                ax.set_title(f"slice {sy_z}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Real ({rs}) vs synth ({args.synth_subject}) — ovary slices (red=real, green=synth mask)",
                 y=1.005)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.out}")


def _draw_contour(ax, mask, color, lw=1.0):
    """Thin contour overlay of a binary mask."""
    if mask is None or not mask.any():
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


if __name__ == "__main__":
    main()
