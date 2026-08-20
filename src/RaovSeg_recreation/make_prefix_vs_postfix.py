#!/usr/bin/env python3
"""
Side-by-side pre-fix vs post-fix synth comparison.

For each variant, loads the pre-fix synth (buggy PatchGAN, smooth) and
the post-fix synth (real PatchGAN, rougher). Picks a set of matched
slice indices where the pre-fix ovary is present, and renders a grid
with pre-fix on top row and post-fix on bottom row for direct visual
comparison.

Assumes pre-fix synth lives at `<synth_root>/<variant>/` and post-fix
at `<synth_root>/<variant>_fixed/`. Uses the same subject in both
(same seed → same anatomy under our sampling protocol).

Usage on HPC:
    python -m src.RaovSeg_recreation.make_prefix_vs_postfix \\
        --synth-root /mnt/parscratch/users/$USER/synth_mri/synth_volumes \\
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \\
        --subject D2-900 \\
        --n-slices 4 \\
        --out-dir figures_fixed/before_after
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


WINDOW = (0.22, 0.30)


def _load_norm(path: Path) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def _pick_ovary_slices(mask: np.ndarray, n: int) -> list[int]:
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    top = np.argsort(counts)[::-1]
    # Take the top-3n candidates, then evenly space
    cand = sorted(top[:max(n * 3, n)].tolist())
    if len(cand) <= n:
        return cand
    idx = np.linspace(0, len(cand) - 1, n).astype(int)
    return [cand[i] for i in idx]


def _in_window_pct(img: np.ndarray, mask: np.ndarray) -> float:
    v = img[mask]
    if v.size == 0:
        return 0.0
    return float(((v >= WINDOW[0]) & (v <= WINDOW[1])).mean()) * 100


def _render(ax, img, mask, title, color):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=0.9)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _one_variant(synth_root: Path, variant: str, subject: str,
                 n_slices: int, out_dir: Path):
    pre_dir = synth_root / variant / subject
    post_dir = synth_root / f"{variant}_fixed" / subject
    pre_img_p = pre_dir / f"{subject}_T2FS.nii.gz"
    pre_ov_p  = pre_dir / f"{subject}_ov.nii.gz"
    post_img_p = post_dir / f"{subject}_T2FS.nii.gz"
    post_ov_p  = post_dir / f"{subject}_ov.nii.gz"

    for p in (pre_img_p, pre_ov_p, post_img_p, post_ov_p):
        if not p.exists():
            print(f"[{variant}] MISSING {p} — skipping")
            return

    pre_img = _load_norm(pre_img_p)
    pre_ov  = _load_mask(pre_ov_p)
    post_img = _load_norm(post_img_p)
    post_ov  = _load_mask(post_ov_p)

    # Pick slices from whichever side has more ovary voxels, so we
    # never accidentally pick a slice with no anatomy to compare.
    pick_from = pre_ov if pre_ov.sum() >= post_ov.sum() else post_ov
    slices = _pick_ovary_slices(pick_from, n_slices)
    if not slices:
        print(f"[{variant}] no ovary slices — skipping")
        return

    # In-window fraction per slice (pre / post)
    pre_pcts  = [_in_window_pct(pre_img[z],  pre_ov[z])  for z in slices]
    post_pcts = [_in_window_pct(post_img[z], post_ov[z]) for z in slices]

    ncols = len(slices)
    fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6.4), squeeze=False)
    for c, z in enumerate(slices):
        _render(axes[0, c], pre_img[z],  pre_ov[z]  if pre_ov  is not None else None,
                f"PRE-FIX  z={z}\nin-win {pre_pcts[c]:.0f}%",  color="#ff5555")
        _render(axes[1, c], post_img[z], post_ov[z] if post_ov is not None else None,
                f"POST-FIX z={z}\nin-win {post_pcts[c]:.0f}%", color="#33ff33")
    axes[0, 0].set_ylabel("PRE (buggy)",  fontsize=10)
    axes[1, 0].set_ylabel("POST (fixed)", fontsize=10)
    fig.suptitle(f"{variant}  /  subject {subject}  —  same slice indices, same seed",
                 y=1.00, fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"before_after_{variant}_{subject}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[{variant}] wrote {out}  "
          f"(pre in-win mean {np.mean(pre_pcts):.1f}% -> "
          f"post {np.mean(post_pcts):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", type=Path, required=True,
                    help="Directory containing both <variant>/ and <variant>_fixed/")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="Base variant names WITHOUT the _fixed suffix")
    ap.add_argument("--subject", type=str, default="D2-900",
                    help="Subject ID present in both pre and post synth dirs")
    ap.add_argument("--n-slices", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    for v in args.variants:
        _one_variant(args.synth_root, v, args.subject, args.n_slices, args.out_dir)

    print(f"\n[done] side-by-side grids in {args.out_dir}/")


if __name__ == "__main__":
    main()
