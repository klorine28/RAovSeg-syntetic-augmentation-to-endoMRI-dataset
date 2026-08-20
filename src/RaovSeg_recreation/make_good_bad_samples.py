#!/usr/bin/env python3
"""
Render N good + M bad sample slices PER VARIANT from the fixed synth.

Default: 20 good + 5 bad per variant = 25 per experiment.
With 5 variants that's 125 images total.

"Good" = high ovary voxel count AND ovary mean intensity near the RAovSeg
enhancement window [0.22, 0.30]. Ranking blends both signals.

"Bad" = the opposite: either tiny/absent ovary OR ovary intensity far from
window. These illustrate failure modes.

Scoring uses ALL ovary-containing slices (not just one per subject) so
that even variants with few subjects can produce 25 samples.

Usage:
    python -m src.RaovSeg_recreation.make_good_bad_samples \\
        --synth-root /mnt/parscratch/users/$USER/synth_mri/synth_volumes \\
        --variants exp1c_concat_fixed exp1c_spade_fixed exp2_fixed exp2_lam05_fixed exp2_lam50_fixed \\
        --n-good 20 --n-bad 5 \\
        --out-dir figures_fixed/samples
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


WINDOW = (0.22, 0.30)   # RAovSeg enhancement window
WINDOW_CENTER = 0.5 * (WINDOW[0] + WINDOW[1])


def _load_norm(path: Path) -> np.ndarray:
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))
    lo, hi = float(np.percentile(img, 1)), float(np.percentile(img, 99))
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def _score_slice(img_slice: np.ndarray, ov_slice: np.ndarray) -> dict | None:
    """Compute a quality score for one slice + its ovary mask."""
    if ov_slice is None or not ov_slice.any():
        return None
    ov_vox = int(ov_slice.sum())
    ov_intensities = img_slice[ov_slice]
    ov_mean = float(ov_intensities.mean())
    ov_in_window = float(((ov_intensities >= WINDOW[0]) & (ov_intensities <= WINDOW[1])).mean())
    window_distance = abs(ov_mean - WINDOW_CENTER)
    # Quality proxy: reward ovary size + in-window match, penalise distance
    # from window. Small/absent ovary gets low score; well-placed intensity
    # gets high score.
    score = np.log1p(ov_vox) * (ov_in_window + 0.5) - 3.0 * window_distance
    return {
        "ov_vox": ov_vox,
        "ov_mean": ov_mean,
        "ov_in_window": ov_in_window,
        "score": float(score),
    }


def _scan_variant(synth_root: Path, variant: str) -> list[dict]:
    """Score EVERY ovary-containing slice across all subjects in the variant."""
    variant_dir = synth_root / variant
    entries = []
    for subj_dir in sorted(variant_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        img_p = subj_dir / f"{sid}_T2FS.nii.gz"
        ov_p = subj_dir / f"{sid}_ov.nii.gz"
        if not img_p.exists() or not ov_p.exists():
            continue
        img = _load_norm(img_p)
        ov = _load_mask(ov_p)
        if ov is None or not ov.any():
            continue
        for z in range(img.shape[0]):
            s = _score_slice(img[z], ov[z])
            if s is None:
                continue
            s["z"] = z
            s["subject"] = sid
            s["variant"] = variant
            s["img_path"] = str(img_p)
            s["ov_path"] = str(ov_p)
            entries.append(s)
    return entries


def _save_slice_png(entry: dict, out: Path, label: str, tag: str):
    img = _load_norm(Path(entry["img_path"]))[entry["z"]]
    ov = _load_mask(Path(entry["ov_path"]))[entry["z"]]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.contour(ov.astype(float), levels=[0.5], colors=["#33ff33"], linewidths=1.0)
    title = (f"[{tag}] {entry['variant']} / {entry['subject']} / z={entry['z']}\n"
             f"ov_vox={entry['ov_vox']}  mean={entry['ov_mean']:.3f}  "
             f"in_win={entry['ov_in_window']*100:.0f}%  score={entry['score']:+.2f}")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_grid(entries: list[dict], out: Path, title: str, ncols: int = 5):
    n = len(entries)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for i, e in enumerate(entries):
        ax = axes[i // ncols, i % ncols]
        img = _load_norm(Path(e["img_path"]))[e["z"]]
        ov = _load_mask(Path(e["ov_path"]))[e["z"]]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.contour(ov.astype(float), levels=[0.5], colors=["#33ff33"], linewidths=0.8)
        ax.set_title(f"{e['variant'].replace('_fixed', '')}/{e['subject']}\n"
                     f"in_win={e['ov_in_window']*100:.0f}%",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(title, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _print_summary(label: str, entries: list[dict]):
    print(f"\n=== {label} samples ({len(entries)}) ===")
    print(f"{'idx':<4} {'variant':<22} {'subject':<8} {'z':>3} {'ov_vox':>7} {'mean':>7} {'in_win':>7} {'score':>7}")
    for i, e in enumerate(entries):
        print(f"{i+1:<4} {e['variant']:<22} {e['subject']:<8} {e['z']:>3} "
              f"{e['ov_vox']:>7} {e['ov_mean']:>7.3f} {e['ov_in_window']*100:>6.1f}% {e['score']:>+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-good", type=int, default=20,
                    help="good samples PER VARIANT (default 20)")
    ap.add_argument("--n-bad", type=int, default=5,
                    help="bad samples PER VARIANT (default 5)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    # Per-variant scan + per-variant selection
    for v in args.variants:
        print(f"\n{'='*60}\n[variant] {v}\n{'='*60}")
        vdir_good = args.out_dir / v / "good"
        vdir_bad  = args.out_dir / v / "bad"
        vdir_good.mkdir(parents=True, exist_ok=True)
        vdir_bad.mkdir(parents=True, exist_ok=True)

        entries = _scan_variant(args.synth_root, v)
        print(f"  {len(entries)} candidate slices (all ovary-containing)")

        if not entries:
            print("  skip — no candidates")
            continue

        # Sort by score (high → low)
        entries.sort(key=lambda e: -e["score"])

        n_good = min(args.n_good, len(entries))
        n_bad  = min(args.n_bad, max(0, len(entries) - n_good))
        good = entries[:n_good]
        bad  = entries[-n_bad:] if n_bad > 0 else []

        # Individual PNGs
        for i, e in enumerate(good):
            _save_slice_png(e,
                vdir_good / f"good_{i:02d}_{e['subject']}_z{e['z']}.png",
                label="good", tag=f"{v} good {i+1}/{n_good}")
        for i, e in enumerate(bad):
            _save_slice_png(e,
                vdir_bad / f"bad_{i:02d}_{e['subject']}_z{e['z']}.png",
                label="bad", tag=f"{v} bad {i+1}/{n_bad}")

        # Per-variant grid
        _save_grid(good, args.out_dir / v / f"{v}_good_grid.png",
                   title=f"{v} — top {n_good} slices by quality score")
        if bad:
            _save_grid(bad, args.out_dir / v / f"{v}_bad_grid.png",
                       title=f"{v} — bottom {n_bad} slices (failure modes)")

        _print_summary(f"{v} GOOD", good)
        if bad:
            _print_summary(f"{v} BAD", bad)

    print(f"\n[done] outputs in {args.out_dir}/<variant>/")


if __name__ == "__main__":
    main()
