"""
Phase 2 diagnostic: compare real D2 T2FS to synth from exp2, exp2_lam05,
and 1c_spade — all after RAovSeg's percentile-clip + minmax normalization
(no enhancement) so we see them as the segmenter would.

Answers three questions in one figure:

  Q1 (visual grid, top): how do real and synth look side by side across
     several subjects and z-slices?
  Q2 (overall histogram, middle): where do body voxels sit on the
     [0, 1] intensity axis for each variant?
  Q3 (ovary histogram, bottom): specifically the ovary-labeled voxels —
     are they inside RAovSeg's [0.22, 0.30] enhancement window, or
     misplaced?

Usage on HPC (interactive CPU is fine, ~2 min):

  python -m src.RaovSeg_recreation.diagnose_synth_vs_real \
    --real-dir UT-EndoMRI/D2_TCPW \
    --real-subjects D2-016 D2-017 D2-024 \
    --synth-dirs \
        exp2=/mnt/parscratch/users/ijp25lg/synth_mri/synth_volumes/exp2 \
        exp2_lam05=/mnt/parscratch/users/ijp25lg/synth_mri/synth_volumes/exp2_lam05 \
        1c_spade=/mnt/parscratch/users/ijp25lg/synth_mri/synth_volumes/exp1c_spade \
    --synth-subjects D2-900 D2-901 D2-902 \
    --out-png diag_synth_vs_real.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "RAovSeg"))
from RAovSeg_tools import ImgResample, ImgNorm  # noqa: E402


# Match RAovSeg preprocess constants (src/RaovSeg_recreation/preprocess.py)
OUT_SPACING = (0.35, 0.35, 6.0)
OUT_XY = 512
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99
O1 = 0.22   # enhancement window low
O2 = 0.30   # enhancement window high
# Path B target used at assembly time (informational only)
DEFAULT_OVARY_TARGET = 0.26

# Subject-side thresholds
BODY_THRESHOLD = 0.05   # after normalization, air is < this


def _out_size(img: sitk.Image) -> tuple[int, int, int]:
    return (OUT_XY, OUT_XY, int(img.GetSize()[2]))


def _load_preprocessed_image(img_path: Path) -> np.ndarray:
    """RAovSeg-style: resample → percentile clip 1-99 → minmax to [0, 1].
    NO enhancement (we want to see the segmenter's input distribution)."""
    img = sitk.ReadImage(str(img_path), sitk.sitkFloat64)
    img = ImgResample(img, out_spacing=OUT_SPACING, out_size=_out_size(img),
                      is_label=False, pad_value=0)
    arr = sitk.GetArrayFromImage(img)   # (Z, H, W)
    arr = ImgNorm(arr, norm_type="percentile_clip",
                  percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH)
    return arr.astype(np.float32)


def _load_preprocessed_mask(mask_path: Path, ref_z: int) -> np.ndarray:
    """Load an ovary mask and resample to the same (Z, 512, 512) grid.
    Nearest-neighbour interpolation; returns uint8 binary."""
    m = sitk.ReadImage(str(mask_path))
    # Force size to match image after resampling
    m = ImgResample(m, out_spacing=OUT_SPACING, out_size=(OUT_XY, OUT_XY, ref_z),
                    is_label=True, pad_value=0)
    arr = sitk.GetArrayFromImage(m)
    return (arr > 0).astype(np.uint8)


def _pick_slices(image: np.ndarray, mask: np.ndarray, n: int = 3) -> list[int]:
    """Pick n z-slices to display. Prefer slices where the ovary mask is
    non-empty (so we see the interesting anatomy). Fall back to evenly spaced."""
    z = image.shape[0]
    ov_slices = [k for k in range(z) if mask[k].sum() > 0]
    if len(ov_slices) >= n:
        idxs = np.linspace(0, len(ov_slices) - 1, n).round().astype(int)
        return [ov_slices[i] for i in idxs]
    return list(np.linspace(z // 4, 3 * z // 4, n).round().astype(int))


def _find_synth_subject(synth_root: Path, requested: list[str]) -> str | None:
    """Return the first requested D2-9XX id that exists in synth_root, else
    None. Falls back to the first directory matching D2-9?? if nothing on
    the requested list is found. Returns None (no crash) if synth_root
    itself does not exist."""
    if not synth_root.exists():
        return None
    for s in requested:
        if (synth_root / s).is_dir():
            return s
    for d in sorted(synth_root.iterdir()):
        if d.is_dir() and d.name.startswith("D2-9"):
            return d.name
    return None


def _load_real_subject(real_dir: Path, subj: str) -> tuple[np.ndarray, np.ndarray]:
    img_path = real_dir / subj / f"{subj}_T2FS.nii.gz"
    mask_path = real_dir / subj / f"{subj}_ov.nii.gz"
    img = _load_preprocessed_image(img_path)
    mask = _load_preprocessed_mask(mask_path, img.shape[0])
    return img, mask


def _load_synth_subject(synth_root: Path, subj: str) -> tuple[np.ndarray, np.ndarray]:
    img_path = synth_root / subj / f"{subj}_T2FS.nii.gz"
    mask_path = synth_root / subj / f"{subj}_ov.nii.gz"
    img = _load_preprocessed_image(img_path)
    mask = _load_preprocessed_mask(mask_path, img.shape[0])
    return img, mask


def _plot_grid(ax, image: np.ndarray, mask: np.ndarray, z: int, title: str) -> None:
    ax.imshow(image[z], cmap="gray", vmin=0, vmax=1)
    if mask[z].any():
        # outline ovary in red
        ax.contour(mask[z], levels=[0.5], colors="red", linewidths=0.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def _plot_hist(ax, values: np.ndarray, label: str, color: str,
               bins: int = 80, alpha: float = 0.55) -> None:
    ax.hist(values, bins=bins, range=(0, 1), density=True, alpha=alpha,
            color=color, label=label, histtype="stepfilled", edgecolor=color,
            linewidth=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True, type=Path,
                    help="Root of UT-EndoMRI/D2_TCPW")
    ap.add_argument("--real-subjects", nargs="+", required=True,
                    help="D2-XXX subject IDs from the real cohort")
    ap.add_argument("--synth-dirs", nargs="+", required=True,
                    help="Space-separated 'label=path' pairs of synth dirs")
    ap.add_argument("--synth-subjects", nargs="+", required=True,
                    help="Preferred synth subject IDs (e.g. D2-900). "
                         "Falls back to first D2-9?? in each synth dir if missing.")
    ap.add_argument("--out-png", type=Path, required=True)
    args = ap.parse_args()

    # Parse synth dirs
    synth_variants: dict[str, Path] = {}
    for spec in args.synth_dirs:
        name, path = spec.split("=", 1)
        synth_variants[name] = Path(path)

    # Load real subjects
    print("[load] real D2 subjects")
    real_subjects = []
    for s in args.real_subjects:
        try:
            img, mask = _load_real_subject(args.real_dir, s)
            slices = _pick_slices(img, mask, n=3)
            real_subjects.append({"subj": s, "img": img, "mask": mask, "slices": slices})
            print(f"  {s}: shape={img.shape}, ov voxels={int(mask.sum())}, "
                  f"slices={slices}")
        except Exception as e:
            print(f"  {s}: FAILED {type(e).__name__}: {e}")

    # For each synth variant, pick one subject and load it
    print("[load] synth subjects (one per variant)")
    synth_loaded: dict[str, dict] = {}
    for name, root in synth_variants.items():
        picked = _find_synth_subject(root, args.synth_subjects)
        if picked is None:
            print(f"  {name}: NO subjects found in {root}")
            continue
        img, mask = _load_synth_subject(root, picked)
        slices = _pick_slices(img, mask, n=3)
        synth_loaded[name] = {"subj": picked, "img": img, "mask": mask, "slices": slices}
        print(f"  {name}: {picked} shape={img.shape}, ov voxels={int(mask.sum())}, "
              f"slices={slices}")

    # -----------------------------------------------------------------
    # Figure: 3 rows of visual comparison + 2 rows of histograms
    # -----------------------------------------------------------------
    n_variants = 1 + len(synth_loaded)   # real + each synth
    n_display_subj = min(len(real_subjects), 1)   # 1 real subject × 3 slices
    n_visual_rows = 3   # 3 z-slices from one real subject alongside 3 synth slices

    n_hist_rows = 2
    fig_h = 2.2 * n_visual_rows + 2.0 * n_hist_rows
    fig_w = 2.2 * n_variants
    fig, axes = plt.subplots(n_visual_rows + n_hist_rows, n_variants,
                              figsize=(fig_w, fig_h),
                              gridspec_kw={"height_ratios":
                                           [1] * n_visual_rows + [2, 2]})

    real0 = real_subjects[0]
    variant_order = ["real"] + list(synth_loaded.keys())

    # Visual grid — top n_visual_rows rows.
    # Use the REAL subject's chosen slices for EVERY variant so rows compare
    # like-for-like z-slices. Synth volumes with different Z depth get their
    # index clamped to a valid range; a trailing '*' on the printed z label
    # flags when clamping happened.
    ref_slices = real0["slices"]
    for r in range(n_visual_rows):
        z_ref = ref_slices[r] if r < len(ref_slices) else 0
        for c, vname in enumerate(variant_order):
            ax = axes[r, c]
            if vname == "real":
                _plot_grid(ax, real0["img"], real0["mask"], z_ref,
                           f"real {real0['subj']}  z={z_ref}" if r == 0
                           else f"z={z_ref}")
            else:
                v = synth_loaded[vname]
                z_v = min(int(z_ref), v["img"].shape[0] - 1)
                clamped = z_v != z_ref
                z_label = f"z={z_v}{'*' if clamped else ''}"
                _plot_grid(ax, v["img"], v["mask"], z_v,
                           f"{vname}  {v['subj']}  {z_label}" if r == 0
                           else z_label)

    # ------------------- Histograms -------------------
    # Row: overall body-region intensity (compare shapes across variants)
    def _body_voxel_values(entry: dict) -> np.ndarray:
        img = entry["img"]
        body = img > BODY_THRESHOLD
        return img[body]

    def _ov_voxel_values(entry: dict) -> np.ndarray:
        img = entry["img"]
        mask = entry["mask"]
        if mask.sum() == 0:
            return np.array([], dtype=np.float32)
        return img[mask.astype(bool)]

    # Row before-last: overall body-region intensity
    hist_row = n_visual_rows
    for c, vname in enumerate(variant_order):
        ax = axes[hist_row, c]
        if vname == "real":
            for e in real_subjects:
                vals = _body_voxel_values(e)
                _plot_hist(ax, vals, e["subj"], "tab:blue")
            title = "real body voxels"
        else:
            v = synth_loaded[vname]
            _plot_hist(ax, _body_voxel_values(v), v["subj"], "tab:orange")
            title = f"{vname} body voxels"
        # Mark the enhancement window
        ax.axvspan(O1, O2, alpha=0.15, color="green",
                   label=f"enh window [{O1}, {O2}]")
        ax.axvline(DEFAULT_OVARY_TARGET, linestyle="--", color="darkgreen",
                   linewidth=1, label=f"t={DEFAULT_OVARY_TARGET}")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("normalised intensity", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    # Last row: OVARY-ONLY intensity distribution
    hist_row = n_visual_rows + 1
    for c, vname in enumerate(variant_order):
        ax = axes[hist_row, c]
        collected = []
        if vname == "real":
            for e in real_subjects:
                collected.append(_ov_voxel_values(e))
                label = "real (pooled)"
            color = "tab:blue"
        else:
            v = synth_loaded[vname]
            collected.append(_ov_voxel_values(v))
            label = f"{vname} {v['subj']}"
            color = "tab:orange"
        pooled = np.concatenate([c for c in collected if c.size > 0]) \
                    if any(c.size > 0 for c in collected) else np.array([])
        if pooled.size > 0:
            _plot_hist(ax, pooled, label, color, bins=60, alpha=0.7)
            mean_v = float(pooled.mean())
            median_v = float(np.median(pooled))
            ax.axvline(mean_v, linestyle=":", color="black",
                       linewidth=1, label=f"mean={mean_v:.3f}")
            ax.axvline(median_v, linestyle="-.", color="black",
                       linewidth=1, label=f"median={median_v:.3f}")
        else:
            ax.text(0.5, 0.5, "no ovary voxels", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.axvspan(O1, O2, alpha=0.15, color="green",
                   label=f"enh window [{O1}, {O2}]")
        ax.set_xlim(0, 1)
        ax.set_title(f"{'real' if vname == 'real' else vname} OVARY voxels",
                     fontsize=8)
        ax.set_xlabel("normalised intensity", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right")

    fig.suptitle("Synth vs real D2 T2FS — after RAovSeg percentile-clip+minmax "
                 "(no enhancement).\nGreen band = enhancement window "
                 f"[{O1}, {O2}]. Ovaries must land here for the enhancement "
                 "to fire.", fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {args.out_png}")

    # -----------------------------------------------------------------
    # Numeric report — the actionable output
    # -----------------------------------------------------------------
    print("\n=== ovary voxel intensity summary (post RAovSeg normalize) ===")
    print(f"{'variant':<20} {'n_vox':>10} {'mean':>8} {'median':>8} "
          f"{'p10':>8} {'p90':>8} {'in_window%':>11}")

    def _row(name: str, vals: np.ndarray) -> None:
        n = vals.size
        if n == 0:
            print(f"{name:<20} {n:>10} {'-':>8} {'-':>8} "
                  f"{'-':>8} {'-':>8} {'-':>11}")
            return
        in_win = ((vals >= O1) & (vals <= O2)).mean() * 100
        print(f"{name:<20} {n:>10} "
              f"{vals.mean():>8.3f} {np.median(vals):>8.3f} "
              f"{np.percentile(vals, 10):>8.3f} "
              f"{np.percentile(vals, 90):>8.3f} "
              f"{in_win:>10.1f}%")

    all_real = np.concatenate([_ov_voxel_values(e) for e in real_subjects
                                if _ov_voxel_values(e).size > 0]) \
                    if real_subjects else np.array([])
    _row("real (pooled)", all_real)
    for name, v in synth_loaded.items():
        _row(name, _ov_voxel_values(v))

    # Suggest re-calibration
    print("\n=== Path B ovary target suggestion ===")
    print("If synth ovary voxels are outside the enhancement window,")
    print(f"reassemble with --ovary-target-intensity set to the value that")
    print(f"maps the current synth-ovary mean to the middle of [{O1}, {O2}] "
          f"(target = {(O1 + O2) / 2:.3f}).")
    print("Formula: new_t = old_t + (target - synth_ovary_mean)")


if __name__ == "__main__":
    main()
