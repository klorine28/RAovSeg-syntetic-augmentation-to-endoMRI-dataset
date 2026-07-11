"""
Dissertation mechanism figures — real vs synth D2 T2FS.

Sister script to `diagnose_synth_vs_real.py`, but produces publication-quality
figures (matching `make_dissertation_figures.ipynb` house style) and a numeric
summary table.

Supports ONE synth variant (`--synth-dir`) or SEVERAL (`--synth-dirs
name=path ...`). With several, real + every variant are rendered on a single
overlay grid and single histograms (one curve per variant).

Figures written to `--out-dir` (default `figures/`):
    fig_mech_overlay.png     — rows = real + each synth variant, ovary contoured
    fig_mech_body_hist.png   — body-voxel intensity, real + each variant,
                                with the [0.22, 0.30] enhancement band
    fig_mech_ovary_hist.png  — pooled ovary-voxel intensity, real + each variant,
                                with means and the enhancement band

Numeric summary also written to `mech_ovary_intensity_table.csv`.

Usage (single variant, from repo root):
    python -m src.RaovSeg_recreation.mechanism_figures \
        --real-dir UT-EndoMRI/D2_TCPW \
        --real-subjects D2-016 D2-017 D2-024 \
        --synth-dir exp1c_spade_samples --synth-label "Synth (1c SPADE)" \
        --synth-subjects D2-900 D2-901 D2-902 \
        --out-dir figures

Usage (multi variant):
    python -m src.RaovSeg_recreation.mechanism_figures \
        --real-dir UT-EndoMRI/D2_TCPW \
        --real-subjects D2-016 D2-017 D2-024 \
        --synth-dirs "1c SPADE (P1)=exp1c_spade_samples" \
                     "exp2 (P2)=exp2_samples_volumes" \
                     "exp2_lam05 (P2)=exp2_lam05_samples_volumes" \
        --synth-subjects D2-900 D2-901 D2-902 \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# Inlined from RAovSeg/RAovSeg_tools.py to avoid a torch import at plot time.


def ImgNorm(img: np.ndarray, norm_type: str = "percentile_clip",
            percentile_low: float | None = None,
            percentile_high: float | None = None) -> np.ndarray:
    if norm_type == "percentile_clip":
        mn = np.percentile(img, percentile_low)
        img[img < mn] = mn
        mx = np.percentile(img, percentile_high)
        img[img > mx] = mx
        return (img - mn) / (mx - mn)
    if norm_type == "minmax":
        mn, mx = float(np.min(img)), float(np.max(img))
        return (img - mn) / (mx - mn)
    raise ValueError(f"unknown norm_type {norm_type!r}")


def ImgResample(image: sitk.Image, out_spacing=(0.5, 0.5, 0.5),
                out_size=None, is_label: bool = False,
                pad_value: float = 0) -> sitk.Image:
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    if out_size is None:
        out_size = np.round(
            original_size * original_spacing / np.array(out_spacing)
        ).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing), -1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing)
    r.SetSize(out_size.tolist())
    r.SetOutputDirection(image.GetDirection())
    r.SetOutputOrigin(out_origin.tolist())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(pad_value)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return r.Execute(sitk.Cast(image, sitk.sitkFloat32))


OUT_SPACING = (0.35, 0.35, 6.0)
OUT_XY = 512
PERCENTILE_LOW = 1
PERCENTILE_HIGH = 99
O1 = 0.22
O2 = 0.30
DEFAULT_OVARY_TARGET = 0.26
BODY_THRESHOLD = 0.05

# Real is always blue; synth variants cycle through this palette.
REAL_COLOR = "#4C72B0"
SYNTH_PALETTE = ["#C44E52", "#8172B3", "#DD8452", "#55A868", "#937860"]
BAND_COLOR = "#2CA02C"

RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.frameon": False,
}


@dataclass
class Loaded:
    subj: str
    img: np.ndarray
    mask: np.ndarray
    slice_idxs: list[int]


@dataclass
class Group:
    """A row/curve in the figures: real, or one synth variant."""
    label: str
    color: str
    is_real: bool
    loaded: list[Loaded] = field(default_factory=list)


def _out_size(img: sitk.Image) -> tuple[int, int, int]:
    return (OUT_XY, OUT_XY, int(img.GetSize()[2]))


def _load_image(img_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(img_path), sitk.sitkFloat64)
    img = ImgResample(img, out_spacing=OUT_SPACING, out_size=_out_size(img),
                      is_label=False, pad_value=0)
    arr = sitk.GetArrayFromImage(img)
    arr = ImgNorm(arr, norm_type="percentile_clip",
                  percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH)
    return arr.astype(np.float32)


def _load_mask(mask_path: Path, ref_z: int) -> np.ndarray:
    m = sitk.ReadImage(str(mask_path))
    m = ImgResample(m, out_spacing=OUT_SPACING,
                    out_size=(OUT_XY, OUT_XY, ref_z),
                    is_label=True, pad_value=0)
    return (sitk.GetArrayFromImage(m) > 0).astype(np.uint8)


def _pick_slices(image: np.ndarray, mask: np.ndarray, n: int = 1) -> list[int]:
    z = image.shape[0]
    ov = [k for k in range(z) if mask[k].sum() > 0]
    if len(ov) >= n:
        idxs = np.linspace(0, len(ov) - 1, n).round().astype(int)
        return [ov[i] for i in idxs]
    return list(np.linspace(z // 4, 3 * z // 4, n).round().astype(int))


def _load_subject(root: Path, subj: str, n_slices: int = 1) -> Loaded:
    img = _load_image(root / subj / f"{subj}_T2FS.nii.gz")
    mask = _load_mask(root / subj / f"{subj}_ov.nii.gz", img.shape[0])
    return Loaded(subj=subj, img=img, mask=mask,
                  slice_idxs=_pick_slices(img, mask, n=n_slices))


def _body_voxels(l: Loaded) -> np.ndarray:
    return l.img[l.img > BODY_THRESHOLD]


def _ovary_voxels(l: Loaded) -> np.ndarray:
    if l.mask.sum() == 0:
        return np.array([], dtype=np.float32)
    return l.img[l.mask.astype(bool)]


def _pool_ovary(g: Group) -> np.ndarray:
    parts = [v for v in (_ovary_voxels(e) for e in g.loaded) if v.size > 0]
    return np.concatenate(parts) if parts else np.array([])


def _pool_body(g: Group) -> np.ndarray:
    parts = [_body_voxels(e) for e in g.loaded if e.img.size > 0]
    return np.concatenate(parts) if parts else np.array([])


def fig_overlay(groups: list[Group], n_cols: int, out_path: Path) -> None:
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.7 * n_cols, 2.75 * n_rows))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and n_rows != 1:
        axes = axes.T
    for r, g in enumerate(groups):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= len(g.loaded):
                ax.axis("off")
                continue
            entry = g.loaded[c]
            z = entry.slice_idxs[0]
            ax.imshow(entry.img[z], cmap="gray", vmin=0, vmax=1)
            if entry.mask[z].any():
                ax.contour(entry.mask[z], levels=[0.5], colors="red", linewidths=1.0)
            ax.set_title(f"{g.label} — {entry.subj} (z={z})", fontsize=8.5,
                         color=(REAL_COLOR if g.is_real else g.color))
    fig.suptitle("Real vs synth D2 T2FS — ovary contoured in red",
                 fontsize=12.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


def fig_body_hist(groups: list[Group], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.axvspan(O1, O2, alpha=0.16, color=BAND_COLOR,
               label=f"enhancement window [{O1}, {O2}]")
    ax.axvline(DEFAULT_OVARY_TARGET, ls="--", color="darkgreen", lw=1.2,
               label=f"Path B target t = {DEFAULT_OVARY_TARGET}")
    for g in groups:
        pool = _pool_body(g)
        if pool.size == 0:
            continue
        ax.hist(pool, bins=80, range=(0, 1), density=True, histtype="step",
                color=g.color, lw=1.9,
                label=f"{g.label} (n={len(g.loaded)})")
        ax.hist(pool, bins=80, range=(0, 1), density=True, histtype="stepfilled",
                color=g.color, alpha=0.10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised intensity (post RAovSeg percentile clip + minmax)")
    ax.set_ylabel("Density")
    ax.set_title("Body-voxel intensity distribution")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


def fig_ovary_hist(groups: list[Group], out_path: Path,
                   summary_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.axvspan(O1, O2, alpha=0.16, color=BAND_COLOR,
               label=f"enhancement window [{O1}, {O2}]")
    ax.axvline(DEFAULT_OVARY_TARGET, ls="--", color="darkgreen", lw=1.2,
               label=f"Path B target t = {DEFAULT_OVARY_TARGET}")
    for g in groups:
        pool = _pool_ovary(g)
        if pool.size == 0:
            continue
        ax.hist(pool, bins=60, range=(0, 1), density=True, histtype="step",
                color=g.color, lw=2.0,
                label=f"{g.label} (n={pool.size:,} vox, mean={pool.mean():.3f})")
        ax.hist(pool, bins=60, range=(0, 1), density=True, histtype="stepfilled",
                color=g.color, alpha=0.12)
        ax.axvline(pool.mean(), ls=":", color=g.color, lw=1.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised intensity (post RAovSeg percentile clip + minmax)")
    ax.set_ylabel("Density")
    ax.set_title("Ovary-voxel intensity distribution — the mechanism figure")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")

    def _row(name: str, vals: np.ndarray) -> dict:
        if vals.size == 0:
            return {"variant": name, "n_vox": 0, "mean": None, "median": None,
                    "p10": None, "p90": None, "pct_in_window": None}
        return {
            "variant": name,
            "n_vox": int(vals.size),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90)),
            "pct_in_window": float(((vals >= O1) & (vals <= O2)).mean() * 100.0),
        }

    for g in groups:
        summary_rows.append(_row(f"{g.label} (pooled)", _pool_ovary(g)))
        for e in g.loaded:
            tag = "real" if g.is_real else g.label
            summary_rows.append(_row(f"{tag} / {e.subj}", _ovary_voxels(e)))


def _print_and_save_table(rows: list[dict], out_csv: Path) -> None:
    print("\n=== ovary voxel intensity summary (post RAovSeg normalize) ===")
    print(f"{'variant':<36} {'n_vox':>10} {'mean':>8} {'median':>8} "
          f"{'p10':>8} {'p90':>8} {'in_window%':>11}")
    for r in rows:
        if r["n_vox"] == 0:
            print(f"{r['variant']:<36} {r['n_vox']:>10} {'-':>8} {'-':>8} "
                  f"{'-':>8} {'-':>8} {'-':>11}")
            continue
        print(f"{r['variant']:<36} {r['n_vox']:>10} "
              f"{r['mean']:>8.3f} {r['median']:>8.3f} "
              f"{r['p10']:>8.3f} {r['p90']:>8.3f} "
              f"{r['pct_in_window']:>10.1f}%")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[saved] {out_csv}")


def _parse_synth_dirs(args) -> list[tuple[str, Path]]:
    """Return ordered [(label, path)] from either --synth-dirs or --synth-dir."""
    if args.synth_dirs:
        out = []
        for item in args.synth_dirs:
            if "=" not in item:
                raise SystemExit(f"--synth-dirs entries must be name=path, got {item!r}")
            name, path = item.split("=", 1)
            out.append((name.strip(), Path(path.strip())))
        return out
    if args.synth_dir:
        return [(args.synth_label, args.synth_dir)]
    raise SystemExit("Provide either --synth-dir or --synth-dirs.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", type=Path, required=True)
    ap.add_argument("--real-subjects", nargs="+", required=True)
    ap.add_argument("--synth-dir", type=Path, default=None,
                    help="Single synth variant directory.")
    ap.add_argument("--synth-dirs", nargs="+", default=None,
                    help="Multiple variants as name=path (space separated).")
    ap.add_argument("--synth-subjects", nargs="+", required=True,
                    help="Synth subject IDs, shared across all variants.")
    ap.add_argument("--real-label", default="Real D2")
    ap.add_argument("--synth-label", default="Synth")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    plt.rcParams.update(RC)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- real group ---
    print(f"[load] {len(args.real_subjects)} real subjects from {args.real_dir}")
    real_group = Group(label=args.real_label, color=REAL_COLOR, is_real=True)
    for s in args.real_subjects:
        try:
            l = _load_subject(args.real_dir, s, n_slices=1)
            real_group.loaded.append(l)
            print(f"  {s}: shape={l.img.shape}, ov_vox={int(l.mask.sum())}, z={l.slice_idxs}")
        except Exception as e:
            print(f"  {s}: FAILED {type(e).__name__}: {e}")

    # --- synth groups ---
    synth_specs = _parse_synth_dirs(args)
    synth_groups: list[Group] = []
    for i, (label, root) in enumerate(synth_specs):
        g = Group(label=label, color=SYNTH_PALETTE[i % len(SYNTH_PALETTE)], is_real=False)
        print(f"[load] variant {label!r} from {root}")
        for s in args.synth_subjects:
            try:
                l = _load_subject(root, s, n_slices=1)
                g.loaded.append(l)
                print(f"  {s}: shape={l.img.shape}, ov_vox={int(l.mask.sum())}, z={l.slice_idxs}")
            except Exception as e:
                print(f"  {s}: FAILED {type(e).__name__}: {e}")
        synth_groups.append(g)

    if not real_group.loaded or not any(g.loaded for g in synth_groups):
        raise SystemExit("Need at least one real and one synth subject to render.")

    groups = [real_group] + [g for g in synth_groups if g.loaded]
    n_cols = max(len(g.loaded) for g in groups)

    summary_rows: list[dict] = []
    fig_overlay(groups, n_cols, args.out_dir / "fig_mech_overlay.png")
    fig_body_hist(groups, args.out_dir / "fig_mech_body_hist.png")
    fig_ovary_hist(groups, args.out_dir / "fig_mech_ovary_hist.png", summary_rows)
    _print_and_save_table(summary_rows, args.out_dir / "mech_ovary_intensity_table.csv")


if __name__ == "__main__":
    main()
