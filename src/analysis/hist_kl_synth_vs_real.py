#!/usr/bin/env python3
"""
Compute intensity-histogram KL divergence between a synth volume dir and a
pool of raw real subjects. Cheap per-Tier-1-trial diagnostic — reports both
"how much synth's intensity distribution deviates from real" (a proxy for
realism) and the underlying histograms so drift can be inspected offline.

Both synth and real go through the same normalisation pipeline the
segmenter (RAovSeg) applies:
    resample to isotropic → percentile-clip (1st/99th) → minmax → [0, 1]

Then the two pooled distributions are histogram-KL'd:
    KL(P_real || P_synth) = Σ p_r · log(p_r / p_s)

Kept lightweight (numpy + SimpleITK only). No torch dependency, no model
load — runs in <15 s per trial on CPU.

Usage:
    python -m src.analysis.hist_kl_synth_vs_real \\
        --synth-dir  /path/to/synth/exp1c_spade \\
        --real-dir   /path/to/UT-EndoMRI/D2_TCPW \\
        --real-subjects D2-001 D2-007 D2-008 ... \\
        --sequence T2FS \\
        --out-json  metrics/hist_kl.json

Or (auto-pick real subjects from a split JSON):
    python -m src.analysis.hist_kl_synth_vs_real \\
        --synth-dir  /path/to/synth/exp1c_spade \\
        --real-dir   /path/to/UT-EndoMRI/D2_TCPW \\
        --real-split-file data/splits/d2_generator_split.json \\
        --real-split-key train \\
        --sequence T2FS \\
        --out-json  metrics/hist_kl.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


DEFAULT_OUT_SPACING = (0.35, 0.35, 6.0)
DEFAULT_OUT_XY = 512
DEFAULT_PERCENTILE_LOW = 1
DEFAULT_PERCENTILE_HIGH = 99


def _img_resample(image: sitk.Image, out_spacing=DEFAULT_OUT_SPACING,
                  out_xy: int = DEFAULT_OUT_XY, pad_value: float = 0.0) -> sitk.Image:
    """Resample to isotropic spacing, matching what RAovSeg's preprocess does."""
    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())
    out_size = (out_xy, out_xy, int(original_size[2]))
    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing), -1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)
    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing)
    r.SetSize(list(out_size))
    r.SetOutputDirection(image.GetDirection())
    r.SetOutputOrigin(out_origin.tolist())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(pad_value)
    r.SetInterpolator(sitk.sitkLinear)
    return r.Execute(sitk.Cast(image, sitk.sitkFloat32))


def _load_and_normalise(path: Path,
                        percentile_low: float,
                        percentile_high: float) -> np.ndarray:
    """Load a NIfTI, resample, percentile-clip + minmax to [0, 1]. Returns
    a flat float32 array of voxel intensities."""
    img = sitk.ReadImage(str(path), sitk.sitkFloat32)
    img = _img_resample(img)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mn = float(np.percentile(arr, percentile_low))
    mx = float(np.percentile(arr, percentile_high))
    arr = np.clip(arr, mn, mx)
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr.ravel()


def _pool_from_dir(root: Path, subjects: list[str],
                   sequence: str,
                   percentile_low: float,
                   percentile_high: float,
                   file_pattern: str = "{subj}_{sequence}.nii.gz",
                   max_subjects: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Load each subject's normalised voxels into one pooled 1-D array.
    Returns (pooled_voxels, list_of_subjects_actually_loaded)."""
    if max_subjects is not None:
        subjects = subjects[:max_subjects]
    pools = []
    loaded = []
    for subj in subjects:
        pth = root / subj / file_pattern.format(subj=subj, sequence=sequence)
        if not pth.exists():
            print(f"  skip {subj}: file not found at {pth}")
            continue
        try:
            v = _load_and_normalise(pth, percentile_low, percentile_high)
            pools.append(v)
            loaded.append(subj)
        except Exception as e:
            print(f"  skip {subj}: {type(e).__name__}: {e}")
            continue
    if not pools:
        raise RuntimeError(f"no subjects successfully loaded from {root}")
    return np.concatenate(pools), loaded


def _hist_kl(p_r: np.ndarray, p_s: np.ndarray, n_bins: int = 100,
             range_: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return KL(P_real || P_synth) and (symmetric) Jensen-Shannon divergence,
    plus the normalised histograms so callers can save them."""
    edges = np.linspace(range_[0], range_[1], n_bins + 1)
    h_r, _ = np.histogram(p_r, bins=edges, density=False)
    h_s, _ = np.histogram(p_s, bins=edges, density=False)
    # Laplace smoothing to avoid log(0)
    h_r = h_r.astype(np.float64) + 1e-10
    h_s = h_s.astype(np.float64) + 1e-10
    h_r /= h_r.sum()
    h_s /= h_s.sum()
    kl = float(np.sum(h_r * np.log(h_r / h_s)))
    m = 0.5 * (h_r + h_s)
    jsd = 0.5 * float(np.sum(h_r * np.log(h_r / m))) + \
          0.5 * float(np.sum(h_s * np.log(h_s / m)))
    return kl, jsd, h_r, h_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--synth-dir", required=True, type=Path,
                    help="Root dir containing synth D2-9XX/... NIfTI subject folders")
    ap.add_argument("--real-dir", required=True, type=Path,
                    help="Root dir containing real D2-XXX/... NIfTI subject folders")
    ap.add_argument("--sequence", default="T2FS",
                    help="Sequence suffix; file is <subj>_<sequence>.nii.gz (default T2FS)")

    # Real subjects: pass either explicit list or a split file
    ap.add_argument("--real-subjects", nargs="+", default=None,
                    help="Explicit list of real subject IDs to include")
    ap.add_argument("--real-split-file", type=Path, default=None,
                    help="Path to a split JSON to read real subjects from")
    ap.add_argument("--real-split-key", default="train",
                    help="Key inside the split JSON (default: train)")
    ap.add_argument("--max-real", type=int, default=None,
                    help="Cap number of real subjects (default: no cap)")

    ap.add_argument("--n-bins", type=int, default=100)
    ap.add_argument("--percentile-low", type=float, default=DEFAULT_PERCENTILE_LOW)
    ap.add_argument("--percentile-high", type=float, default=DEFAULT_PERCENTILE_HIGH)
    ap.add_argument("--out-json", required=True, type=Path,
                    help="Where to write the metrics + histograms JSON")
    ap.add_argument("--label", default="",
                    help="Optional tag written into the JSON for downstream aggregation")
    args = ap.parse_args()

    # --- Resolve real subjects list ---
    if args.real_subjects:
        real_subjects = list(args.real_subjects)
    elif args.real_split_file:
        with open(args.real_split_file) as f:
            split = json.load(f)
        real_subjects = split[args.real_split_key]
    else:
        # Fallback: use every subdirectory of --real-dir
        real_subjects = sorted(p.name for p in args.real_dir.iterdir() if p.is_dir())
    print(f"[hist_kl] real subjects: {len(real_subjects)}")

    # --- Synth subjects: every subdir of --synth-dir ---
    synth_subjects = sorted(p.name for p in args.synth_dir.iterdir() if p.is_dir())
    print(f"[hist_kl] synth subjects: {len(synth_subjects)}")

    # --- Load pooled voxels ---
    print("[hist_kl] loading real ...")
    real_voxels, real_loaded = _pool_from_dir(
        args.real_dir, real_subjects, args.sequence,
        args.percentile_low, args.percentile_high,
        max_subjects=args.max_real,
    )
    print(f"[hist_kl] real pool: {real_voxels.size:,} voxels from {len(real_loaded)} subjects")

    print("[hist_kl] loading synth ...")
    synth_voxels, synth_loaded = _pool_from_dir(
        args.synth_dir, synth_subjects, args.sequence,
        args.percentile_low, args.percentile_high,
    )
    print(f"[hist_kl] synth pool: {synth_voxels.size:,} voxels from {len(synth_loaded)} subjects")

    # --- Compute divergences ---
    kl, jsd, h_r, h_s = _hist_kl(real_voxels, synth_voxels, n_bins=args.n_bins)
    edges = np.linspace(0.0, 1.0, args.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    print(f"[hist_kl] KL(real||synth) = {kl:.4f}")
    print(f"[hist_kl] JSD              = {jsd:.4f}")

    # --- Save ---
    out = {
        "label":         args.label,
        "synth_dir":     str(args.synth_dir),
        "real_dir":      str(args.real_dir),
        "sequence":      args.sequence,
        "n_bins":        int(args.n_bins),
        "percentile_low":  float(args.percentile_low),
        "percentile_high": float(args.percentile_high),
        "n_real_subjects":  len(real_loaded),
        "n_synth_subjects": len(synth_loaded),
        "n_real_voxels":    int(real_voxels.size),
        "n_synth_voxels":   int(synth_voxels.size),
        "kl_real_synth":    float(kl),
        "jsd":              float(jsd),
        "hist_bin_centers": centers.tolist(),
        "hist_real":        h_r.tolist(),
        "hist_synth":       h_s.tolist(),
        "real_subjects":    real_loaded,
        "synth_subjects":   synth_loaded,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()
