#!/usr/bin/env python3
"""
Exp 1 — offline ovary-intensity retargeting for the dose-response experiment.

Takes an existing synth volume directory (produced by
`assemble_synthetic_volumes.py`) and produces a copy at a new ovary target
intensity, WITHOUT a GPU pass through the diffusion model. Path B is a per-
volume additive offset applied to the ovary region only, so it composes
freely with a second offset — inverting the original 0.26 target and applying
a new one is equivalent to producing that volume fresh with the new target.

For each subject D2-9NN under --src-dir:
  1. Load {subject}_T2FS.nii.gz (float, raw intensity range) and
     {subject}_ov.nii.gz (binary mask; already resampled to the raw real
     subject's frame by assemble).
  2. Compute per-volume p1/p99 (matches RAovSeg's percentile-clip window).
  3. Compute the raw intensity that lands at --target-normalized after
     RAovSeg's clip+minmax: target_raw = p1 + t * (p99 - p1).
  4. Shift ovary voxels: img[ov] += target_raw - img[ov].mean().
  5. Write the retargeted volume to --out-dir/{subject}/{subject}_T2FS.nii.gz
     and copy the unchanged ov mask alongside it.

Also writes an `intensity_retarget_manifest.csv` recording the achieved
ovary mean per volume — the extreme targets clip against RAovSeg's [p1, p99]
window at percentile-clip time, so the achieved post-normalization mean is
NOT identical to --target-normalized. The dose-response x-axis should use
the achieved values, not the nominal ones (per the design doc).

Usage:
    python scripts/retarget_ovary_intensity.py \\
        --src-dir  synth_volumes/exp1c_spade_fixed \\
        --out-dir  synth_volumes/exp1c_spade_fixed_t005 \\
        --target-normalized 0.05
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SUBJECT_RE = re.compile(r"^D2-\d{3}$")


def retarget_subject(t2fs_path: Path, ov_path: Path, out_dir: Path,
                     target_normalized: float) -> dict:
    """Retarget a single subject. Returns a manifest row."""
    img_sitk = sitk.ReadImage(str(t2fs_path), sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

    ov_sitk = sitk.ReadImage(str(ov_path))
    ov_arr = sitk.GetArrayFromImage(ov_sitk)
    ov_mask = ov_arr > 0

    if ov_mask.shape != img_arr.shape:
        raise ValueError(
            f"{t2fs_path}: image {img_arr.shape} vs mask {ov_mask.shape} mismatch")

    # p1/p99 mirror RAovSeg's percentile-clip step. Recompute per-subject so
    # the raw target we hit is the one that actually lands at
    # target_normalized after the downstream normalisation runs.
    p1, p99 = np.percentile(img_arr, [1, 99])
    p1, p99 = float(p1), float(p99)
    span = max(p99 - p1, 1e-6)

    if not ov_mask.any():
        # Nothing to do — write the file back unchanged and report NaNs.
        sitk.WriteImage(img_sitk, str(out_dir / t2fs_path.name))
        return {
            "subject": t2fs_path.stem.split("_")[0],
            "p1": p1,
            "p99": p99,
            "target_normalized": target_normalized,
            "target_raw": p1 + target_normalized * span,
            "ovary_mean_pre_raw": float("nan"),
            "ovary_mean_post_raw": float("nan"),
            "achieved_normalized_mean": float("nan"),
            "ovary_voxels": 0,
            "voxels_clipped_low": 0,
            "voxels_clipped_high": 0,
        }

    target_raw = p1 + target_normalized * span
    ovary_vals = img_arr[ov_mask]
    ovary_mean_pre = float(ovary_vals.mean())
    offset = target_raw - ovary_mean_pre

    img_arr[ov_mask] = img_arr[ov_mask] + offset
    ovary_mean_post = float(img_arr[ov_mask].mean())

    # RAovSeg's clip+minmax will pin anything <= p1 to 0 and anything >= p99
    # to 1; report the count so extreme targets are legible in the manifest.
    clipped_low = int((img_arr[ov_mask] < p1).sum())
    clipped_high = int((img_arr[ov_mask] > p99).sum())

    # Achieved post-normalization ovary mean, matching RAovSeg's math:
    # clip to [p1, p99] then minmax → [0, 1].
    ov_after = np.clip(img_arr[ov_mask], p1, p99)
    achieved_norm = float(((ov_after - p1) / span).mean())

    out_img = sitk.GetImageFromArray(img_arr)
    out_img.CopyInformation(img_sitk)
    sitk.WriteImage(out_img, str(out_dir / t2fs_path.name))

    return {
        "subject": t2fs_path.stem.split("_")[0],
        "p1": p1,
        "p99": p99,
        "target_normalized": target_normalized,
        "target_raw": target_raw,
        "ovary_mean_pre_raw": ovary_mean_pre,
        "ovary_mean_post_raw": ovary_mean_post,
        "achieved_normalized_mean": achieved_norm,
        "ovary_voxels": int(ov_mask.sum()),
        "voxels_clipped_low": clipped_low,
        "voxels_clipped_high": clipped_high,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True,
                        help="Existing synth volume dir (subjects: D2-9NN/*_T2FS.nii.gz)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to write the retargeted subjects (mirrors src-dir layout)")
    parser.add_argument("--target-normalized", type=float, required=True,
                        help="Desired post-normalization ovary mean intensity in [0, 1]. "
                             "RAovSeg's published enhancement window is [0.22, 0.30].")
    parser.add_argument("--sequence", default="T2FS",
                        help="MRI sequence suffix (default T2FS)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite --out-dir if it exists")
    args = parser.parse_args()

    if not args.src_dir.exists():
        print(f"ERROR: src-dir not found: {args.src_dir}", file=sys.stderr)
        return 1
    if not (0.0 <= args.target_normalized <= 1.0):
        print(f"ERROR: --target-normalized must be in [0, 1], got "
              f"{args.target_normalized}", file=sys.stderr)
        return 1

    if args.out_dir.exists() and not args.overwrite:
        print(f"ERROR: --out-dir exists ({args.out_dir}); pass --overwrite to replace",
              file=sys.stderr)
        return 1
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(d for d in args.src_dir.iterdir()
                      if d.is_dir() and SUBJECT_RE.match(d.name))
    if not subjects:
        print(f"ERROR: no D2-NNN subject dirs found in {args.src_dir}",
              file=sys.stderr)
        return 1

    print(f"[retarget] {len(subjects)} subjects, target_normalized={args.target_normalized}")
    rows: list[dict] = []
    for subj_dir in subjects:
        sid = subj_dir.name
        t2fs = subj_dir / f"{sid}_{args.sequence}.nii.gz"
        ov = subj_dir / f"{sid}_ov.nii.gz"
        if not t2fs.exists() or not ov.exists():
            print(f"  SKIP {sid}: missing {t2fs.name} or {ov.name}")
            continue

        out_subj = args.out_dir / sid
        out_subj.mkdir(parents=True, exist_ok=True)
        row = retarget_subject(t2fs, ov, out_subj, args.target_normalized)
        # Copy the ov mask unchanged — the mask is what RAovSeg's preprocess
        # will resample; we only altered the T2FS intensities.
        shutil.copy2(ov, out_subj / ov.name)
        rows.append(row)
        print(f"  {sid}: ov_mean {row['ovary_mean_pre_raw']:.4f} → "
              f"{row['ovary_mean_post_raw']:.4f}  "
              f"(post-norm ≈ {row['achieved_normalized_mean']:.4f}, "
              f"clipped low/high = {row['voxels_clipped_low']}/{row['voxels_clipped_high']})")

    manifest = args.out_dir / "intensity_retarget_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[retarget] wrote {manifest}")

    achieved = np.array([r["achieved_normalized_mean"] for r in rows
                         if not np.isnan(r["achieved_normalized_mean"])])
    if achieved.size:
        print(f"[retarget] pooled achieved normalized ovary mean: "
              f"{achieved.mean():.4f} ± {achieved.std():.4f} "
              f"(nominal: {args.target_normalized})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
