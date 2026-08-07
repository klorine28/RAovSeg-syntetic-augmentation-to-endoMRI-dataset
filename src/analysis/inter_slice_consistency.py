#!/usr/bin/env python3
"""
Inter-slice consistency calibration on real D2 T2FS volumes.

Computes per-subject metrics that measure how "3D-coherent" real D2 volumes
are, then aggregates them across the cohort to give a reference distribution.
Synthetic volumes produced by the DDPM assembly pipeline can then be scored
against this profile to detect inter-slice jitter that would otherwise pass
undetected.

Metrics (per organ + global for the image):
    adjacent_slice_mask_dice
        DSC between mask[z] and mask[z+1], computed only on interior pairs
        (both slices nonempty). Measures shape stability along z.
    centroid_drift_mm
        L2 distance in mm between the centroid of the organ on slice z and
        slice z+1. Uses the SimpleITK spacing (x, y) for the in-plane
        conversion. Measures how much the organ moves between slices.
    adjacent_slice_L1
        Mean |I[z+1] - I[z]| after per-volume min-max intensity
        normalisation to [0, 1]. Global image-level jitter.
    z_axis_TV
        Same as adjacent_slice_L1 but summed rather than averaged, and
        reported per mm of slice spacing.

Usage:
    python -m src.analysis.inter_slice_consistency \
        --data-dir /path/to/D2_TCPW \
        --out metrics/real_iscs_profile.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

ORGAN_SUFFIXES = {
    "ovary": "_ov.nii.gz",
    "uterus": "_ut.nii.gz",
    "endometrioma": "_em.nii.gz",
    "cyst": "_cy.nii.gz",
}


@dataclass
class OrganMetrics:
    n_pairs: int = 0
    n_interior_pairs: int = 0
    adjacent_slice_mask_dice: Optional[float] = None
    centroid_drift_mm: Optional[float] = None
    n_slices_with_mask: int = 0


@dataclass
class SubjectMetrics:
    subject: str
    n_slices: int
    voxel_spacing_xyz: tuple
    organs: dict = field(default_factory=dict)
    adjacent_slice_L1: Optional[float] = None
    z_axis_TV_per_mm: Optional[float] = None


def slicewise_dice(a: np.ndarray, b: np.ndarray) -> float:
    """DSC between two binary masks. Returns NaN if both are empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def slice_centroid(mask_2d: np.ndarray) -> Optional[np.ndarray]:
    """Return (row, col) centroid of a 2D binary mask, or None if empty."""
    idx = np.argwhere(mask_2d.astype(bool))
    if idx.size == 0:
        return None
    return idx.mean(axis=0)


def compute_organ_metrics(
    mask_3d: np.ndarray, spacing_xy_mm: tuple
) -> OrganMetrics:
    """Compute inter-slice metrics for a single organ mask volume.

    mask_3d has shape (Z, Y, X) — SimpleITK's GetArrayFromImage convention.
    spacing_xy_mm is (spacing_x, spacing_y).
    """
    m = OrganMetrics()
    m.n_slices_with_mask = int((mask_3d.reshape(mask_3d.shape[0], -1).sum(axis=1) > 0).sum())

    dice_pairs = []
    drift_pairs = []
    for z in range(mask_3d.shape[0] - 1):
        a = mask_3d[z]
        b = mask_3d[z + 1]
        a_nonempty = a.any()
        b_nonempty = b.any()
        m.n_pairs += 1
        if a_nonempty and b_nonempty:
            m.n_interior_pairs += 1
            dice_pairs.append(slicewise_dice(a, b))
            ca = slice_centroid(a)
            cb = slice_centroid(b)
            # Centroid indices are (row, col) → (y, x).
            drow_mm = (cb[0] - ca[0]) * spacing_xy_mm[1]
            dcol_mm = (cb[1] - ca[1]) * spacing_xy_mm[0]
            drift_pairs.append(float(np.hypot(drow_mm, dcol_mm)))

    if dice_pairs:
        m.adjacent_slice_mask_dice = float(np.mean(dice_pairs))
        m.centroid_drift_mm = float(np.mean(drift_pairs))
    return m


def compute_image_metrics(
    img_3d: np.ndarray, spacing_z_mm: float
) -> tuple[float, float]:
    """Compute image-level inter-slice jitter.

    Returns (adjacent_slice_L1_normalised, z_axis_TV_per_mm).
    """
    img = img_3d.astype(np.float32)
    lo, hi = float(np.percentile(img, 1.0)), float(np.percentile(img, 99.0))
    if hi - lo < 1e-6:
        return float("nan"), float("nan")
    img_n = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    diffs = np.abs(img_n[1:] - img_n[:-1])
    adjacent_l1 = float(diffs.mean())
    z_tv_per_mm = float(diffs.sum(axis=(1, 2)).mean() / max(spacing_z_mm, 1e-3))
    return adjacent_l1, z_tv_per_mm


def process_subject(subject_dir: Path) -> Optional[SubjectMetrics]:
    """Analyse one subject dir. Returns None if no T2FS+organ masks available."""
    sid = subject_dir.name
    t2fs_path = subject_dir / f"{sid}_T2FS.nii.gz"
    if not t2fs_path.exists():
        return None

    img_sitk = sitk.ReadImage(str(t2fs_path), sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(img_sitk)  # (Z, Y, X)
    # SimpleITK spacing is (x, y, z); array is (z, y, x).
    sx, sy, sz = img_sitk.GetSpacing()

    subj = SubjectMetrics(
        subject=sid,
        n_slices=int(img_arr.shape[0]),
        voxel_spacing_xyz=(float(sx), float(sy), float(sz)),
    )
    subj.adjacent_slice_L1, subj.z_axis_TV_per_mm = compute_image_metrics(
        img_arr, spacing_z_mm=float(sz)
    )

    for organ, suffix in ORGAN_SUFFIXES.items():
        mask_path = subject_dir / f"{sid}{suffix}"
        if not mask_path.exists():
            continue
        mask_sitk = sitk.ReadImage(str(mask_path))
        mask_arr = sitk.GetArrayFromImage(mask_sitk)
        if mask_arr.shape != img_arr.shape:
            # Silent skip: known to happen for a few subjects with mask/image
            # resample mismatches; the calibration report should just exclude
            # them rather than crash.
            continue
        subj.organs[organ] = asdict(
            compute_organ_metrics(mask_arr, spacing_xy_mm=(sx, sy))
        )

    return subj


def aggregate_profile(subjects: list[SubjectMetrics]) -> dict:
    """Collapse per-subject metrics into per-organ percentile bands."""
    per_organ = {organ: {"n_subjects_with_mask": 0, "raw": {"dice": [], "drift_mm": []}} for organ in ORGAN_SUFFIXES}
    img_L1 = []
    img_TV = []
    for s in subjects:
        if s.adjacent_slice_L1 is not None and not np.isnan(s.adjacent_slice_L1):
            img_L1.append(s.adjacent_slice_L1)
        if s.z_axis_TV_per_mm is not None and not np.isnan(s.z_axis_TV_per_mm):
            img_TV.append(s.z_axis_TV_per_mm)
        for organ, om in s.organs.items():
            if om.get("adjacent_slice_mask_dice") is not None:
                per_organ[organ]["raw"]["dice"].append(om["adjacent_slice_mask_dice"])
                per_organ[organ]["raw"]["drift_mm"].append(om["centroid_drift_mm"])
                per_organ[organ]["n_subjects_with_mask"] += 1

    def pct(vals):
        if not vals:
            return None
        arr = np.array(vals, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    profile = {
        "n_subjects_total": len(subjects),
        "image": {
            "adjacent_slice_L1_normalised": pct(img_L1),
            "z_axis_TV_per_mm": pct(img_TV),
        },
        "per_organ": {},
    }
    for organ in ORGAN_SUFFIXES:
        raw = per_organ[organ]["raw"]
        profile["per_organ"][organ] = {
            "n_subjects_with_mask": per_organ[organ]["n_subjects_with_mask"],
            "adjacent_slice_mask_dice": pct(raw["dice"]),
            "centroid_drift_mm": pct(raw["drift_mm"]),
        }
    return profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("UT-EndoMRI/D2_TCPW"),
        help="Root dir containing D2-XXX subject dirs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("metrics/real_iscs_profile.json"),
        help="Output JSON path.",
    )
    p.add_argument(
        "--per-subject-out",
        type=Path,
        default=None,
        help="Optional path to also dump the full per-subject records.",
    )
    args = p.parse_args()

    subjects = sorted(d for d in args.data_dir.iterdir() if d.is_dir() and d.name.startswith("D2-"))
    print(f"[iscs] scanning {len(subjects)} subject dirs under {args.data_dir}")

    per_subject = []
    for i, sd in enumerate(subjects):
        try:
            r = process_subject(sd)
        except Exception as e:
            print(f"[iscs] {sd.name}: ERROR {e!r}")
            continue
        if r is None:
            continue
        per_subject.append(r)
        if (i + 1) % 20 == 0:
            print(f"[iscs] processed {i + 1}/{len(subjects)}")

    print(f"[iscs] {len(per_subject)} subjects with T2FS analysed")
    profile = aggregate_profile(per_subject)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(profile, f, indent=2)
    print(f"[iscs] wrote {args.out}")

    if args.per_subject_out is not None:
        args.per_subject_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_subject_out.open("w") as f:
            json.dump([asdict(s) for s in per_subject], f, indent=2)
        print(f"[iscs] wrote {args.per_subject_out}")

    # Terse stdout summary
    print("\n=== real D2 inter-slice consistency profile ===")
    for organ, block in profile["per_organ"].items():
        d = block["adjacent_slice_mask_dice"]
        r = block["centroid_drift_mm"]
        if d is None:
            print(f"  {organ:<14} n=0 — no masks found")
            continue
        print(
            f"  {organ:<14} n={block['n_subjects_with_mask']:>3}  "
            f"dice(p5/p50/p95)={d['p05']:.3f}/{d['p50']:.3f}/{d['p95']:.3f}  "
            f"drift_mm(p5/p50/p95)={r['p05']:.2f}/{r['p50']:.2f}/{r['p95']:.2f}"
        )
    img_L1 = profile["image"]["adjacent_slice_L1_normalised"]
    print(
        f"  {'IMAGE_L1':<14} n={img_L1['n']:>3}  "
        f"L1(p5/p50/p95)={img_L1['p05']:.4f}/{img_L1['p50']:.4f}/{img_L1['p95']:.4f}"
    )


if __name__ == "__main__":
    main()
