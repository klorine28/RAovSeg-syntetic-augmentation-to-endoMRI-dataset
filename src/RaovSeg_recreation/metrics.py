"""
Segmentation metric bundle for RAovSeg evaluation.

All metrics operate on 3D binary masks (Z, Y, X). HD95 is reported in
MILLIMETRES using RAovSeg's fixed post-resample spacing
(z=6.0 mm, y=0.35 mm, x=0.35 mm). Reporting in voxel units would be
misleading because the grid is highly anisotropic (17× ratio between z
and in-plane spacing).

Returned by `compute_metric_bundle`:
    dsc              — Dice similarity coefficient
    iou              — intersection over union (Jaccard)
    sensitivity      — TP / (TP + FN) (recall)
    precision        — TP / (TP + FP)
    hd95_mm          — 95th percentile symmetric Hausdorff distance, mm
    volume_error     — (V_pred - V_gt) / V_gt; NaN if V_gt == 0
    volume_pred      — |pred| in voxels
    volume_gt        — |gt| in voxels

Aggregation utilities:
    bootstrap_ci     — 95% CI on the mean via 1000-sample bootstrap
    summary_row      — mean, std, 95% CI as a dict for one metric across subjects
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


METRIC_KEYS = (
    "dsc",
    "iou",
    "sensitivity",
    "precision",
    "hd95_mm",
    "volume_error",
    "volume_pred",
    "volume_gt",
)

# RAovSeg's fixed post-resample spacing in mm. Order matches the array
# axis order (Z, Y, X) used throughout this module. Keep in sync with
# OUT_SPACING in preprocess.py — that tuple is (x, y, z) SimpleITK style.
RAOVSEG_SPACING_ZYX_MM = (6.0, 0.35, 0.35)


def _safe_ratio(num: float, den: float) -> float:
    """0/0 → NaN; anything else = num/den."""
    if den <= 0:
        return float("nan")
    return float(num) / float(den)


def _boundary_voxels(mask: np.ndarray) -> np.ndarray:
    """Return the boundary voxels of a 3D binary mask as an (N, 3) array."""
    m = mask.astype(bool)
    if not m.any():
        return np.empty((0, 3), dtype=np.int64)
    eroded = binary_erosion(m, iterations=1)
    surface = m & (~eroded)
    return np.argwhere(surface)


def hausdorff_95_mm(pred: np.ndarray, gt: np.ndarray,
                    spacing_zyx_mm: tuple = RAOVSEG_SPACING_ZYX_MM) -> float:
    """Symmetric 95th-percentile Hausdorff distance in millimetres.

    Uses anisotropic spacing so that a 1-voxel shift in z (6 mm) counts
    the correct amount vs. a 1-voxel shift in x/y (0.35 mm).

    Convention when one mask is empty:
        both empty  -> 0.0 (nothing to reconcile)
        one empty   -> NaN (undefined — signals "hallucinated" or "missed")
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("nan")

    # EDT with `sampling` yields distances in mm.
    dt_pred = distance_transform_edt(~pred_b, sampling=spacing_zyx_mm)
    dt_gt = distance_transform_edt(~gt_b, sampling=spacing_zyx_mm)

    pred_surf = _boundary_voxels(pred_b)
    gt_surf = _boundary_voxels(gt_b)
    d_pred_to_gt = dt_gt[pred_surf[:, 0], pred_surf[:, 1], pred_surf[:, 2]]
    d_gt_to_pred = dt_pred[gt_surf[:, 0], gt_surf[:, 1], gt_surf[:, 2]]

    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


def compute_metric_bundle(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all metrics for one subject. pred and gt are 3D binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())

    v_pred = int(pred_b.sum())
    v_gt = int(gt_b.sum())

    dsc = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    iou = _safe_ratio(tp, tp + fp + fn)
    sensitivity = _safe_ratio(tp, tp + fn)
    precision = _safe_ratio(tp, tp + fp)
    hd95 = hausdorff_95_mm(pred_b, gt_b)
    vol_err = _safe_ratio(v_pred - v_gt, v_gt) if v_gt > 0 else float("nan")

    return {
        "dsc": dsc,
        "iou": iou,
        "sensitivity": sensitivity,
        "precision": precision,
        "hd95_mm": hd95,
        "volume_error": vol_err,
        "volume_pred": v_pred,
        "volume_gt": v_gt,
    }


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Non-parametric bootstrap CI on the mean. NaNs are dropped.

    Returns (lower, upper) at the (alpha/2, 1 - alpha/2) percentiles. When
    fewer than 2 non-NaN values remain, returns (NaN, NaN).
    """
    arr = np.array([v for v in values if not (v is None or np.isnan(v))], dtype=np.float64)
    if arr.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def summary_row(values: list[float], name: Optional[str] = None) -> dict:
    """Mean, std, 95% CI, n for one metric across subjects (NaN-safe)."""
    arr = np.array([v for v in values if not (v is None or np.isnan(v))], dtype=np.float64)
    ci_lo, ci_hi = bootstrap_ci(values)
    return {
        "name": name,
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size > 0 else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
    }
