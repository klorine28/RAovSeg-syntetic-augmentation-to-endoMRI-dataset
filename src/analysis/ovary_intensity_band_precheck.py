#!/usr/bin/env python3
"""
Exp 2 pre-check — free (CPU-only) offline sweep of the RAovSeg enhancement
window [o1, o2] on real D2 training subjects.

Loads each subject through steps 1–3 of preprocess.py (resample, percentile
clip, minmax — stopping BEFORE the o1/o2 enhancement), splits body voxels
into ovary vs non-ovary using the ovary label, and for every candidate
(o1, o2) reports:

    ovary_capture     = P(o1 <= x <= o2 | ovary)
    non_ovary_capture = P(o1 <= x <= o2 | body \\ ovary)   ("noise" for the rule)
    precision         = ovary_hits / (ovary_hits + non_ovary_hits)   voxel-level
    ovary_median      = intensity of the median ovary voxel

The published band [0.22, 0.30] is included as a control row. This tells you
in minutes whether the Exp 2 GPU sweep is worth the seven hours — if the
published band already sits at the knee of the precision/recall curve, the
interesting version of Exp 2 evaporates.

Usage:
    python -m src.analysis.ovary_intensity_band_precheck \\
        --data-dir UT-EndoMRI/D2_TCPW \\
        --out-csv  figures/exp2_band_precheck.csv \\
        [--bands 0.22,0.30 0.15,0.25 0.30,0.42 0.42,0.56]

Also writes a companion histogram JSON so downstream figure scripts can
plot the ovary/non-ovary distributions on the same axes as the bands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Reuse the preprocessing pipeline's steps 1–3 (resample + percentile-clip +
# minmax) so the numbers we report line up 1:1 with what train_attuseg sees.
from src.RaovSeg_recreation.preprocess import (
    O1, O2, SUBJECT_RE,
    find_best_sequence, preprocess_image, preprocess_label,
    scan_and_classify,
)


DEFAULT_BANDS = [
    (0.22, 0.30),   # published control
    (0.15, 0.25),
    (0.30, 0.42),
    (0.42, 0.56),
]


def _parse_band(spec: str) -> tuple[float, float]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"band must be 'o1,o2', got {spec!r}")
    a, b = float(parts[0]), float(parts[1])
    if not (0.0 <= a < b <= 1.0):
        raise argparse.ArgumentTypeError(f"require 0 <= o1 < o2 <= 1, got {a},{b}")
    return (a, b)


def load_subject_pre_enhancement(subject_dir: Path, subject_id: str
                                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (image_after_step_3, ovary_mask). image is float32 in [0,1];
    mask is bool with the same shape. Returns None if the subject has no ov
    label or no MRI sequence.
    """
    img_path = find_best_sequence(subject_dir, subject_id)
    if img_path is None:
        return None
    ov_path = subject_dir / f"{subject_id}_ov.nii.gz"
    if not ov_path.exists():
        return None
    # skip_enhancement=True returns the volume after step 3 (percentile-clip
    # + minmax) — exactly the intensity range the enhancement rule operates on.
    img = preprocess_image(img_path, skip_enhancement=True).astype(np.float32)
    lbl = preprocess_label(ov_path)
    mask = lbl > 0
    if mask.shape != img.shape:
        raise ValueError(
            f"{subject_id}: image {img.shape} vs mask {mask.shape} mismatch")
    return img, mask


def pool_voxels(data_dir: Path, only_train_val: bool = True
                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Walk data_dir, preprocess each included subject, and return
    (ovary_voxels, non_ovary_body_voxels, subject_ids).

    Non-ovary "body" is everything with intensity > 0 after the pipeline's
    percentile-clip + minmax — a cheap body silhouette that matches how the
    enhancement rule fires anyway (background is at or near 0).
    """
    rows = scan_and_classify(data_dir)
    subj_ids: list[str] = []
    ovary_chunks: list[np.ndarray] = []
    body_chunks: list[np.ndarray] = []
    for r in rows:
        if not r["included"]:
            continue
        if only_train_val and r["split"] != "train_val":
            continue
        sid = r["subject_id"]
        loaded = load_subject_pre_enhancement(data_dir / sid, sid)
        if loaded is None:
            print(f"  SKIP {sid}: missing sequence or ov label")
            continue
        img, ov_mask = loaded
        body_mask = img > 0
        non_ov_body = body_mask & ~ov_mask
        n_ov = int(ov_mask.sum())
        n_body = int(non_ov_body.sum())
        print(f"  {sid}: {n_ov:>7d} ovary vox, {n_body:>9d} non-ovary body vox")
        ovary_chunks.append(img[ov_mask].astype(np.float32))
        body_chunks.append(img[non_ov_body].astype(np.float32))
        subj_ids.append(sid)
    if not subj_ids:
        raise RuntimeError(
            f"No eligible subjects found in {data_dir} "
            f"(only_train_val={only_train_val})")
    return np.concatenate(ovary_chunks), np.concatenate(body_chunks), subj_ids


def band_stats(ovary_vox: np.ndarray, body_vox: np.ndarray,
               o1: float, o2: float) -> dict:
    ov_hits = int(((ovary_vox >= o1) & (ovary_vox <= o2)).sum())
    body_hits = int(((body_vox >= o1) & (body_vox <= o2)).sum())
    ov_recall = ov_hits / max(ovary_vox.size, 1)
    body_capture = body_hits / max(body_vox.size, 1)
    denom = ov_hits + body_hits
    precision = ov_hits / denom if denom else float("nan")
    return {
        "o1": o1,
        "o2": o2,
        "ovary_hits": ov_hits,
        "non_ovary_body_hits": body_hits,
        "ovary_recall": ov_recall,
        "non_ovary_body_capture": body_capture,
        "voxel_precision": precision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2]
                                / "UT-EndoMRI" / "D2_TCPW")
    parser.add_argument("--out-csv", type=Path,
                        default=Path("figures/exp2_band_precheck.csv"))
    parser.add_argument("--out-hist-json", type=Path,
                        default=Path("figures/exp2_band_precheck_hist.json"),
                        help="Companion histogram JSON (ovary vs non-ovary "
                             "body distributions) for downstream plotting.")
    parser.add_argument("--bands", nargs="+", type=_parse_band, default=None,
                        help="Extra bands to sweep, format 'o1,o2'. The "
                             f"published control ({O1},{O2}) is always "
                             f"included. Default set: {DEFAULT_BANDS}")
    parser.add_argument("--include-test-subjects", action="store_true",
                        help="Also pool the 8 sacred test subjects. Default "
                             "is train_val only, matching what a real Exp 2 "
                             "run would see at training time.")
    parser.add_argument("--n-hist-bins", type=int, default=100)
    args = parser.parse_args()

    print(f"[precheck] data-dir={args.data_dir}")
    print(f"[precheck] scope: "
          + ("train_val + test" if args.include_test_subjects else "train_val only"))

    ovary_vox, body_vox, subj_ids = pool_voxels(
        args.data_dir, only_train_val=not args.include_test_subjects)
    print(f"[precheck] pooled {len(subj_ids)} subjects → "
          f"{ovary_vox.size:,} ovary voxels, {body_vox.size:,} non-ovary body voxels")

    ovary_pctiles = {int(p): float(np.percentile(ovary_vox, p))
                     for p in (1, 5, 25, 50, 75, 95, 99)}
    body_pctiles = {int(p): float(np.percentile(body_vox, p))
                    for p in (1, 5, 25, 50, 75, 95, 99)}
    print(f"[precheck] ovary percentiles: {ovary_pctiles}")
    print(f"[precheck] body percentiles:  {body_pctiles}")

    bands = list(DEFAULT_BANDS)
    if args.bands:
        for b in args.bands:
            if b not in bands:
                bands.append(b)

    rows = [band_stats(ovary_vox, body_vox, o1, o2) for (o1, o2) in bands]
    # Sort by ovary_recall (ascending) so the CSV reads left-to-right along the
    # precision/recall trade-off curve.
    rows.sort(key=lambda r: (r["ovary_recall"], r["o1"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[precheck] wrote {args.out_csv}")

    # Companion histogram — 100 bins on [0, 1] for both populations.
    edges = np.linspace(0.0, 1.0, args.n_hist_bins + 1)
    ov_hist, _ = np.histogram(ovary_vox, bins=edges, density=True)
    body_hist, _ = np.histogram(body_vox, bins=edges, density=True)
    args.out_hist_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_hist_json, "w") as f:
        json.dump({
            "bin_edges": edges.tolist(),
            "ovary_density": ov_hist.tolist(),
            "non_ovary_body_density": body_hist.tolist(),
            "n_ovary_voxels": int(ovary_vox.size),
            "n_non_ovary_body_voxels": int(body_vox.size),
            "ovary_percentiles": ovary_pctiles,
            "non_ovary_body_percentiles": body_pctiles,
            "subjects": subj_ids,
            "published_band": [O1, O2],
        }, f, indent=2)
    print(f"[precheck] wrote {args.out_hist_json}")

    # Human-readable summary to stdout — the interesting numbers.
    print("\n=== Band sweep ===")
    print(f"{'o1':>6} {'o2':>6} | {'ov_recall':>10} {'body_cap':>10} {'prec':>10}")
    for r in rows:
        marker = "  <-- published" if (r["o1"], r["o2"]) == (O1, O2) else ""
        print(f"{r['o1']:>6.3f} {r['o2']:>6.3f} | "
              f"{r['ovary_recall']:>10.4f} {r['non_ovary_body_capture']:>10.4f} "
              f"{r['voxel_precision']:>10.4f}{marker}")


if __name__ == "__main__":
    main()
