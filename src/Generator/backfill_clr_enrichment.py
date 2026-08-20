#!/usr/bin/env python3
"""
Backfill CLR enrichment on existing per-sample metrics JSONs — no re-run of
explain.py required.

Rationale:
    Old per-sample JSONs (sample_XX_metrics.json) already contain CLR_per_channel
    but do not contain the label masks used at explain time, so we cannot
    compute the per-sample area fraction exactly. Instead we compute a
    COHORT-LEVEL mean area fraction per organ across the D2/D1 preprocessed
    training labels, and divide each per-sample CLR by that. The result
    (CLR_enrichment_per_channel) is added to each JSON in-place.

    The cohort-mean area is a reasonable null-baseline approximation because
    the enrichment claim is order-of-magnitude (3× concat vs 42× SPADE);
    per-sample deviations from the cohort mean are small enough not to
    change the qualitative conclusion.

    For strict per-sample enrichment, re-run explain.py — the fresh metrics
    JSONs will contain CLR_enrichment_per_channel computed exactly.

Usage:
    python -m src.Generator.backfill_clr_enrichment \\
        --preprocessed-root data/processed_generator/D2 \\
        --split-file data/splits/d2_generator_split.json \\
        --split train \\
        --sequence T2FS \\
        --explain-dirs \\
            /path/to/1a/current/explain \\
            /path/to/1b/current/explain \\
            /path/to/1c/concat/explain \\
            /path/to/1c/spade/explain \\
        --num-label-channels 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Channel index → semantic name (matches explain.py ORGAN_NAMES)
CHANNEL_TO_NAME = {
    0: "outside_body",
    1: "uterus",
    2: "ov_L",
    3: "ov_R",
    4: "em",
    5: "body_other",
}


def _load_label_nifti(path: Path, num_channels: int) -> np.ndarray:
    """Return label as (C, Z, H, W) uint8."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if arr.ndim == 3:
        return np.stack([(arr == c).astype(np.uint8) for c in range(num_channels)], axis=0)
    if arr.ndim == 4:
        if arr.shape[-1] == num_channels:
            return np.transpose(arr, (3, 0, 1, 2)).astype(np.uint8)
        if arr.shape[0] == num_channels:
            return arr.astype(np.uint8)
    raise ValueError(f"unexpected label shape {arr.shape} at {path}")


def cohort_mean_area_fractions(preprocessed_root: Path,
                               subjects: list[str],
                               sequence: str,
                               num_channels: int) -> dict[str, float]:
    """For each channel c, mean over all subjects × slices of |mask_c| / |image|.

    We compute per-slice, then average across slices and subjects (equal weight
    per slice — which matches how explain.py picks representative slices).
    Slices with zero mask contribute 0 to the mean (they are still valid
    denominators for the "typical area fraction seen at inference time").
    """
    per_channel_fractions: dict[str, list[float]] = {name: [] for name in CHANNEL_TO_NAME.values()}
    n_slices_total = 0
    for subj in subjects:
        lbl_path = preprocessed_root / subj / f"label_{sequence}.nii.gz"
        if not lbl_path.exists():
            print(f"  skip {subj}: {lbl_path} not found")
            continue
        try:
            arr = _load_label_nifti(lbl_path, num_channels)  # (C, Z, H, W)
        except Exception as e:
            print(f"  skip {subj}: {type(e).__name__}: {e}")
            continue
        _, Z, H, W = arr.shape
        per_slice_area = float(H * W)
        for z in range(Z):
            for ch in range(num_channels):
                name = CHANNEL_TO_NAME.get(ch, f"ch{ch}")
                fr = float(arr[ch, z].sum()) / per_slice_area
                per_channel_fractions[name].append(fr)
            n_slices_total += 1

    mean_fracs = {name: (float(np.mean(vals)) if vals else float("nan"))
                  for name, vals in per_channel_fractions.items()}
    print(f"\n[cohort] n_slices contributing = {n_slices_total}")
    for name, mf in mean_fracs.items():
        print(f"  area_frac[{name:12s}] = {mf:.6f}")
    return mean_fracs


def backfill_dir(explain_dir: Path,
                 mean_area_fracs: dict[str, float],
                 dry_run: bool = False) -> int:
    """Amend every sample_XX_metrics.json in a directory with CLR_area_frac and
    CLR_enrichment fields derived from the cohort-mean area fractions."""
    n_amended = 0
    for jp in sorted(explain_dir.glob("sample_*_metrics.json")):
        try:
            with jp.open() as f:
                # Handle NaN in the JSON (explain.py writes NaN literals).
                text = f.read().replace("NaN", "null")
                d = json.loads(text)
        except Exception as e:
            print(f"  skip {jp.name}: {type(e).__name__}: {e}")
            continue
        clr = d.get("CLR_per_channel")
        if not clr:
            continue

        area_frac = {name: mean_area_fracs.get(name, float("nan")) for name in clr}
        enrichment: dict[str, float] = {}
        for name, val in clr.items():
            af = area_frac.get(name, float("nan"))
            if val is None or af is None or af <= 0:
                enrichment[name] = float("nan")
            else:
                try:
                    enrichment[name] = float(val) / float(af)
                except (TypeError, ValueError, ZeroDivisionError):
                    enrichment[name] = float("nan")

        d.setdefault("CLR_area_frac_per_channel", area_frac)
        d["CLR_enrichment_per_channel"] = enrichment
        d.setdefault("_clr_enrichment_backfill_note",
                     "cohort-mean area fractions used as null baseline; see "
                     "src/Generator/backfill_clr_enrichment.py for method.")

        if dry_run:
            print(f"  would amend {jp.name}: enrichment = "
                  + " ".join(f"{k}={v:.2f}" for k, v in enrichment.items() if not np.isnan(v)))
        else:
            # Round-trip through json to ensure NaN → null on write too.
            def _clean(o):
                if isinstance(o, float) and np.isnan(o):
                    return None
                if isinstance(o, dict):
                    return {k: _clean(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_clean(v) for v in o]
                return o
            with jp.open("w") as f:
                json.dump(_clean(d), f, indent=2)
        n_amended += 1
    print(f"  {explain_dir}: amended {n_amended} JSON(s)")
    return n_amended


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--preprocessed-root", type=Path, required=True,
                    help="e.g. data/processed_generator/D2")
    ap.add_argument("--split-file", type=Path, required=True,
                    help="e.g. data/splits/d2_generator_split.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sequence", default="T2FS",
                    help="Suffix of label file: label_<sequence>.nii.gz")
    ap.add_argument("--num-label-channels", type=int, default=6)
    ap.add_argument("--explain-dirs", nargs="+", required=True, type=Path,
                    help="One or more directories containing sample_*_metrics.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change, don't write back to files")
    args = ap.parse_args()

    subjects = json.load(args.split_file.open())[args.split]
    print(f"[cohort] loaded {len(subjects)} subjects from split '{args.split}'")
    mean_fracs = cohort_mean_area_fractions(
        args.preprocessed_root, subjects, args.sequence, args.num_label_channels
    )

    total = 0
    for d in args.explain_dirs:
        if not d.exists():
            print(f"  missing dir: {d}"); continue
        total += backfill_dir(d, mean_fracs, dry_run=args.dry_run)
    print(f"\n[done] amended {total} per-sample JSON(s) across "
          f"{len(args.explain_dirs)} explain dir(s)")


if __name__ == "__main__":
    main()
