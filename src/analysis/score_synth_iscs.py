#!/usr/bin/env python3
"""
Score synthetic volumes against the real D2 inter-slice consistency profile.

Reuses `process_subject` from the calibration module, which computes the same
four metrics (adjacent-slice mask DSC, centroid drift, image L1, z-axis TV) as
it did on real D2. Each synth subject's metric is compared against the real
cohort's 5th and 95th percentile band. A synth value is "in-band" if it falls
inside that band — meaning it is neither too jittery (below p05 for DSC / above
p95 for drift) nor too consistent (above p95 for DSC / below p05 for drift).

Composite ISCS-score = fraction of subject × metric evaluations that landed
in-band, per organ and overall.

Usage:
    python -m src.analysis.score_synth_iscs \\
        --synth-dir synth_volumes/exp1c_spade \\
        --profile metrics/real_iscs_profile.json \\
        --out metrics/exp1c_spade_iscs_score.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .inter_slice_consistency import ORGAN_SUFFIXES, process_subject


def load_profile(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def score_value(value: Optional[float], band: Optional[dict]) -> dict:
    """Score a single metric value against the real profile band.

    Returns dict with: value, in_band (bool), z_score, direction ("low"/"high"/"in").
    """
    if value is None or band is None:
        return {"value": value, "in_band": None, "z_score": None, "direction": None}
    p05, p95 = band["p05"], band["p95"]
    mean, std = band["mean"], max(band["std"], 1e-6)
    if value < p05:
        direction = "low"
        in_band = False
    elif value > p95:
        direction = "high"
        in_band = False
    else:
        direction = "in"
        in_band = True
    z = (value - mean) / std
    return {
        "value": float(value),
        "in_band": bool(in_band),
        "z_score": float(z),
        "direction": direction,
    }


def score_subject(subj, profile: dict) -> dict:
    """Score all metrics for one subject against the real profile."""
    result = {
        "subject": subj.subject,
        "n_slices": subj.n_slices,
        "voxel_spacing_xyz": subj.voxel_spacing_xyz,
        "image": {},
        "per_organ": {},
    }

    img_bands = profile.get("image", {})
    result["image"]["adjacent_slice_L1_normalised"] = score_value(
        subj.adjacent_slice_L1, img_bands.get("adjacent_slice_L1_normalised")
    )
    result["image"]["z_axis_TV_per_mm"] = score_value(
        subj.z_axis_TV_per_mm, img_bands.get("z_axis_TV_per_mm")
    )

    organ_bands = profile.get("per_organ", {})
    for organ in ORGAN_SUFFIXES:
        om = subj.organs.get(organ)
        band = organ_bands.get(organ, {})
        if not om:
            result["per_organ"][organ] = {"present": False}
            continue
        result["per_organ"][organ] = {
            "present": True,
            "n_interior_pairs": om["n_interior_pairs"],
            "adjacent_slice_mask_dice": score_value(
                om["adjacent_slice_mask_dice"], band.get("adjacent_slice_mask_dice")
            ),
            "centroid_drift_mm": score_value(
                om["centroid_drift_mm"], band.get("centroid_drift_mm")
            ),
        }
    return result


def aggregate_scores(per_subject: list[dict]) -> dict:
    """Compute per-organ and overall in-band fraction (ISCS-score)."""
    metric_columns = {
        "image": ["adjacent_slice_L1_normalised", "z_axis_TV_per_mm"],
        "per_organ": ["adjacent_slice_mask_dice", "centroid_drift_mm"],
    }

    aggregate = {"n_subjects": len(per_subject), "image": {}, "per_organ": {}}

    for m in metric_columns["image"]:
        hits = [s["image"][m]["in_band"] for s in per_subject if s["image"][m]["in_band"] is not None]
        zs = [s["image"][m]["z_score"] for s in per_subject if s["image"][m]["z_score"] is not None]
        aggregate["image"][m] = {
            "n": len(hits),
            "in_band_frac": float(np.mean(hits)) if hits else None,
            "z_score_mean": float(np.mean(zs)) if zs else None,
            "z_score_p50": float(np.percentile(zs, 50)) if zs else None,
        }

    for organ in ORGAN_SUFFIXES:
        aggregate["per_organ"][organ] = {}
        for m in metric_columns["per_organ"]:
            hits = []
            zs = []
            for s in per_subject:
                block = s["per_organ"].get(organ, {})
                if not block.get("present"):
                    continue
                b = block.get(m, {})
                if b.get("in_band") is not None:
                    hits.append(b["in_band"])
                if b.get("z_score") is not None:
                    zs.append(b["z_score"])
            aggregate["per_organ"][organ][m] = {
                "n": len(hits),
                "in_band_frac": float(np.mean(hits)) if hits else None,
                "z_score_mean": float(np.mean(zs)) if zs else None,
                "z_score_p50": float(np.percentile(zs, 50)) if zs else None,
            }

    # Composite ISCS-score: mean in-band fraction across all metrics that had
    # any evaluations. Weighted uniformly across image + per-organ metrics.
    all_fracs = []
    for m_block in list(aggregate["image"].values()) + [
        m for organ_block in aggregate["per_organ"].values() for m in organ_block.values()
    ]:
        f = m_block.get("in_band_frac")
        if f is not None:
            all_fracs.append(f)
    aggregate["iscs_score_overall"] = float(np.mean(all_fracs)) if all_fracs else None

    return aggregate


def per_subject_verdict(scored: dict) -> str:
    """One-line diagnostic per subject listing out-of-band metrics."""
    bad = []
    for m, r in scored["image"].items():
        if r.get("in_band") is False:
            bad.append(f"image.{m}:{r['direction']}(z={r['z_score']:.2f})")
    for organ, block in scored["per_organ"].items():
        if not block.get("present"):
            continue
        for m in ("adjacent_slice_mask_dice", "centroid_drift_mm"):
            r = block.get(m, {})
            if r.get("in_band") is False:
                bad.append(f"{organ}.{m.split('_')[0]}:{r['direction']}(z={r['z_score']:.2f})")
    if not bad:
        return "  OK"
    return "  OOB: " + ", ".join(bad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth-dir", type=Path, required=True,
                   help="Root dir of synth subjects (D2-9XX/D2-9XX_T2FS.nii.gz layout).")
    p.add_argument("--profile", type=Path, default=Path("metrics/real_iscs_profile.json"),
                   help="Real-D2 reference profile JSON.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path for the score summary.")
    p.add_argument("--per-subject-out", type=Path, default=None,
                   help="Optional path to also dump the full per-subject scored records.")
    p.add_argument("--label", type=str, default=None,
                   help="Optional label saved into the output for cross-run comparison.")
    args = p.parse_args()

    profile = load_profile(args.profile)
    subject_dirs = sorted(d for d in args.synth_dir.iterdir() if d.is_dir())
    print(f"[iscs-score] {len(subject_dirs)} subject dirs under {args.synth_dir}")
    print(f"[iscs-score] reference profile: {args.profile} "
          f"(n_subjects_total={profile.get('n_subjects_total')})")

    per_subject_scored = []
    for sd in subject_dirs:
        try:
            subj = process_subject(sd)
        except Exception as e:
            print(f"  {sd.name}: ERROR {e!r}")
            continue
        if subj is None:
            continue
        scored = score_subject(subj, profile)
        per_subject_scored.append(scored)
        print(f"[{sd.name}] " + per_subject_verdict(scored)[2:])

    print(f"\n[iscs-score] {len(per_subject_scored)} subjects scored")
    aggregate = aggregate_scores(per_subject_scored)

    output = {
        "label": args.label,
        "synth_dir": str(args.synth_dir),
        "profile_source": str(args.profile),
        "aggregate": aggregate,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"[iscs-score] wrote {args.out}")

    if args.per_subject_out is not None:
        args.per_subject_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_subject_out.open("w") as f:
            json.dump(per_subject_scored, f, indent=2)
        print(f"[iscs-score] wrote {args.per_subject_out}")

    # Terse stdout summary
    print("\n=== synth ISCS-score summary ===")
    print(f"  Overall ISCS-score: {aggregate['iscs_score_overall']:.3f}   (1.0 = all metrics in real 5-95 percentile band)")
    for m, block in aggregate["image"].items():
        f = block["in_band_frac"]
        z = block["z_score_p50"]
        f_str = f"{f:.3f}" if f is not None else "n/a"
        z_str = f"{z:+.2f}" if z is not None else "n/a"
        print(f"  image.{m:<28} n={block['n']:>3}  in_band={f_str}  z_p50={z_str}")
    for organ, block in aggregate["per_organ"].items():
        for m, sub in block.items():
            f = sub["in_band_frac"]
            z = sub["z_score_p50"]
            if f is None:
                continue
            print(f"  {organ:<12}.{m.split('_')[0]:<8} n={sub['n']:>3}  "
                  f"in_band={f:.3f}  z_p50={z:+.2f}")


if __name__ == "__main__":
    main()
