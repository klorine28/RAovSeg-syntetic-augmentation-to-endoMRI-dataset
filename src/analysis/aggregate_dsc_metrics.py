#!/usr/bin/env python3
"""
Aggregate per-seed metrics_ov.json into per-variant summary.

For each variant, reads N seeds' worth of RAovSeg output JSONs and
computes mean ± std across seeds for every downstream metric
(DSC, IoU, sensitivity, precision, HD95_mm, volume_error).

Input layout (on HPC):
    <runs_root>/raov_aug_<variant>_fixed_seed<S>/metrics_ov.json

Output:
    <out_dir>/<variant>_agg.json — dict with per-metric mean/std

Usage:
    python -m src.analysis.aggregate_dsc_metrics \\
        --runs-root /mnt/parscratch/users/$USER/synth_mri/runs \\
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \\
        --seeds 0 1 2 \\
        --out-dir metrics/fixed/agg
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

METRICS = ["dsc", "iou", "sensitivity", "precision", "hd95_mm", "volume_error"]


def _load_seed(path: Path) -> dict | None:
    if not path.exists():
        print(f"  MISSING: {path}")
        return None
    return json.load(path.open())


def _aggregate_variant(runs_root: Path, variant: str, seeds: list[int]) -> dict:
    """Read each seed's metrics_ov.json, aggregate across seeds."""
    # For each seed, take the "full" (post-processed) aggregate
    per_seed = []
    per_seed_per_subject = []  # list of per-subject metric dicts
    for s in seeds:
        p = runs_root / f"raov_aug_{variant}_fixed_seed{s}" / "metrics_ov.json"
        d = _load_seed(p)
        if d is None:
            continue
        agg = d.get("aggregate", {}).get("full", {})
        per_seed.append({m: agg.get(m, {}).get("mean", np.nan) for m in METRICS})
        per_seed_per_subject.append(d.get("per_subject", {}).get("full", []))

    if not per_seed:
        return {"variant": variant, "n_seeds": 0, "status": "no data"}

    arr = {m: np.array([s[m] for s in per_seed], dtype=float) for m in METRICS}
    out = {
        "variant": variant,
        "n_seeds": len(per_seed),
        "seed_ids": [s for s, d in zip(seeds, per_seed) if True],
    }
    for m in METRICS:
        v = arr[m][~np.isnan(arr[m])]
        out[m] = {
            "mean": float(np.mean(v)) if v.size else float("nan"),
            "std":  float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "n":    int(v.size),
            "per_seed": arr[m].tolist(),
        }

    # Also aggregate the per-subject DSC vector across seeds (useful for
    # per-subject std across-seeds later).
    if per_seed_per_subject:
        subj_ids = sorted({r["subject"] for seed_rows in per_seed_per_subject for r in seed_rows})
        subj_dsc = {sid: [] for sid in subj_ids}
        subj_hd95 = {sid: [] for sid in subj_ids}
        for seed_rows in per_seed_per_subject:
            for r in seed_rows:
                subj_dsc[r["subject"]].append(r.get("dsc", np.nan))
                subj_hd95[r["subject"]].append(r.get("hd95_mm", np.nan))
        out["per_subject"] = {
            sid: {
                "dsc_mean":   float(np.nanmean(subj_dsc[sid])),
                "dsc_std":    float(np.nanstd(subj_dsc[sid], ddof=1)) if len(subj_dsc[sid]) > 1 else 0.0,
                "hd95_mean":  float(np.nanmean(subj_hd95[sid])),
                "hd95_std":   float(np.nanstd(subj_hd95[sid], ddof=1)) if len(subj_hd95[sid]) > 1 else 0.0,
            }
            for sid in subj_ids
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'variant':22} {'n_seed':>6} {'DSC':>13} {'HD95_mm':>13} "
          f"{'sens':>11} {'prec':>11} {'vol_err':>11}")
    for v in args.variants:
        agg = _aggregate_variant(args.runs_root, v, args.seeds)
        out_p = args.out_dir / f"{v}_agg.json"
        out_p.write_text(json.dumps(agg, indent=2))
        if agg.get("n_seeds", 0) == 0:
            print(f"{v:22} 0    no data")
            continue
        print(f"{v:22} {agg['n_seeds']:>6}  "
              f"{agg['dsc']['mean']:.3f}±{agg['dsc']['std']:.3f}   "
              f"{agg['hd95_mm']['mean']:6.1f}±{agg['hd95_mm']['std']:5.1f}   "
              f"{agg['sensitivity']['mean']:.3f}±{agg['sensitivity']['std']:.3f}   "
              f"{agg['precision']['mean']:.3f}±{agg['precision']['std']:.3f}   "
              f"{agg['volume_error']['mean']:5.2f}±{agg['volume_error']['std']:4.2f}")

    print(f"\n[done] wrote {len(args.variants)} aggregated JSONs to {args.out_dir}/")


if __name__ == "__main__":
    main()
