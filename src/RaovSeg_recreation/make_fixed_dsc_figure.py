#!/usr/bin/env python3
"""
Generate the post-fix DSC forest plot from actual metrics_ov.json files.

Replaces the hardcoded numbers in make_result_figures.py::fig_dsc_forest.
Reads DSC per (variant, seed) from the RAovSeg augmentation runs and
plots mean ± std vs the baseline references.

Usage:
    python -m src.RaovSeg_recreation.make_fixed_dsc_figure \
        --dsc-root /mnt/parscratch/users/$USER/synth_mri/runs \
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \
        --n-seeds 3 \
        --out-dir figures_fixed/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

BASELINE_REAL_ONLY = 0.290
BASELINE_PHASE1_V3 = 0.178

# Pre-fix DSC (from Chapter 4) for the comparison arrows
PRE_FIX = {
    "exp1c_concat":  0.053,
    "exp1c_spade":   0.178,
    "exp2":          0.020,
    "exp2_lam05":    0.020,
    "exp2_lam50":    0.020,
}


def load_dsc(dsc_root: Path, variants, n_seeds):
    out = {}
    for v in variants:
        seeds = []
        for s in range(n_seeds):
            p = dsc_root / f"raov_aug_{v}_fixed_seed{s}" / "metrics_ov.json"
            if p.exists():
                d = json.loads(p.read_text())
                seeds.append(d["aggregate"]["full"]["dsc"]["mean"])
        if seeds:
            out[v] = {
                "mean": float(np.mean(seeds)),
                "std":  float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0,
                "n":    len(seeds),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsc-root", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dsc = load_dsc(args.dsc_root, args.variants, args.n_seeds)
    print(f"Loaded DSC for {len(dsc)} variants")

    # Order variants for display (Phase 1 first, then Phase 2)
    order = ["exp1c_concat", "exp1c_spade", "exp2", "exp2_lam05", "exp2_lam50"]
    order = [v for v in order if v in dsc]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    means = [dsc[v]["mean"] for v in order]
    stds  = [dsc[v]["std"]  for v in order]

    # Post-fix DSC bars
    ax.errorbar(means, y, xerr=stds, fmt="o", color="#2ecc71", capsize=6,
                ms=10, elinewidth=2, alpha=0.9, label="Post-fix (n=3 seeds)")

    # Pre-fix DSC dots for comparison (with arrows showing the change)
    for i, v in enumerate(order):
        if v in PRE_FIX:
            ax.plot(PRE_FIX[v], y[i], "x", color="#888", ms=10, mew=2)
            if abs(means[i] - PRE_FIX[v]) > 0.005:
                ax.annotate("", xy=(means[i], y[i] + 0.15),
                            xytext=(PRE_FIX[v], y[i] + 0.15),
                            arrowprops={"arrowstyle": "->", "color": "#888", "alpha": 0.6})

    # Baseline reference lines
    ax.axvline(BASELINE_REAL_ONLY, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Real-only baseline = {BASELINE_REAL_ONLY:.3f}")
    ax.axvline(BASELINE_PHASE1_V3, color="orange", ls=":", lw=1.5, alpha=0.7,
               label=f"Phase 1 v3 SPADE (pre-fix ceiling) = {BASELINE_PHASE1_V3:.3f}")

    ax.set_yticks(y)
    ax.set_yticklabels([v + "_fixed" for v in order])
    ax.set_xlabel("Ovary DSC (mean ± std across seeds)")
    ax.set_title(f"Post-fix RAovSeg augmentation DSC ({sum(dsc[v]['n'] for v in order)} runs)")
    ax.grid(True, alpha=0.3, axis="x")

    # Combined legend: post-fix + pre-fix markers + baselines
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#2ecc71", ms=10, lw=0, label="Post-fix DSC"),
        Line2D([0], [0], marker="x", color="#888", ms=10, mew=2, lw=0, label="Pre-fix DSC"),
        Line2D([0], [0], color="red", ls="--", label=f"Real-only baseline (0.290)"),
        Line2D([0], [0], color="orange", ls=":", label=f"Phase 1 v3 ceiling (0.178)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 0.35)

    out = args.out_dir / "fig_dsc_forest_fixed.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

    # Also print the numbers so they're easy to copy into the writeup
    print("\n=== DSC summary ===")
    print(f"{'variant':<18s}  {'post-fix':<18s}  {'pre-fix':<10s}  {'Δ'}")
    for v in order:
        pre = PRE_FIX.get(v, float("nan"))
        m, s, n = dsc[v]["mean"], dsc[v]["std"], dsc[v]["n"]
        print(f"{v:<18s}  {m:.3f} ± {s:.3f} (n={n})  {pre:.3f}     {m - pre:+.3f}")


if __name__ == "__main__":
    main()
