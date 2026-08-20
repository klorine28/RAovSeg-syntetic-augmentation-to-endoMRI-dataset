#!/usr/bin/env python3
"""
Extended correlation analysis — 5 image-domain metrics × 4 downstream metrics.

Builds a 5×4 correlation matrix (Pearson r, Spearman ρ) between:
    Image-domain (rows):
        FID, LPIPS_mean, hist_KL, ovary_mean, in_window_pct
    Downstream (cols):
        DSC, HD95_mm, sensitivity, volume_error

Also renders a heatmap of Pearson r values so you can see the block
structure at a glance (task-specific + intensity-domain metrics
correlate one way with downstream; feature-domain metrics correlate
the opposite way).

Usage (from repo root, after aggregate_dsc_metrics.py has run):
    python -m src.analysis.extended_correlation \\
        --dsc-agg-dir metrics/fixed/agg \\
        --quality-dir metrics/fixed \\
        --mech-csv hpc_pulled/fixed_analysis/figures_fixed/mechanism/mech_ovary_intensity_table.csv \\
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \\
        --out-dir figures_fixed/correlation_extended
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


IMAGE_METRICS = ["fid", "lpips_mean", "hist_kl", "ovary_mean", "in_window_pct"]
# Direction hint: "up_is_better" means larger value = better realism / better utility
# ("−" if lower is better)
IMAGE_METRIC_HINT = {
    "fid": "−",           # lower FID = better realism
    "lpips_mean": "−",    # lower LPIPS = closer perceptually
    "hist_kl": "−",       # lower KL = closer histograms
    "ovary_mean": "→win", # closer to enhancement-window centre
    "in_window_pct": "+", # higher % = more voxels in window
}
DOWNSTREAM_METRICS = ["dsc", "hd95_mm", "sensitivity", "volume_error"]
DOWNSTREAM_HINT = {
    "dsc": "+",             # higher = better
    "hd95_mm": "−",         # lower = better (closer boundaries)
    "sensitivity": "+",     # higher = better (found the ovary)
    "volume_error": "−",    # lower = better (accurate volume)
}


def _load_image_metrics(quality_dir: Path, mech_csv: Path, variants: list[str]) -> pd.DataFrame:
    """One row per variant with the five image-domain metrics."""
    mech = pd.read_csv(mech_csv)
    mech_lookup = {}
    for v in variants:
        row = mech[mech["variant"] == f"{v.replace('exp1c_', '').replace('_fixed', '')}_fixed (pooled)"]
        if row.empty:
            # try direct
            row = mech[mech["variant"].str.startswith(v)]
        if row.empty:
            print(f"  WARN: mech table has no row matching {v}")
            mech_lookup[v] = {"ovary_mean": np.nan, "in_window_pct": np.nan}
        else:
            r = row.iloc[0]
            mech_lookup[v] = {
                "ovary_mean": float(r["mean"]),
                "in_window_pct": float(r["pct_in_window"]),
            }

    rows = []
    for v in variants:
        qp = quality_dir / f"quality_{v}_fixed.json"
        if not qp.exists():
            print(f"  WARN: missing quality JSON for {v}")
            rows.append({"variant": v, "fid": np.nan, "lpips_mean": np.nan, "hist_kl": np.nan,
                         **mech_lookup[v]})
            continue
        q = json.load(qp.open())
        rows.append({
            "variant": v,
            "fid": q.get("fid", np.nan),
            "lpips_mean": q.get("lpips_nn", {}).get("mean", np.nan),
            "hist_kl": q.get("hist_kl", np.nan),
            **mech_lookup[v],
        })
    return pd.DataFrame(rows).set_index("variant")


def _load_downstream(dsc_agg_dir: Path, variants: list[str]) -> pd.DataFrame:
    rows = []
    for v in variants:
        p = dsc_agg_dir / f"{v}_agg.json"
        if not p.exists():
            print(f"  WARN: missing DSC agg JSON for {v}")
            continue
        d = json.load(p.open())
        row = {"variant": v}
        for m in DOWNSTREAM_METRICS:
            row[m] = d.get(m, {}).get("mean", np.nan)
            row[f"{m}_std"] = d.get(m, {}).get("std", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def _correlate(image_df: pd.DataFrame, down_df: pd.DataFrame) -> pd.DataFrame:
    joined = image_df.join(down_df, how="inner")
    print(f"\n[joined] {len(joined)} variants:\n{joined.round(3)}\n")

    out_rows = []
    for im in IMAGE_METRICS:
        for dm in DOWNSTREAM_METRICS:
            x = joined[im].values
            y = joined[dm].values
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 3:
                out_rows.append({"image_metric": im, "downstream": dm, "n": int(m.sum()),
                                 "pearson_r": np.nan, "pearson_p": np.nan,
                                 "spearman_rho": np.nan, "spearman_p": np.nan})
                continue
            r, pr = stats.pearsonr(x[m], y[m])
            rho, ps = stats.spearmanr(x[m], y[m])
            out_rows.append({"image_metric": im, "downstream": dm, "n": int(m.sum()),
                             "pearson_r": r, "pearson_p": pr,
                             "spearman_rho": rho, "spearman_p": ps})
    return pd.DataFrame(out_rows)


def _heatmap(corr: pd.DataFrame, out_path: Path):
    """5×4 heatmap of Pearson r. Colour convention:
       red = higher image-metric → higher downstream metric.
       blue = higher image-metric → lower downstream metric."""
    pivot = corr.pivot(index="image_metric", columns="downstream", values="pearson_r")
    pivot = pivot.reindex(index=IMAGE_METRICS, columns=DOWNSTREAM_METRICS)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(DOWNSTREAM_METRICS)))
    ax.set_xticklabels([f"{d}\n({DOWNSTREAM_HINT[d]})" for d in DOWNSTREAM_METRICS], rotation=0)
    ax.set_yticks(range(len(IMAGE_METRICS)))
    ax.set_yticklabels([f"{im_m} ({IMAGE_METRIC_HINT[im_m]})" for im_m in IMAGE_METRICS])
    for i in range(len(IMAGE_METRICS)):
        for j in range(len(DOWNSTREAM_METRICS)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Image-domain metric × downstream metric — Pearson r\n"
                 "(post-fix, n=5 variants; +red = worse image = better downstream)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsc-agg-dir", type=Path, required=True)
    ap.add_argument("--quality-dir", type=Path, required=True)
    ap.add_argument("--mech-csv", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_df = _load_image_metrics(args.quality_dir, args.mech_csv, args.variants)
    down_df  = _load_downstream(args.dsc_agg_dir, args.variants)

    corr = _correlate(image_df, down_df)

    # Save the joined variant table
    joined = image_df.join(down_df, how="inner")
    joined.to_csv(args.out_dir / "variant_metrics_joined.csv")

    # Save the long-format correlation table
    corr.to_csv(args.out_dir / "correlation_matrix.csv", index=False)

    # Save a pretty-printed Pearson r pivot
    pivot = corr.pivot(index="image_metric", columns="downstream", values="pearson_r")
    pivot = pivot.reindex(index=IMAGE_METRICS, columns=DOWNSTREAM_METRICS).round(3)
    pivot.to_csv(args.out_dir / "pearson_r_pivot.csv")

    print("\n[Pearson r]")
    print(pivot.to_string())

    _heatmap(corr, args.out_dir / "heatmap_pearson_r.png")

    print(f"\n[done] wrote outputs to {args.out_dir}/")


if __name__ == "__main__":
    main()
