#!/usr/bin/env python3
"""
Two robustness analyses on top of the post-fix correlation table:

(A) COMBINED pre-fix + post-fix correlation.
    Adds the two matched Phase-1 pre-fix points (exp1c_concat_pre,
    exp1c_spade_pre) to the 5 post-fix variants → n=7. Tests whether
    the FID-vs-DSC negative correlation is a within-post-fix pattern
    or a real cross-transition effect. Pre-fix data comes from
    metrics/master_metrics.csv + pre-fix DSC values from §4.11.1.

(B) BOOTSTRAP CIs on the 5×4 Pearson r matrix.
    5,000 draws with replacement. Reports 95% CI for each r cell.
    Gives defensible statistical statements at n=5.

Outputs:
    - combined_correlation.csv (n=7 correlations for the 4 metrics
      with matched pre-fix data: FID, LPIPS, hist_KL vs DSC)
    - bootstrap_ci_matrix.csv (5×4 grid of r ± 95% CI)
    - bootstrap_summary_pivot.csv (r_lo/r/r_hi formatted per cell)

Usage:
    python -m src.analysis.combined_and_bootstrap
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO = Path(__file__).resolve().parents[2]

# Manually curated pre-fix data (from master_metrics.csv + pre-fix DSC
# from dissertation §4.11.1). Only the two matched-generator variants
# are included; exp1a/exp1b are exp1c-with-no-PatchGAN and were never
# put through RAovSeg augmentation so we have no DSC for them.
PREFIX_DATA = {
    "exp1c_concat_pre": {
        "fid": 166.48, "lpips_mean": 0.7732, "hist_kl": 5.7929,
        "ovary_mean": np.nan,       # not measured pre-fix in mech csv
        "in_window_pct": np.nan,    # not measured pre-fix in mech csv
        "dsc": 0.053,               # pre-fix v3+PathB, from §4.11.1
        "phase": "pre",
    },
    "exp1c_spade_pre": {
        "fid": 188.14, "lpips_mean": 0.6992, "hist_kl": 7.2046,
        "ovary_mean": np.nan,
        "in_window_pct": np.nan,
        "dsc": 0.178,               # pre-fix v3+PathB n=8, from §4.11.1
        "phase": "pre",
    },
}

POSTFIX_VARIANTS = ["exp1c_concat", "exp1c_spade", "exp2", "exp2_lam05", "exp2_lam50"]
IMAGE_METRICS_ALL = ["fid", "lpips_mean", "hist_kl", "ovary_mean", "in_window_pct"]
DOWN_METRICS_ALL = ["dsc", "hd95_mm", "sensitivity", "volume_error"]


def _load_postfix() -> pd.DataFrame:
    """Load post-fix image + downstream metrics into one dataframe."""
    mech = pd.read_csv(REPO / "hpc_pulled/fixed_analysis/figures_fixed/mechanism/mech_ovary_intensity_table.csv")
    rows = []
    for v in POSTFIX_VARIANTS:
        q = json.load((REPO / f"metrics/fixed/quality_{v}_fixed.json").open())
        agg = json.load((REPO / f"metrics/fixed/agg/{v}_agg.json").open())
        mech_row = mech[mech["variant"] == f"{v.replace('exp1c_', '')}_fixed (pooled)"]
        if mech_row.empty:
            mech_row = mech[mech["variant"].str.startswith(v)]
        mech_lookup = {
            "ovary_mean": float(mech_row.iloc[0]["mean"]) if not mech_row.empty else np.nan,
            "in_window_pct": float(mech_row.iloc[0]["pct_in_window"]) if not mech_row.empty else np.nan,
        }
        rows.append({
            "variant": f"{v}_post",
            "phase": "post",
            "fid": q["fid"],
            "lpips_mean": q["lpips_nn"]["mean"],
            "hist_kl": q["hist_kl"],
            **mech_lookup,
            "dsc":          agg["dsc"]["mean"],
            "hd95_mm":      agg["hd95_mm"]["mean"],
            "sensitivity":  agg["sensitivity"]["mean"],
            "volume_error": agg["volume_error"]["mean"],
        })
    return pd.DataFrame(rows).set_index("variant")


def _make_combined_df(post_df: pd.DataFrame) -> pd.DataFrame:
    """Post + 2 pre-fix rows, aligned to same schema."""
    pre_rows = []
    for name, d in PREFIX_DATA.items():
        pre_rows.append({
            "variant": name,
            "phase": "pre",
            "fid": d["fid"],
            "lpips_mean": d["lpips_mean"],
            "hist_kl": d["hist_kl"],
            "ovary_mean": d["ovary_mean"],
            "in_window_pct": d["in_window_pct"],
            "dsc": d["dsc"],
            "hd95_mm": np.nan,
            "sensitivity": np.nan,
            "volume_error": np.nan,
        })
    pre_df = pd.DataFrame(pre_rows).set_index("variant")
    return pd.concat([pre_df, post_df])


def _pearson_bootstrap(x, y, n_boot=5000, ci=95):
    """Bootstrap r with resampling; return (r, lo, hi)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed=42)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        # Skip degenerate draws (same values → NaN r)
        try:
            r = stats.pearsonr(x[idx], y[idx])[0]
            if not np.isnan(r):
                rs.append(r)
        except Exception:
            pass
    if not rs:
        return np.nan, np.nan, np.nan
    r_pt = stats.pearsonr(x, y)[0]
    lo = np.percentile(rs, (100 - ci) / 2)
    hi = np.percentile(rs, 100 - (100 - ci) / 2)
    return r_pt, lo, hi


def analysis_combined(combined: pd.DataFrame) -> pd.DataFrame:
    """Correlations across n=7 (5 post + 2 pre) for the 3 metrics
    with matched pre-fix data (FID, LPIPS, hist_KL) vs DSC only."""
    print("\n=== (A) COMBINED pre-fix + post-fix (n=7) — FID/LPIPS/hist_KL vs DSC ===\n")
    print(combined[["phase", "fid", "lpips_mean", "hist_kl", "dsc"]].round(3))
    rows = []
    for im in ["fid", "lpips_mean", "hist_kl"]:
        r, lo, hi = _pearson_bootstrap(combined[im], combined["dsc"])
        rho, ps = stats.spearmanr(combined[im], combined["dsc"], nan_policy="omit")
        r_full, p_full = stats.pearsonr(
            combined[im].dropna(), combined.loc[combined[im].notna(), "dsc"]
        )
        rows.append({
            "image_metric": im,
            "downstream": "dsc",
            "n": int(combined[im].notna().sum()),
            "pearson_r": r_full, "pearson_p": p_full,
            "r_ci_lo": lo, "r_ci_hi": hi,
            "spearman_rho": rho, "spearman_p": ps,
        })
    df = pd.DataFrame(rows)
    print("\nCombined correlation table:")
    print(df.round(3).to_string(index=False))
    return df


def analysis_bootstrap(post: pd.DataFrame) -> pd.DataFrame:
    """5,000-draw bootstrap CIs on all 5×4 correlations (post-fix only)."""
    print("\n\n=== (B) BOOTSTRAP CIs on the 5×4 post-fix matrix (n=5, 5000 draws) ===\n")
    rows = []
    for im in IMAGE_METRICS_ALL:
        for dm in DOWN_METRICS_ALL:
            r, lo, hi = _pearson_bootstrap(post[im], post[dm])
            rows.append({
                "image_metric": im, "downstream": dm,
                "pearson_r": r, "r_ci_lo": lo, "r_ci_hi": hi,
                "signif_at_95": (lo * hi > 0),   # CI excludes 0
            })
    df = pd.DataFrame(rows)
    # Pretty pivot
    def _fmt(row):
        if np.isnan(row["pearson_r"]):
            return "n/a"
        star = "*" if row["signif_at_95"] else ""
        return f"{row['pearson_r']:+.2f} [{row['r_ci_lo']:+.2f},{row['r_ci_hi']:+.2f}]{star}"
    df["cell"] = df.apply(_fmt, axis=1)
    pivot = df.pivot(index="image_metric", columns="downstream", values="cell")
    pivot = pivot.reindex(index=IMAGE_METRICS_ALL, columns=DOWN_METRICS_ALL)
    print("\nr [95% CI] pivot (* = CI excludes 0):")
    print(pivot.to_string())
    return df


def main():
    post_df = _load_postfix()
    combined = _make_combined_df(post_df)

    out_dir = REPO / "hpc_pulled/fixed_analysis/figures_fixed/correlation_extended"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_combined = analysis_combined(combined)
    df_boot     = analysis_bootstrap(post_df)

    df_combined.to_csv(out_dir / "combined_correlation.csv", index=False)
    df_boot.to_csv(out_dir / "bootstrap_ci_matrix.csv", index=False)
    combined.to_csv(out_dir / "combined_variant_data.csv")

    # Bootstrap pretty pivot
    df_boot["cell"] = df_boot.apply(
        lambda row: (f"{row['pearson_r']:+.2f} [{row['r_ci_lo']:+.2f},{row['r_ci_hi']:+.2f}]"
                     + ("*" if row["signif_at_95"] else "")),
        axis=1
    )
    piv = df_boot.pivot(index="image_metric", columns="downstream", values="cell").reindex(
        index=IMAGE_METRICS_ALL, columns=DOWN_METRICS_ALL
    )
    piv.to_csv(out_dir / "bootstrap_ci_pivot.csv")

    print(f"\n\n[done] wrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
