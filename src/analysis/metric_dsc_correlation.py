#!/usr/bin/env python3
"""
Metric-DSC correlation analysis.

For each generator variant, correlates synthetic quality metrics
(in_window %, ovary mean intensity, CLR, and optionally FID / LPIPS /
hist_KL from a master metrics CSV) against downstream RAovSeg DSC
(mean across seeds).

Produces:
    - stdout summary table with Pearson r and Spearman rho + p-values
    - correlation_table.csv
    - scatter_<metric>.png per metric
    - summary_grid.png (all scatter plots in one figure)

See METRIC_DSC_CORRELATION.md for the methodology.

Usage:
    python -m src.analysis.metric_dsc_correlation \\
        --mechanism-csv figures_fixed/mechanism/mech_ovary_intensity_table.csv \\
        --dsc-root /mnt/parscratch/users/$USER/synth_mri/runs \\
        --variants exp1c_concat exp1c_spade exp2 exp2_lam05 exp2_lam50 \\
        --n-seeds 3 \\
        --out-dir figures_fixed/correlation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:
    stats = None


def _pearson(x, y):
    """Pearson r and two-sided p-value. Falls back to numpy if scipy missing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan"), float("nan")
    if stats is not None:
        r, p = stats.pearsonr(x, y)
        return float(r), float(p)
    # Manual fallback (no p-value)
    r = float(np.corrcoef(x, y)[0, 1])
    return r, float("nan")


def _spearman(x, y):
    """Spearman rho and two-sided p-value. Falls back to numpy if scipy missing."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return float("nan"), float("nan")
    if stats is not None:
        rho, p = stats.spearmanr(x, y)
        return float(rho), float(p)
    # Manual rank-based fallback
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, float("nan")


def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform confidence interval on Pearson r."""
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = 1.959963984540054  # two-tailed 95%
    lo, hi = z - z_crit * se, z + z_crit * se
    return math.tanh(lo), math.tanh(hi)


def _interpret(r: float) -> str:
    if not math.isfinite(r):
        return "insufficient data"
    ar = abs(r)
    if ar > 0.9:
        return "solid predictor"
    if ar > 0.7:
        return "suggestive"
    if ar > 0.4:
        return "weak signal"
    return "no evidence"


def load_mechanism_metrics(csv_path: Path, variants: list[str]) -> dict:
    """Read the mechanism CSV and pick out the pooled row per variant.

    The mechanism CSV has a `variant` column with entries like
    'spade_fixed (pooled)' — we match against the variant token.
    """
    out = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = row["variant"]
            if "pooled" not in key.lower():
                continue
            # Extract the variant token — e.g., 'spade_fixed (pooled)' → 'spade_fixed'
            token = key.split("(")[0].strip()
            # Map short names back to canonical variant names
            for v in variants:
                short = v.replace("exp1c_", "")
                if token in (v, f"{v}_fixed", short, f"{short}_fixed"):
                    # Column names in mechanism CSV: `pct_in_window`, `mean`, `median`
                    # (fall back to `in_window%` for older CSV format)
                    iw_raw = row.get("pct_in_window") or row.get("in_window%") or "nan"
                    try:
                        iw = float(str(iw_raw).rstrip("%"))
                    except ValueError:
                        iw = float("nan")
                    out[v] = {
                        "in_window_pct": iw,
                        "ovary_mean": float(row["mean"]),
                        "ovary_median": float(row["median"]),
                    }
                    break
    return out


def load_dsc(dsc_root: Path, variants: list[str], n_seeds: int) -> dict:
    """Read metrics_ov.json for each (variant, seed) and aggregate DSC."""
    out = {}
    for v in variants:
        dsc_list = []
        for s in range(n_seeds):
            metrics_path = dsc_root / f"raov_aug_{v}_fixed_seed{s}" / "metrics_ov.json"
            if not metrics_path.exists():
                continue
            try:
                d = json.loads(metrics_path.read_text())
                dsc = d["aggregate"]["full"]["dsc"]["mean"]
                dsc_list.append(dsc)
            except (KeyError, json.JSONDecodeError):
                pass
        if dsc_list:
            out[v] = {
                "dsc_mean": float(np.mean(dsc_list)),
                "dsc_std": float(np.std(dsc_list, ddof=1)) if len(dsc_list) > 1 else 0.0,
                "n_seeds_found": len(dsc_list),
            }
    return out


def load_master_metrics(csv_path: Path, variants: list[str]) -> dict:
    """Optional: read FID / LPIPS / hist_KL from master_metrics.csv if provided."""
    if not csv_path or not csv_path.exists():
        return {}
    out = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            variant = row.get("variant", "").strip()
            for v in variants:
                if variant in (v, f"{v}_fixed"):
                    def _get(key):
                        val = row.get(key, "")
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            return float("nan")
                    out[v] = {
                        "FID": _get("FID"),
                        "LPIPS": _get("LPIPS"),
                        "hist_KL": _get("hist_KL"),
                    }
                    break
    return out


def load_clr(clr_root: Path, variants: list[str]) -> dict:
    """Optional: read CLR (uterus) from explain.py output per variant."""
    if not clr_root:
        return {}
    out = {}
    for v in variants:
        # Map to the explain.py output path convention used in the sbatch scripts
        if v == "exp1c_concat":
            path = clr_root / "1c/concat/explain/sample_00_metrics.json"
        elif v == "exp1c_spade":
            path = clr_root / "1c/spade/explain/sample_00_metrics.json"
        else:
            path = clr_root / f"phase2/{v}/explain/sample_00_metrics.json"
        if path.exists():
            try:
                d = json.loads(path.read_text())
                # Prefer uterus CLR (used in make_clr_counterfactual figure);
                # fall back to average across organs if present
                clr = d.get("CLR_per_channel", {}).get("uterus")
                if clr is None:
                    per_ch = d.get("CLR_per_channel", {})
                    if per_ch:
                        clr = float(np.mean(list(per_ch.values())))
                out[v] = {"CLR": float(clr)} if clr is not None else {}
            except (KeyError, json.JSONDecodeError):
                pass
    return out


def scatter_plot(ax, xs, ys, y_errs, labels, xlabel, ylabel, title):
    """One scatter plot with linear fit + labeled points."""
    ax.errorbar(xs, ys, yerr=y_errs, fmt="o", color="#4a89dc",
                capsize=4, ms=8, alpha=0.85, ecolor="#88a", elinewidth=1)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, xy=(x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#333")
    # Linear fit if we have enough non-NaN pairs
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() >= 3:
        z = np.polyfit(np.asarray(xs)[mask], np.asarray(ys)[mask], 1)
        xr = np.linspace(np.nanmin(xs), np.nanmax(xs), 50)
        ax.plot(xr, np.polyval(z, xr), "--", color="#d95f5f", alpha=0.6, lw=1.3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism-csv", type=Path, required=True)
    ap.add_argument("--dsc-root", type=Path, required=True,
                    help="Root dir containing raov_aug_<variant>_fixed_seed{0,1,2}/ subdirs")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--master-csv", type=Path, default=None,
                    help="Optional: master_metrics.csv for FID/LPIPS/hist_KL")
    ap.add_argument("--clr-root", type=Path, default=None,
                    help="Optional: root dir containing 1c/concat/explain/... etc.")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load all data
    mech = load_mechanism_metrics(args.mechanism_csv, args.variants)
    dsc = load_dsc(args.dsc_root, args.variants, args.n_seeds)
    master = load_master_metrics(args.master_csv, args.variants) if args.master_csv else {}
    clr = load_clr(args.clr_root, args.variants) if args.clr_root else {}

    print(f"\n=== data availability across {len(args.variants)} variants ===")
    for v in args.variants:
        m_ok = "Y" if v in mech else "-"
        d_ok = f"Y (n={dsc[v]['n_seeds_found']})" if v in dsc else "-"
        f_ok = "Y" if v in master else "-"
        c_ok = "Y" if v in clr else "-"
        print(f"  {v:24s}  mech={m_ok}  dsc={d_ok}  FID/LPIPS/KL={f_ok}  CLR={c_ok}")

    # Build parallel arrays
    kept = [v for v in args.variants if v in mech and v in dsc]
    if len(kept) < 3:
        print(f"\nERROR: only {len(kept)} variants have both mechanism + DSC data. Need >=3.")
        return

    dsc_means = np.array([dsc[v]["dsc_mean"] for v in kept])
    dsc_stds  = np.array([dsc[v]["dsc_std"]  for v in kept])

    metric_data = {
        "in_window_pct": np.array([mech[v]["in_window_pct"] for v in kept]),
        "ovary_mean":    np.array([mech[v]["ovary_mean"]    for v in kept]),
    }
    if all(v in clr for v in kept):
        metric_data["CLR"] = np.array([clr[v]["CLR"] for v in kept])
    if all(v in master for v in kept):
        for k in ("FID", "LPIPS", "hist_KL"):
            arr = np.array([master[v][k] for v in kept])
            if np.isfinite(arr).sum() >= 3:
                metric_data[k] = arr

    # Compute correlations
    n = len(kept)
    rows = []
    for name, xs in metric_data.items():
        pr, pp = _pearson(xs, dsc_means)
        sr, sp = _spearman(xs, dsc_means)
        ci_lo, ci_hi = _fisher_ci(pr, n)
        rows.append({
            "metric": name,
            "n": n,
            "pearson_r": pr,
            "pearson_p": pp,
            "pearson_ci_lo": ci_lo,
            "pearson_ci_hi": ci_hi,
            "spearman_rho": sr,
            "spearman_p": sp,
            "interpretation": _interpret(sr if math.isfinite(sr) else pr),
        })

    # Print table
    print(f"\n=== Metric <-> DSC correlation (n={n} variants: {', '.join(kept)}) ===")
    hdr = f"{'metric':<16} {'r':>7}  {'p_r':>6}  {'r 95% CI':<18} {'rho':>7}  {'p_rho':>6}   {'interpretation'}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        ci = f"[{row['pearson_ci_lo']:+.2f},{row['pearson_ci_hi']:+.2f}]"
        print(f"{row['metric']:<16} {row['pearson_r']:>+7.3f}  {row['pearson_p']:>6.3f}  "
              f"{ci:<18} {row['spearman_rho']:>+7.3f}  {row['spearman_p']:>6.3f}   {row['interpretation']}")

    # Write CSV
    csv_out = args.out_dir / "correlation_table.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {csv_out}")

    # Individual scatter plots + grid
    n_metrics = len(metric_data)
    ncols = min(3, n_metrics)
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, (name, xs) in enumerate(metric_data.items()):
        # Individual figure
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        r = rows[i]
        title1 = f"{name} vs DSC   (r={r['pearson_r']:+.2f}, rho={r['spearman_rho']:+.2f}, n={n})"
        scatter_plot(ax1, xs, dsc_means, dsc_stds, kept, name, "Ovary DSC (mean +/- std across seeds)", title1)
        fig1.tight_layout()
        fig1.savefig(args.out_dir / f"scatter_{name}.png", dpi=130, bbox_inches="tight")
        plt.close(fig1)

        # Grid entry
        scatter_plot(axes[i], xs, dsc_means, dsc_stds, kept, name, "DSC",
                     f"{name} (r={r['pearson_r']:+.2f}, rho={r['spearman_rho']:+.2f})")

    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Synth metrics vs downstream DSC (n={n} variants)", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_dir / "summary_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {args.out_dir}/scatter_*.png")
    print(f"[saved] {args.out_dir}/summary_grid.png")


if __name__ == "__main__":
    main()
