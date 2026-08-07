#!/usr/bin/env python3
"""
Tier-1 sweep aggregator + "do the cheap metrics predict downstream DSC?"
scatter plots.

For every completed tier-1 trial (trial_<id>/seed_<n>/DONE marker exists),
load:
    metrics_ov.json      → DSC_ov (mean over 8 test subjects)
    metrics_ut.json      → DSC_ut  (if present)
    iscs_score.json      → composite ISCS in-band score
    hist_kl.json         → intensity-histogram KL vs real (added Jul 2026)

Optional per-trial config (`config.json` written by the coordinator) is
also merged in if present — knobs like ovary_target_intensity, iscs_alpha,
etc. This lets you filter/colour scatter points by trial parameters.

Outputs:
    <out-dir>/tier1_all_trials.csv         — one row per (trial, seed) combo
    <out-dir>/tier1_metric_vs_dsc_ov.png   — 4-panel scatter: hist_kl, iscs,
                                             dsc_ut vs dsc_ov, with Spearman ρ
    <out-dir>/tier1_metric_vs_dsc_ut.png   — same panels but vs dsc_ut
    <out-dir>/tier1_correlations.json      — full Spearman/Pearson table

Usage:
    python -m src.analysis.tier1_scatter_metrics_vs_dsc \\
        --sweep-root /mnt/parscratch/.../synth_mri/sweep/tier1 \\
        --out-dir    figures/tier1_summary

Interpretation:
    - Strong positive ρ between an auxiliary metric and DSC → that metric
      is a useful proxy for downstream utility; use it to filter future
      sweep params.
    - Weak or opposite ρ → the metric measures something orthogonal to
      DSC utility; don't use it to gate trials.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"  skip {p}: {type(e).__name__}: {e}")
        return None


def _mean_from_per_subject(metrics: dict, metric_key: str = "dsc",
                           mode: str = "full") -> float | None:
    """Aggregate per-subject metric across the 8 test subjects → mean."""
    per_subj = metrics.get("per_subject", {})
    if not per_subj:
        # Fall back to top-level aggregate if per-subject not present
        agg = metrics.get("aggregate", {})
        for k in (f"{metric_key}_mean", metric_key):
            if k in agg:
                try:
                    return float(agg[k])
                except (TypeError, ValueError):
                    pass
        return None
    m = mode if mode in per_subj else next(iter(per_subj))
    vals = []
    for rec in per_subj[m]:
        v = rec.get(metric_key)
        if v is None:
            for k in ("dice", "dsc_mean"):
                if k in rec:
                    v = rec[k]; break
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return float(np.mean(vals)) if vals else None


def _iscs_composite(iscs: dict) -> float | None:
    """Overall in-band fraction from score_synth_iscs.py output."""
    if iscs is None:
        return None
    for k in ("overall_in_band_frac", "composite", "score", "overall"):
        if k in iscs:
            try:
                return float(iscs[k])
            except (TypeError, ValueError):
                pass
    per_organ = iscs.get("per_organ", {}) or {}
    fracs = []
    for organ, entry in per_organ.items():
        if isinstance(entry, dict) and "in_band_frac" in entry:
            try:
                fracs.append(float(entry["in_band_frac"]))
            except (TypeError, ValueError):
                pass
    return float(np.mean(fracs)) if fracs else None


def _kl_from_hist_kl(hist_kl: dict) -> float | None:
    if hist_kl is None:
        return None
    v = hist_kl.get("kl_real_synth")
    if v is None:
        v = hist_kl.get("jsd")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _walk_sweep(sweep_root: Path) -> List[Dict[str, Any]]:
    """Yield one row per completed (trial_ID, seed_N) combo under
    sweep_root/trial_*/seed_*/. Only includes rows where DONE marker
    exists and metrics_ov.json is loadable."""
    rows: List[Dict[str, Any]] = []
    trial_dirs = sorted([p for p in sweep_root.glob("trial_*") if p.is_dir()])
    for tdir in trial_dirs:
        trial_id = tdir.name.replace("trial_", "")
        # Coordinator sometimes writes trial-level config; try to pick it up.
        trial_cfg = _load_json(tdir / "config.json") or _load_json(tdir / "params.json") or {}
        for sdir in sorted(p for p in tdir.glob("seed_*") if p.is_dir()):
            if not (sdir / "DONE").exists():
                continue
            m_ov = _load_json(sdir / "metrics_ov.json")
            if m_ov is None:
                continue
            row: Dict[str, Any] = {
                "trial_id": trial_id,
                "seed":     sdir.name.replace("seed_", ""),
                "dsc_ov":   _mean_from_per_subject(m_ov, "dsc"),
                "hd95_ov":  _mean_from_per_subject(m_ov, "hd95_mm"),
                "sens_ov":  _mean_from_per_subject(m_ov, "sensitivity"),
                "prec_ov":  _mean_from_per_subject(m_ov, "precision"),
            }
            m_ut = _load_json(sdir / "metrics_ut.json")
            if m_ut is not None:
                row["dsc_ut"]  = _mean_from_per_subject(m_ut, "dsc")
                row["hd95_ut"] = _mean_from_per_subject(m_ut, "hd95_mm")
                row["sens_ut"] = _mean_from_per_subject(m_ut, "sensitivity")
                row["prec_ut"] = _mean_from_per_subject(m_ut, "precision")
            iscs = _load_json(sdir / "iscs_score.json")
            row["iscs_composite"] = _iscs_composite(iscs) if iscs else None
            hkl = _load_json(sdir / "hist_kl.json")
            row["hist_kl"] = _kl_from_hist_kl(hkl) if hkl else None
            # Merge in any trial config keys as extra columns
            for k, v in trial_cfg.items():
                if isinstance(v, (int, float, str, bool)):
                    row[f"cfg_{k}"] = v
            rows.append(row)
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        raise SystemExit("scipy not installed. `pip install scipy` and retry.")
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return (float("nan"), float("nan"))
    r, p = spearmanr(x[finite], y[finite])
    return (float(r), float(p))


def _scatter_grid(rows: List[Dict[str, Any]], target: str, out_png: Path,
                  corr_json: Dict[str, Any]) -> None:
    """4-panel scatter of auxiliary metrics vs downstream DSC for `target`."""
    dsc_col = f"dsc_{target}"
    others = [
        ("hist_kl",        "intensity-hist KL (real ‖ synth)"),
        ("iscs_composite", "ISCS composite (in-band frac)"),
        (f"dsc_{'ut' if target == 'ov' else 'ov'}",
         f"DSC on the OTHER target ({'uterus' if target == 'ov' else 'ovary'})"),
        ("hd95_" + target, f"HD95 [mm] on {target}"),
    ]

    dsc = np.array([r.get(dsc_col, np.nan) for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for ax, (col, label) in zip(axes.ravel(), others):
        xs = np.array([r.get(col, np.nan) for r in rows], dtype=float)
        finite = np.isfinite(xs) & np.isfinite(dsc)
        if finite.sum() < 3:
            ax.text(0.5, 0.5, f"insufficient data\nfor {col}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888")
            ax.set_title(label, fontsize=10)
            continue
        r, p = _spearman(xs, dsc)
        ax.scatter(xs[finite], dsc[finite], alpha=0.7, s=32, color="#4C72B0")
        # Linear fit for eye guidance (not overplotting a claim)
        if finite.sum() >= 3:
            m, b = np.polyfit(xs[finite], dsc[finite], 1)
            xr = np.linspace(xs[finite].min(), xs[finite].max(), 100)
            ax.plot(xr, m * xr + b, color="#C44E52", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel(f"DSC ({target})")
        ax.set_title(f"ρ = {r:+.3f}  (p = {p:.3g},  n = {int(finite.sum())})",
                     fontsize=10.5)
        ax.grid(alpha=0.3)

        corr_json.setdefault(f"vs_dsc_{target}", {})[col] = {
            "spearman_rho": r, "p_value": p, "n": int(finite.sum()),
        }

    fig.suptitle(f"Do the cheap metrics predict downstream DSC ({target})? — "
                 f"{sum(np.isfinite(dsc))} trial×seed combos",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--sweep-root", type=Path, required=True,
                    help="Root under which trial_<id>/seed_<n>/... dirs live")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", default=["ov", "ut"],
                    help="Which target scatter panels to produce (default: ov ut)")
    args = ap.parse_args()

    print(f"[tier1] scanning {args.sweep_root} ...")
    rows = _walk_sweep(args.sweep_root)
    print(f"[tier1] loaded {len(rows)} completed (trial, seed) rows")

    if not rows:
        raise SystemExit("no completed trials found — nothing to plot.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV dump ---
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    csv_path = args.out_dir / "tier1_all_trials.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[saved] {csv_path}")

    # --- Scatter plots + correlations ---
    corr: Dict[str, Any] = {}
    for tgt in args.targets:
        col = f"dsc_{tgt}"
        n_present = sum(1 for r in rows if r.get(col) is not None)
        if n_present < 3:
            print(f"[tier1] skip target={tgt}: only {n_present} rows have {col}")
            continue
        _scatter_grid(rows, tgt, args.out_dir / f"tier1_metric_vs_dsc_{tgt}.png", corr)

    with (args.out_dir / "tier1_correlations.json").open("w") as f:
        json.dump({
            "sweep_root": str(args.sweep_root),
            "n_rows": len(rows),
            "correlations": corr,
        }, f, indent=2)
    print(f"[saved] {args.out_dir / 'tier1_correlations.json'}")

    # --- Console summary ---
    print("\n=== Spearman ρ summary ===")
    for target_key, corrs in corr.items():
        print(f"\n{target_key}:")
        for col, s in sorted(corrs.items(), key=lambda kv: -abs(kv[1]["spearman_rho"])):
            print(f"  {col:<25} ρ={s['spearman_rho']:+.3f}  "
                  f"p={s['p_value']:.3g}  n={s['n']}")


if __name__ == "__main__":
    main()
