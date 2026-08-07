"""
High-value results figures:
    fig_quality_heatmatrix.png    — Table 14 as a heat-styled matrix
    fig_dsc_forest.png            — DSC forest plot vs baselines (data-driven)
    fig_exp2_collapse.png         — exp2 synthetic output over training
    fig_tier1_landscape.png       — Tier-1 sweep DSC distribution (best/median/all)
    fig_tier1_metric_vs_dsc_*.png — cheap metric vs DSC scatter, one per target

The tier-1 figures require a per-trial aggregate CSV produced by
    python -m src.analysis.tier1_scatter_metrics_vs_dsc \\
        --sweep-root ... --out-dir figures/tier1_summary
which writes `tier1_all_trials.csv`. If that file isn't present, the tier-1
figures are silently skipped and the older figures are still produced.

Run:  python -m src.RaovSeg_recreation.make_result_figures --root . --out-dir figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "font.size": 11})


def fig_quality_heatmatrix(root: Path, out_dir: Path) -> None:
    rows = {r["variant"]: r for r in csv.DictReader(open(root / "metrics/master_metrics.csv"))}
    order = ["exp1a_ddpm_concat", "exp1b_ddpm_spade", "exp1c_concat", "exp1c_spade"]
    ylab = {"exp1a_ddpm_concat": "1a concat", "exp1b_ddpm_spade": "1b SPADE",
            "exp1c_concat": "1c concat+PG", "exp1c_spade": "1c SPADE+PG"}
    # (column, label, direction) direction +1 = higher better, -1 = lower better
    metrics = [("CLR_uterus", "CLR\nuterus", 1), ("CLR_ov_L", "CLR\nov_L", 1),
               ("CLR_em", "CLR\nem", 1), ("OSI_max_organ_corr", "OSI\norgan", 1),
               ("fid", "FID", -1), ("hist_kl", "hist_KL", -1), ("lpips_nn_mean", "LPIPS", -1)]

    def val(v, m):
        x = rows[v][m]
        return float(x) if x not in ("", "nan", "NaN") else np.nan

    M = np.array([[val(v, m) for m, _, _ in metrics] for v in order])
    colors = np.full((*M.shape, 4), 0.92)
    cmap = plt.get_cmap("RdYlGn")
    for j, (_, _, d) in enumerate(metrics):
        col = M[:, j] * d
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            lo, hi = np.nanmin(col), np.nanmax(col)
            norm = (col - lo) / (hi - lo + 1e-9)
            for i in range(M.shape[0]):
                colors[i, j] = cmap(norm[i]) if valid[i] else (0.9, 0.9, 0.9, 1)

    fig, ax = plt.subplots(figsize=(1.35 * len(metrics) + 2.4, 0.85 * len(order) + 2.2))
    ax.imshow(colors, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([l for _, l, _ in metrics], fontsize=9.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([ylab[v] for v in order], fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "n/a" if np.isnan(v) else (f"{v:.0f}" if abs(v) >= 10 else f"{v:.3f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5); ax.tick_params(which="minor", length=0)
    ax.set_title("Generator quality metrics — green = best in column, red = worst\n"
                 "(CLR / OSI: higher is better · FID / hist_KL / LPIPS: lower is better)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_dir / "fig_quality_heatmatrix.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_quality_heatmatrix.png'}")


def _load_dsc_from_summary(root: Path, name: str) -> tuple[float, float] | None:
    """Load (mean, std) across seeds from a metrics/<name>_dsc_summary.json."""
    p = root / "metrics" / f"{name}_dsc_summary.json"
    if not p.exists():
        return None
    try:
        with p.open() as f:
            j = json.load(f)
    except Exception:
        return None
    means = [s["dsc_mean"] for s in j.get("seeds", [])
             if s.get("status") == "ok" and "dsc_mean" in s]
    if not means:
        return None
    return float(np.mean(means)), float(np.std(means))


def _tier1_dsc_rows(root: Path) -> list[tuple[str, float, float, str]]:
    """Return best / median / worst tier-1 trial rows for the DSC forest,
    if the tier1 aggregate CSV exists. Colour = orange (tier1-specific)."""
    csv_path = root / "figures" / "tier1_summary" / "tier1_all_trials.csv"
    if not csv_path.exists():
        return []
    dscs = []
    for r in csv.DictReader(csv_path.open()):
        v = r.get("dsc_ov")
        if v in (None, "", "None"):
            continue
        try:
            dscs.append((r["trial_id"], float(v)))
        except (TypeError, ValueError):
            continue
    if len(dscs) < 3:
        return []
    dscs.sort(key=lambda x: x[1], reverse=True)
    best = dscs[0]
    median = dscs[len(dscs) // 2]
    worst = dscs[-1]
    n = len(dscs)
    return [
        (f"Tier-1 best (trial {best[0]}, of {n})",  best[1],   0.0, "#DD8452"),
        (f"Tier-1 median (of {n})",                 median[1], 0.0, "#DD8452"),
        (f"Tier-1 worst (of {n})",                  worst[1],  0.0, "#DD8452"),
    ]


def fig_dsc_forest(root: Path, out_dir: Path) -> None:
    """DSC forest across experiments. Reads real values from
    metrics/*_dsc_summary.json when present; falls back to hard-coded
    values where JSONs don't exist (e.g., pre-JSON runs)."""
    # (label, name-or-None, fallback_mean, fallback_std, colour)
    candidates = [
        ("Recreation baseline (real-only)", None,        0.220, 0.290, "#404040"),
        ("Phase 1 · v3 SPADE (t=0.26)",     "spade_v3",  0.178, 0.054, "#4C72B0"),
        ("Phase 2 · exp2_pathC",            "exp2_pathC", 0.152, 0.054, "#8172B3"),
        ("Phase 2 · exp2_lam05 (λ=0.05)",   "lam05_pathC", 0.117, 0.112, "#8172B3"),
        ("Phase 1 · v3 concat (t=0.26)",    "concat_v3", 0.053, 0.056, "#C44E52"),
        ("Phase 2 · exp2 (λ=0.01, naive)",  "exp2",       0.020, 0.010, "#8172B3"),
    ]
    data: list[tuple[str, float, float, str]] = []
    for label, name, fm, fs, colour in candidates:
        loaded = _load_dsc_from_summary(root, name) if name else None
        if loaded is not None:
            m, s = loaded
            data.append((label, m, s, colour))
        else:
            data.append((label, fm, fs, colour))

    # Layer tier-1 rows on top if the aggregate exists
    tier1 = _tier1_dsc_rows(root)
    if tier1:
        data = tier1 + data
        print(f"[dsc_forest] added {len(tier1)} tier-1 rows")

    data = data[::-1]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(data) + 2.2))
    for yi, (name, m, s, c) in zip(y, data):
        ax.errorbar(m, yi, xerr=s, fmt="o", color=c, ecolor=c, elinewidth=2.2,
                    capsize=4, ms=9, zorder=3)
        ax.text(m, yi + 0.22, f"{m:.3f}", ha="center", fontsize=8.5, color=c)
    ax.axvline(0.290, ls="--", color="#404040", lw=1.5, zorder=1)
    ax.text(0.290, len(data) - 0.35, "paper 0.290", color="#404040", fontsize=9, ha="center")
    ax.axvline(0.220, ls=":", color="#777777", lw=1.4, zorder=1)
    ax.text(0.220, -0.75, "recreation 0.220", color="#777777", fontsize=8.5, ha="center")
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data])
    ax.set_xlabel("Ovary DSC (mean ± std across seeds; tier-1 rows show single-trial value)")
    ax.set_xlim(-0.02, 0.56); ax.set_ylim(-1.2, len(data) - 0.3)
    title = "Cross-experiment DSC"
    if tier1:
        title += " — including tier-1 sweep best / median / worst"
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "fig_dsc_forest.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_dsc_forest.png'}")


def _load_tier1_csv(root: Path) -> list[dict] | None:
    """Read the tier-1 aggregate CSV if present. Returns list of row dicts
    with numeric fields converted."""
    p = root / "figures" / "tier1_summary" / "tier1_all_trials.csv"
    if not p.exists():
        return None
    out = []
    for r in csv.DictReader(p.open()):
        row: dict = {}
        for k, v in r.items():
            if v in ("", "None", None):
                row[k] = None
                continue
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        out.append(row)
    return out or None


def fig_tier1_landscape(root: Path, out_dir: Path) -> None:
    """DSC distribution across tier-1 trials + best-vs-worst per-trial box.
    Skipped silently if the tier-1 aggregate CSV isn't present."""
    rows = _load_tier1_csv(root)
    if rows is None:
        print("[skip] fig_tier1_landscape — no tier1_all_trials.csv found")
        return

    dsc_ov = np.array([r["dsc_ov"] for r in rows if r.get("dsc_ov") is not None], float)
    dsc_ut = np.array([r["dsc_ut"] for r in rows if r.get("dsc_ut") is not None], float)
    if dsc_ov.size < 3:
        print(f"[skip] fig_tier1_landscape — only {dsc_ov.size} ov points")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.hist(dsc_ov, bins=20, color="#4C72B0", alpha=0.7,
            edgecolor="white", label=f"OV  (n={dsc_ov.size})")
    if dsc_ut.size >= 3:
        ax.hist(dsc_ut, bins=20, color="#DD8452", alpha=0.6,
                edgecolor="white", label=f"UT  (n={dsc_ut.size})")
    ax.axvline(0.290, ls="--", color="#404040", lw=1.4)
    ax.text(0.290, ax.get_ylim()[1] * 0.9, " paper 0.290", fontsize=9)
    ax.axvline(0.220, ls=":", color="#777777", lw=1.2)
    ax.set_xlabel("DSC")
    ax.set_ylabel("# trials")
    ax.set_title(f"Tier-1 DSC distribution — best OV {dsc_ov.max():.3f}, median {np.median(dsc_ov):.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if dsc_ut.size >= 3:
        n_paired = min(dsc_ov.size, dsc_ut.size)
        # Align by index (each row from the CSV) so we plot only paired trials
        paired_ov = [r["dsc_ov"] for r in rows if r.get("dsc_ov") is not None and r.get("dsc_ut") is not None]
        paired_ut = [r["dsc_ut"] for r in rows if r.get("dsc_ov") is not None and r.get("dsc_ut") is not None]
        ax.scatter(paired_ov, paired_ut, alpha=0.6, s=32, color="#4C72B0")
        m, b = np.polyfit(paired_ov, paired_ut, 1) if len(paired_ov) >= 2 else (0.0, 0.0)
        xr = np.linspace(min(paired_ov), max(paired_ov), 100)
        ax.plot(xr, m * xr + b, color="#C44E52", ls="--", lw=1.0)
        ax.plot([0, 0.5], [0, 0.5], color="#888", ls=":", lw=0.8)
        ax.set_xlabel("DSC ovary")
        ax.set_ylabel("DSC uterus")
        ax.set_title(f"OV vs UT — n={len(paired_ov)} paired trials")
    else:
        ax.text(0.5, 0.5, "uterus data not present\n(pre-2026-07 tier1 trials)",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_axis_off()
    ax.grid(alpha=0.3)

    fig.suptitle("Tier-1 sweep landscape", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fig_tier1_landscape.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_tier1_landscape.png'}")


def fig_tier1_metric_vs_dsc(root: Path, out_dir: Path) -> None:
    """Do the cheap metrics predict downstream DSC? 4-panel scatter per target.
    Silently skipped if tier1_all_trials.csv is missing or too small."""
    rows = _load_tier1_csv(root)
    if rows is None:
        print("[skip] fig_tier1_metric_vs_dsc — no tier1_all_trials.csv found")
        return
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("[skip] fig_tier1_metric_vs_dsc — scipy not installed")
        return

    for target in ("ov", "ut"):
        dsc_col = f"dsc_{target}"
        dsc = np.array([r.get(dsc_col) for r in rows], dtype=object)
        # Convert to float, keep NaN for None
        dsc = np.array([np.nan if v is None else float(v) for v in dsc], float)
        if np.isfinite(dsc).sum() < 3:
            print(f"[skip] fig_tier1_metric_vs_dsc target={target} (n<3)")
            continue

        others = [
            ("hist_kl",         "intensity-hist KL (real ‖ synth)"),
            ("iscs_composite",  "ISCS composite in-band frac"),
            (f"dsc_{'ut' if target == 'ov' else 'ov'}",
             f"DSC on the other target ({'uterus' if target == 'ov' else 'ovary'})"),
            (f"hd95_{target}",  f"HD95 [mm] on {target}"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, (col, label) in zip(axes.ravel(), others):
            xs = np.array([r.get(col) for r in rows], dtype=object)
            xs = np.array([np.nan if v is None else float(v) for v in xs], float)
            finite = np.isfinite(xs) & np.isfinite(dsc)
            if finite.sum() < 3:
                ax.text(0.5, 0.5, f"insufficient data\nfor {col}", ha="center", va="center",
                        transform=ax.transAxes, color="#888")
                ax.set_title(label, fontsize=10); continue
            r, p = spearmanr(xs[finite], dsc[finite])
            ax.scatter(xs[finite], dsc[finite], alpha=0.7, s=32, color="#4C72B0")
            if finite.sum() >= 3:
                m, b = np.polyfit(xs[finite], dsc[finite], 1)
                xr = np.linspace(xs[finite].min(), xs[finite].max(), 100)
                ax.plot(xr, m * xr + b, color="#C44E52", ls="--", lw=1.0, alpha=0.7)
            ax.set_xlabel(label)
            ax.set_ylabel(f"DSC ({target})")
            ax.set_title(f"ρ = {r:+.3f}  (p = {p:.3g},  n = {int(finite.sum())})", fontsize=10.5)
            ax.grid(alpha=0.3)

        fig.suptitle(f"Do the cheap metrics predict DSC ({target})? — "
                     f"{sum(np.isfinite(dsc))} tier-1 trial×seed combos",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_png = out_dir / f"fig_tier1_metric_vs_dsc_{target}.png"
        fig.savefig(out_png); plt.close(fig)
        print(f"[saved] {out_png}")


def fig_exp2_collapse(root: Path, out_dir: Path) -> None:
    steps = [("5k", "step_005000.png"), ("25k", "step_025000.png"),
             ("50k", "step_050000.png"), ("75k", "step_075000.png"),
             ("100k", "step_100000_final.png")]
    crop = (240, 33, 470, 235)                      # synthetic panel, top row
    have = [(l, f) for l, f in steps if (root / "exp2_samples" / f).exists()]
    if not have:
        print("[skip] exp2_samples frames not found"); return
    fig, axes = plt.subplots(1, len(have), figsize=(2.5 * len(have), 3.1))
    for ax, (lab, f) in zip(np.atleast_1d(axes), have):
        ax.imshow(Image.open(root / "exp2_samples" / f).crop(crop), cmap="gray")
        ax.set_title(f"step {lab}", fontsize=10.5); ax.axis("off")
    fig.suptitle("exp2 (Phase 2) synthetic output over training — it emerges from a gray blob into "
                 "plausible D1-style\nanatomy, but never acquires D2's fat-suppressed T2FS appearance "
                 "(MSE on D1 dominates the weak D2-style adversarial signal)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_dir / "fig_exp2_collapse.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_exp2_collapse.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_quality_heatmatrix(args.root, args.out_dir)
    fig_dsc_forest(args.root, args.out_dir)
    fig_exp2_collapse(args.root, args.out_dir)
    # Tier-1 figures — skipped silently if the aggregate CSV isn't present.
    fig_tier1_landscape(args.root, args.out_dir)
    fig_tier1_metric_vs_dsc(args.root, args.out_dir)


if __name__ == "__main__":
    main()
