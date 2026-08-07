"""
The five data-driven result charts, read live from metrics/ (JSON + CSV):
    fig_downstream_trajectory.png   fig_phase2_collapse.png   fig_variance_study.png
    fig_quality_metrics.png         fig_clr_localisation.png

Previously these lived only inside make_dissertation_figures.ipynb; extracting
them here makes every dissertation figure reproducible from a script.

Run:  python -m src.RaovSeg_recreation.make_data_charts --root . --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "legend.frameon": False,
})

C = {"baseline": "#404040", "concat": "#C44E52", "concat_l": "#E39399",
     "spade": "#4C72B0", "spade_l": "#9BB4D6", "phase2": "#8172B3", "accent": "#DD8452"}
BASELINE = 0.290


def _load(root: Path):
    def load_json(name):
        p = root / name
        if not p.exists():
            p = root / "metrics" / Path(name).name
        return json.loads(p.read_text())

    def seed_means(s):
        return np.array([x["dsc_mean"] for x in s["seeds"] if x.get("status") == "ok"])

    return dict(
        spade=seed_means(load_json("metrics/variance_study_summary.json")),
        exp2=seed_means(load_json("exp2_dsc_summary.json")),
        exp2C=seed_means(load_json("exp2_pathC_dsc_summary.json")),
        lam05=seed_means(load_json("lam05_dsc_summary.json")),
        quality=pd.read_csv(root / "metrics" / "master_metrics.csv"),
    )


def fig_downstream_trajectory(d, out_dir: Path) -> None:
    spade_seeds = d["spade"]
    levels = ["v1\n(no fixes)", "v2\n(3 fixes)", "v3\n(+Path B)"]
    concat_dsc = [0.150, 0.044, 0.053]; concat_err = [0.006, 0.039, 0.056]
    spade_dsc = [0.138, 0.169, float(spade_seeds.mean())]
    spade_err = [0.049, 0.037, float(spade_seeds.std())]
    x = np.arange(len(levels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar(x - w / 2, concat_dsc, w, yerr=concat_err, capsize=4, color=C["concat"], label="concat conditioning")
    ax.bar(x + w / 2, spade_dsc, w, yerr=spade_err, capsize=4, color=C["spade"], label="SPADE conditioning")
    ax.axhline(BASELINE, ls="--", lw=1.6, color=C["baseline"])
    ax.text(len(levels) - 0.55, BASELINE + 0.006, f"real-only baseline = {BASELINE:.3f}",
            ha="right", va="bottom", color=C["baseline"], fontsize=9.5)
    for xi, v, e in zip(x - w / 2, concat_dsc, concat_err):
        ax.text(xi, v + e + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=C["concat"])
    for xi, v, e in zip(x + w / 2, spade_dsc, spade_err):
        ax.text(xi, v + e + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color=C["spade"])
    ax.set_xticks(x); ax.set_xticklabels(levels)
    ax.set_ylabel("Ovary DSC (8-subject test set)")
    ax.set_title("Augmentation never reaches the real-only baseline")
    ax.set_ylim(0, 0.34)
    ax.legend(loc="lower left", bbox_to_anchor=(0.005, 0.60), frameon=True,
              facecolor="white", framealpha=1.0, edgecolor="0.85")
    fig.savefig(out_dir / "fig_downstream_trajectory.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_downstream_trajectory.png'}")


def fig_phase2_collapse(d, out_dir: Path) -> None:
    s, e2, e2C, l05 = d["spade"], d["exp2"], d["exp2C"], d["lam05"]
    labels = ["Real-only\nbaseline", "SPADE v3\n(Phase 1)", "Phase 2\nexp2 (lam=.01)",
              "Phase 2\nPath C", "Phase 2\nlam=.05"]
    vals = [BASELINE, s.mean(), e2.mean(), e2C.mean(), l05.mean()]
    errs = [0, s.std(), e2.std(), e2C.std(), l05.std()]
    colors = [C["baseline"], C["spade"], C["phase2"], C["phase2"], C["phase2"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors, alpha=0.92)
    ax.axhline(BASELINE, ls="--", lw=1.2, color=C["baseline"], alpha=0.6)
    for b, v, er in zip(bars, vals, errs):
        txt = f"{v:.3f}" if abs(v - BASELINE) < 1e-9 else f"{v:.3f} ({(v - BASELINE) / BASELINE * 100:+.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + er + 0.012, txt, ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("Ovary DSC (8-subject test set)")
    ax.set_title("Cross-domain Phase 2 generators collapse below baseline")
    ax.set_ylim(0, 0.36)
    fig.savefig(out_dir / "fig_phase2_collapse.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_phase2_collapse.png'}")


def fig_variance_study(d, out_dir: Path) -> None:
    spade_seeds = d["spade"]
    n8_mean, n8_std = spade_seeds.mean(), spade_seeds.std()
    n3_mean = spade_seeds[:3].mean()
    xs = np.arange(len(spade_seeds))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.fill_between([-0.5, len(spade_seeds) - 0.5], n8_mean - n8_std, n8_mean + n8_std,
                    color=C["spade"], alpha=0.22, label=f"n=8 +/-1 SD ({n8_std:.3f})")
    ax.axhline(n8_mean, color=C["spade"], lw=2.6, label=f"n=8 mean = {n8_mean:.3f}")
    ax.axhline(n3_mean, color=C["accent"], ls=":", lw=2.4, label=f"first 3 seeds = {n3_mean:.3f}")
    ax.axhline(BASELINE, ls="--", lw=1.4, color=C["baseline"], label=f"baseline = {BASELINE:.3f}")
    ax.scatter(xs, spade_seeds, s=85, color=C["spade"], edgecolor="white", linewidths=1.0,
               zorder=5, label="per-seed DSC")
    ax.annotate("seeds 3–7 pull the mean\nbelow the n=3 reading", xy=(3, spade_seeds[3]),
                xytext=(3.3, 0.045), fontsize=8.5, color=C["accent"],
                arrowprops=dict(arrowstyle="->", color=C["accent"], lw=1.1))
    ax.set_xticks(xs); ax.set_xticklabels([f"s{i}" for i in xs])
    ax.set_xlim(-0.5, len(spade_seeds) - 0.5); ax.set_ylim(0, 0.43)
    ax.set_xlabel("training seed"); ax.set_ylabel("Ovary DSC")
    ax.set_title("More seeds erase the apparent benefit (0.218 -> 0.178)")
    ax.legend(fontsize=8.5, ncol=2, loc="upper center", frameon=True,
              facecolor="white", framealpha=1.0, edgecolor="0.85")
    fig.savefig(out_dir / "fig_variance_study.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_variance_study.png'}")


def fig_quality_metrics(d, out_dir: Path) -> None:
    qm = d["quality"].copy()
    qm["short"] = ["1a concat", "1b spade", "1c concat", "1c spade"]
    bar_colors = [C["concat"], C["spade"], C["concat"], C["spade"]]
    cols = [("fid", "FID  (lower better)"), ("hist_kl", "hist_KL  (lower better)"),
            ("lpips_nn_mean", "LPIPS-NN  (lower better)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (col, title) in zip(axes, cols):
        ax.bar(qm["short"], qm[col], color=bar_colors)
        best = qm[col].idxmin()
        ax.scatter(best, qm[col].iloc[best], marker="*", s=180, color="#F2C744",
                   edgecolor="black", zorder=5, clip_on=False)
        ax.set_title(title, fontsize=11); ax.tick_params(axis="x", rotation=30); ax.margins(y=0.15)
    fig.suptitle("No single variant wins on quality  (★ = best; concat = red, SPADE = blue)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "fig_quality_metrics.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_quality_metrics.png'}")


def fig_clr_localisation(d, out_dir: Path) -> None:
    qm = d["quality"].copy()
    qm["short"] = ["1a concat", "1b spade", "1c concat", "1c spade"]
    chans = {"uterus": "CLR_uterus", "ovary (L)": "CLR_ov_L", "endometrioma": "CLR_em"}
    cx = np.arange(len(chans)); w = 0.2
    var_colors = [C["concat_l"], C["spade_l"], C["concat"], C["spade"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for i, (_, row) in enumerate(qm.iterrows()):
        ax.bar(cx + (i - 1.5) * w, [row[c] for c in chans.values()], w, label=row["short"], color=var_colors[i])
    ax.axhspan(0, 0.1, color="#C44E52", alpha=0.06)
    ax.text(len(chans) - 1, 0.105, "locked-out zone (CLR < 0.1)", ha="right", va="bottom",
            fontsize=8.5, color=C["concat"])
    ax.set_xticks(cx); ax.set_xticklabels(list(chans.keys()))
    ax.set_ylabel("Counterfactual Localisation Ratio (higher = better)")
    ax.set_title("SPADE localises per-organ; concat does not")
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_clr_localisation.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_clr_localisation.png'}")


def make_all(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _load(root)
    fig_downstream_trajectory(d, out_dir)
    fig_phase2_collapse(d, out_dir)
    fig_variance_study(d, out_dir)
    fig_quality_metrics(d, out_dir)
    fig_clr_localisation(d, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    make_all(args.root, args.out_dir)


if __name__ == "__main__":
    main()
