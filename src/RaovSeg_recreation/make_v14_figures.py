#!/usr/bin/env python3
"""v14 figures — regenerated locally from DATA CSVs and quality JSONs.

Produces four figures aligned with the v14 §3 spec:

  figures/fig_v14_dsc_forest.png          dual-anchor DSC forest (post-fix)
  figures/fig_v14_per_subject_dsc.png     per-subject DSC across variants
  figures/fig_v14_detection_vs_dsc.png    detection-rate vs mean DSC scatter
  figures/fig_v14_ablation_5seed.png      DATA-5 ablation with paper anchors

Reads:
  metrics/data1_per_subject_per_seed_ovary_dsc.csv
  metrics/data3_detection_vs_delineation.csv
  metrics/data5_ablations_5seed.csv
  hpc_pulled/runs/*/quality.json  (optional; only used for DSC-vs-quality overlay)

Anchor policy (v14 §3):
  0.290 — published RAovSeg baseline (paper, n=1)
  0.189 — in-house recreation baseline (n=5 seeds; see DATA-5 `full`)

Both anchors are drawn on all figures where a baseline reference is meaningful.
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- constants -----------------------------------------------------------

ANCHOR_PUB = 0.290       # published RAovSeg
ANCHOR_INHOUSE = 0.189   # in-house recreation (DATA-5 full mean)

VARIANT_ORDER = [
    "raov_aug_exp1c_concat_fixed",
    "raov_aug_exp1c_spade_fixed",
    "raov_aug_exp2_fixed",
    "raov_aug_exp2_lam05_fixed",
    "raov_aug_exp2_lam50_fixed",
    "raovseg_real_only",
]
DISPLAY = {
    "raov_aug_exp1c_concat_fixed": "concat_pg (fixed, n=3)",
    "raov_aug_exp1c_spade_fixed":  "spade_pg (fixed, n=3)",
    "raov_aug_exp2_fixed":         "xdom_l001 (fixed, n=3)",
    "raov_aug_exp2_lam05_fixed":   "xdom_l005 (fixed, n=3)",
    "raov_aug_exp2_lam50_fixed":   "xdom_l050 (fixed, n=3)",
    "raovseg_real_only":           "recreation baseline (n=5)",
}
COLOUR = {
    "raov_aug_exp1c_concat_fixed": "#C44E52",
    "raov_aug_exp1c_spade_fixed":  "#4C72B0",
    "raov_aug_exp2_fixed":         "#8172B3",
    "raov_aug_exp2_lam05_fixed":   "#937860",
    "raov_aug_exp2_lam50_fixed":   "#DA8BC3",
    "raovseg_real_only":           "#404040",
}
UNIV_FAIL = {"D2-005", "D2-023"}  # universal-failure subjects (recreation gets 0.00)


# --- loading -------------------------------------------------------------

def load_data1(root: Path):
    with (root / "metrics/data1_per_subject_per_seed_ovary_dsc.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["dsc"] = float(r["dsc"])
    return rows

def load_data3(root: Path):
    with (root / "metrics/data3_detection_vs_delineation.csv").open() as f:
        return list(csv.DictReader(f))

def load_data5(root: Path):
    with (root / "metrics/data5_ablations_5seed.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ovary_dsc"] = float(r["ovary_dsc"])
    return rows

def load_data4(root: Path):
    with (root / "metrics/data4_in_window_fraction.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["ovary_mean"] = float(r["ovary_mean"])
        r["ovary_sd"] = float(r["ovary_sd"])
        r["in_window_pct"] = float(r["in_window_pct"])
        r["n_volumes"] = int(r["n_volumes"])
    return rows


# --- figure 1: DSC forest (v14 fig 4.8 replacement) ---------------------

def fig_dsc_forest(rows, out_path: Path) -> None:
    per_var = defaultdict(list)
    for r in rows:
        per_var[r["variant"]].append(r["dsc"])
    order = [v for v in VARIANT_ORDER if v in per_var]
    y = np.arange(len(order))
    means = [st.mean(per_var[v]) for v in order]
    sds   = [st.stdev(per_var[v]) if len(per_var[v]) > 1 else 0 for v in order]

    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(order) + 2.2))
    for yi, v, m, s in zip(y, order, means, sds):
        c = COLOUR[v]
        ax.errorbar(m, yi, xerr=s, fmt="o", color=c, ecolor=c,
                    elinewidth=2.2, capsize=4, ms=9, zorder=3)
        ax.text(m, yi + 0.22, f"{m:.3f} ± {s:.3f}  (n={len(per_var[v])})",
                ha="center", fontsize=8.5, color=c)

    ax.axvline(ANCHOR_PUB, ls="--", color="#C44E52", lw=1.5, zorder=1)
    ax.text(ANCHOR_PUB, len(order) - 0.35,
            f" published RAovSeg ({ANCHOR_PUB:.3f})",
            color="#C44E52", fontsize=9, ha="left", va="center")
    ax.axvline(ANCHOR_INHOUSE, ls=":", color="#404040", lw=1.5, zorder=1)
    ax.text(ANCHOR_INHOUSE, -0.75,
            f" in-house anchor ({ANCHOR_INHOUSE:.3f})",
            color="#404040", fontsize=9, ha="left", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[v] for v in order])
    ax.set_xlabel("Ovary DSC (mean ± sd across seeds)")
    ax.set_xlim(-0.02, max(max(means) + max(sds), ANCHOR_PUB) + 0.06)
    ax.set_ylim(-1.2, len(order) - 0.3)
    ax.set_title("RAovSeg augmentation ovary DSC vs baselines")  # no claim in title
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# --- figure 2: per-subject DSC across variants --------------------------

def fig_per_subject_dsc(rows, out_path: Path) -> None:
    per_var_subj = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_var_subj[r["variant"]][r["subject_id"]].append(r["dsc"])

    order = [v for v in VARIANT_ORDER if v in per_var_subj]
    subjects = sorted({s for m in per_var_subj.values() for s in m})
    x = np.arange(len(subjects))
    width = 0.8 / len(order)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, v in enumerate(order):
        means = [st.mean(per_var_subj[v].get(s, [np.nan])) if per_var_subj[v].get(s) else np.nan for s in subjects]
        sds   = [st.stdev(per_var_subj[v].get(s, [0, 0])) if len(per_var_subj[v].get(s, [])) > 1 else 0 for s in subjects]
        offset = (i - (len(order) - 1) / 2) * width
        ax.bar(x + offset, means, width=width * 0.9,
               color=COLOUR[v], alpha=0.9,
               yerr=sds, ecolor="#333", capsize=2,
               label=DISPLAY[v])

    for xi, s in zip(x, subjects):
        if s in UNIV_FAIL:
            ax.axvspan(xi - 0.5, xi + 0.5, color="#FFD1D1", alpha=0.35, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=0)
    ax.set_ylabel("Ovary DSC (mean ± sd across seeds)")
    ax.axhline(ANCHOR_INHOUSE, color="#404040", ls=":", lw=1.3, alpha=0.7,
               label=f"in-house anchor ({ANCHOR_INHOUSE:.3f})")
    ax.axhline(ANCHOR_PUB, color="#C44E52", ls="--", lw=1.3, alpha=0.7,
               label=f"published anchor ({ANCHOR_PUB:.3f})")
    ax.set_title("Per-subject DSC across variants (pink = universal-failure)")
    ax.set_ylim(0, max(0.85, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# --- figure 3: detection rate vs mean DSC scatter -----------------------

def fig_detection_vs_dsc(dsc_rows, det_rows, out_path: Path) -> None:
    per_var_dsc = defaultdict(list)
    for r in dsc_rows:
        per_var_dsc[r["variant"]].append(r["dsc"])
    per_var_det = defaultdict(lambda: {"det": 0, "pairs": 0})
    for r in det_rows:
        if int(r["gt_present"]) != 1: continue
        per_var_det[r["variant"]]["pairs"] += 1
        per_var_det[r["variant"]]["det"] += int(r["detected"])

    fig, ax = plt.subplots(figsize=(8, 6))
    for v in sorted(set(per_var_dsc) & set(per_var_det)):
        dsc = st.mean(per_var_dsc[v])
        det = 100 * per_var_det[v]["det"] / per_var_det[v]["pairs"]
        c = COLOUR.get(v, "#888")
        marker = "s" if v == "raovseg_real_only" else "o"
        ax.scatter(det, dsc, s=110, c=c, marker=marker, edgecolor="k", linewidths=0.8, zorder=3)
        label = v.replace("raov_aug_", "").replace("raovseg_", "")
        ax.annotate(label, (det, dsc), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color=c)

    ax.axhline(ANCHOR_INHOUSE, color="#404040", ls=":", lw=1.3, alpha=0.7,
               label=f"in-house DSC anchor ({ANCHOR_INHOUSE:.3f})")
    ax.axhline(ANCHOR_PUB, color="#C44E52", ls="--", lw=1.3, alpha=0.7,
               label=f"published DSC anchor ({ANCHOR_PUB:.3f})")
    ax.set_xlabel("Detection rate (% of gt-positive volumes)")
    ax.set_ylabel("Mean ovary DSC across seeds")
    ax.set_title("Detection vs delineation — decomposing DSC across variants")
    ax.set_xlim(50, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# --- figure 4: DATA-5 ablation 5-seed bar --------------------------------

def fig_ablation_5seed(rows, out_path: Path) -> None:
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r["ovary_dsc"])

    order = ["full", "no_postprocess", "no_resclass"]
    paper = {"full": 0.290, "no_postprocess": 0.235, "no_resclass": 0.013}
    colours = {"full": "#404040", "no_postprocess": "#DD8452", "no_resclass": "#4C72B0"}

    x = np.arange(len(order))
    means = [st.mean(by_cfg[c]) for c in order]
    sds   = [st.stdev(by_cfg[c]) for c in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    for xi, c, m, s in zip(x, order, means, sds):
        ax.bar(xi - 0.18, m, width=0.34, yerr=s,
               color=colours[c], alpha=0.9, capsize=5, label="in-house (n=5)" if c == "full" else None)
        ax.bar(xi + 0.18, paper[c], width=0.34,
               color=colours[c], alpha=0.4, label="paper (n=1)" if c == "full" else None)
        ax.text(xi - 0.18, m + s + 0.005, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.18, paper[c] + 0.005, f"{paper[c]:.3f}", ha="center", fontsize=9, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Ovary DSC")
    ax.set_title("RAovSeg pipeline ablations — in-house (5 seeds) vs paper (1 seed)")
    ax.set_ylim(0, 0.35)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# --- figure 5: DATA-4 in-window calibration (pre-fix vs post-fix bars) ---

def fig_data4_calibration(rows, out_path: Path) -> None:
    # Group into pairs by base recipe; place pre-fix and post-fix side-by-side.
    pairs = [
        ("exp1c_concat",  "exp1c_concat_fixed",   "exp1c_concat"),
        ("exp1c_spade",   "exp1c_spade_fixed",    "exp1c_spade"),
        ("exp2",          "exp2_fixed",           "exp2 (λ=0.01)"),
        ("exp2_lam05",    "exp2_lam05_fixed",     "exp2 (λ=0.05)"),
        ("exp2_lam50",    "exp2_lam50_fixed",     "exp2 (λ=0.50)"),
    ]
    dial = ["exp1c_spade_t022", "exp1c_spade_t028"]
    dial_labels = ["dial t=0.22", "dial t=0.28"]

    by = {r["variant"]: r for r in rows}
    real_30 = by.get("real_d2")
    real_3 = by.get("real_d2_mech3")

    n_pairs = len(pairs)
    n_dial = len(dial)
    x = np.arange(n_pairs + n_dial + 1)  # +1 spacer between pairs and dials

    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.36

    # Pre/post-fix pairs
    for i, (pre, post, _) in enumerate(pairs):
        pv = by.get(pre, {}).get("in_window_pct")
        po = by.get(post, {}).get("in_window_pct")
        if pv is not None:
            ax.bar(x[i] - width/2, pv, width=width, color="#B0BEC5",
                   edgecolor="k", linewidth=0.5, label="pre-fix" if i == 0 else None)
            ax.text(x[i] - width/2, pv + 1, f"{pv:.1f}", ha="center", fontsize=8.5, color="#333")
        if po is not None:
            ax.bar(x[i] + width/2, po, width=width, color="#4C72B0",
                   edgecolor="k", linewidth=0.5, label="post-fix" if i == 0 else None)
            ax.text(x[i] + width/2, po + 1, f"{po:.1f}", ha="center", fontsize=8.5,
                    color="#333", fontweight="bold")

    # Path-B intensity dial variants (single bars, offset color)
    for j, (dvar, dlabel) in enumerate(zip(dial, dial_labels)):
        pos = x[n_pairs + 1 + j]
        val = by.get(dvar, {}).get("in_window_pct")
        if val is not None:
            ax.bar(pos, val, width=width * 1.6, color="#DD8452",
                   edgecolor="k", linewidth=0.5,
                   label="intensity dial" if j == 0 else None)
            ax.text(pos, val + 1, f"{val:.1f}", ha="center", fontsize=8.5, color="#333")

    # Real reference lines
    if real_30 is not None:
        ax.axhline(real_30["in_window_pct"], color="#404040", ls=":", lw=1.6, alpha=0.85,
                   label=f"real-D2 30-subj ({real_30['in_window_pct']:.2f}%)")
    if real_3 is not None:
        ax.axhline(real_3["in_window_pct"], color="#C44E52", ls="--", lw=1.4, alpha=0.7,
                   label=f"real-D2 mech-3 ({real_3['in_window_pct']:.2f}%)")

    labels = [p[2] for p in pairs] + [""] + dial_labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Ovary voxels in [0.22, 0.30] window (%)")
    ax.set_title("Ovary intensity calibration — pre-fix vs post-fix vs real-D2 reference")
    ax.set_ylim(0, max(60, max(r["in_window_pct"] for r in rows) + 6))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=9, ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


# --- entry point ---------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = load_data1(root)
    d3 = load_data3(root)
    d4 = load_data4(root)
    d5 = load_data5(root)

    fig_dsc_forest(d1, out_dir / "fig_v14_dsc_forest.png")
    fig_per_subject_dsc(d1, out_dir / "fig_v14_per_subject_dsc.png")
    fig_detection_vs_dsc(d1, d3, out_dir / "fig_v14_detection_vs_dsc.png")
    fig_ablation_5seed(d5, out_dir / "fig_v14_ablation_5seed.png")
    fig_data4_calibration(d4, out_dir / "fig_v14_data4_calibration.png")


if __name__ == "__main__":
    main()
