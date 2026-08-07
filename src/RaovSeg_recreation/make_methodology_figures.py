"""
Methodology / appendix schematic figures (no NIfTI data needed):
    fig_raovseg_arch.png   — the RAovSeg segmentation pipeline (ResClass -> AttUSeg -> postproc)
    fig_repro_heat.png     — §3.9 reproduction table, cell-shaded (green = matches, red = not)
    fig_iscs.png           — illustrative ISCS shared-noise ablation (α = 0 / 0.8 / 1)

Run:  python -m src.RaovSeg_recreation.make_methodology_figures --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight",
                     "font.family": "DejaVu Sans", "font.size": 11})
NEU, NEU_BG = "#404040", "#EEEEEE"
GOOD, GOOD_BG = "#2CA02C", "#E3F0E4"
BLU, BLU_BG = "#4C72B0", "#E7EEF6"


def _box(ax, x, y, w, h, text, fc, ec, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                 linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=NEU, lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color, zorder=1))


def fig_raovseg_arch(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set_xlim(0, 132); ax.set_ylim(0, 40); ax.axis("off")
    y, h = 22, 12
    boxes = [
        (2, 18, "T2FS axial\nslice", BLU_BG, BLU),
        (24, 22, "ResClass\nslice classifier\n(ovary present?)", NEU_BG, NEU),
        (52, 22, "Attention U-Net\n(AttUSeg)\nper-slice ovary mask", NEU_BG, NEU),
        (80, 22, "Postprocess\nclosing + largest\nconnected component", NEU_BG, NEU),
        (108, 20, "Ovary DSC\n(8-subject\ntest set)", GOOD_BG, GOOD),
    ]
    xs = []
    for (x, w, t, fc, ec) in boxes:
        _box(ax, x, y, w, h, t, fc, ec, weight="bold" if "DSC" in t else "normal")
        xs.append((x, w))
    _arrow(ax, 20, y + h / 2, 24, y + h / 2)
    _arrow(ax, 46, y + h / 2, 52, y + h / 2); ax.text(49, y + h / 2 + 2.4, "if +", fontsize=8, ha="center", color=NEU)
    _arrow(ax, 74, y + h / 2, 80, y + h / 2)
    _arrow(ax, 102, y + h / 2, 108, y + h / 2)
    ax.text(66, 6.5, "Ablations reported in §3.9:  full = ResClass + AttUSeg + postprocess   ·   "
            "no_pp = skip postprocess   ·   no_rc = AttUSeg on every slice (skip the ResClass gate)",
            ha="center", fontsize=9, color="#333333")
    ax.set_title("RAovSeg — the downstream ovary-segmentation pipeline",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.savefig(out_dir / "fig_raovseg_arch.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_raovseg_arch.png'}")


def fig_repro_heat(out_dir: Path) -> None:
    rows = [("Full pipeline", 0.290, 0.220, -0.070),
            ("No postprocess", 0.235, 0.184, -0.051),
            ("No ResClass", 0.013, 0.095, +0.082)]
    fig, ax = plt.subplots(figsize=(8.2, 3.0)); ax.axis("off")
    cols = ["Ablation", "Paper", "Recreation", "Gap", "Reproduces?"]
    cw = [0.30, 0.15, 0.18, 0.15, 0.22]; x0 = [sum(cw[:i]) for i in range(len(cw))]
    yh = 1.0 / (len(rows) + 1)
    for j, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((x0[j], 1 - yh), cw[j], yh, facecolor=BLU, edgecolor="white"))
        ax.text(x0[j] + cw[j] / 2, 1 - yh / 2, c, ha="center", va="center", color="white", fontweight="bold", fontsize=9.5)
    for i, (name, p, r, g) in enumerate(rows):
        yy = 1 - (i + 2) * yh
        ok = abs(g) <= 0.06
        gap_fc = GOOD_BG if ok else "#F7D5D5"
        gap_txt = "yes (≤ ε)" if ok else "no"
        gap_tc = GOOD if ok else "#C44E52"
        cells = [(name, "#F5F7FA", "#111"), (f"{p:.3f}", "#FFFFFF", "#111"),
                 (f"{r:.3f}", "#FFFFFF", "#111"), (f"{g:+.3f}", gap_fc, "#111"),
                 (gap_txt, gap_fc, gap_tc)]
        for j, (txt, fc, tc) in enumerate(cells):
            ax.add_patch(plt.Rectangle((x0[j], yy), cw[j], yh, facecolor=fc, edgecolor="#DDDDDD"))
            ax.text(x0[j] + cw[j] / 2, yy + yh / 2, txt, ha="center", va="center",
                    color=tc, fontsize=9.5, fontweight="bold" if j >= 3 else "normal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("RAovSeg baseline reproduction (§3.9) — core method reproduces; "
                 "postproc & ResClass-criticality claims do not",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "fig_repro_heat.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_repro_heat.png'}")


def fig_iscs(out_dir: Path) -> None:
    rng = np.random.default_rng(0)
    H = 64
    yy, xx = np.mgrid[0:H, 0:H]
    anat = np.exp(-(((xx - 32) ** 2 + (yy - 30) ** 2) / 240)) * 0.85   # constant across z
    shared = rng.standard_normal((H, H))
    indep = [rng.standard_normal((H, H)) for _ in range(5)]
    NL = 0.6

    def synth(z, alpha):
        noise = alpha * shared + (1 - alpha) * indep[z]
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)
        return np.clip((1 - NL) * anat + NL * noise, 0, 1)

    rows = [("α = 0", "independent → flickers", 0.0),
            ("α = 0.8", "mostly shared → coherent (paper)", 0.8),
            ("α = 1", "identical → over-rigid", 1.0)]
    fig, axes = plt.subplots(3, 5, figsize=(11, 7))
    for r, (_, _, a) in enumerate(rows):
        for z in range(5):
            ax = axes[r, z]
            ax.imshow(synth(z, a), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"z = {z}", fontsize=9.5)
    for (a_lab, desc, _), yc in zip(rows, [0.77, 0.51, 0.25]):
        fig.text(0.015, yc, a_lab, fontsize=11.5, fontweight="bold", rotation=90, va="center", ha="center")
        fig.text(0.042, yc, desc, fontsize=8.5, color="#555555", rotation=90, va="center", ha="center")
    fig.suptitle("Inter-Slice Consistent Stochasticity (ISCS) — illustrative: the shared-noise fraction α "
                 "sets z-coherence\n(schematic, not generator output — anatomy held fixed so only the noise varies)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.06, 0, 1, 0.92])
    fig.savefig(out_dir / "fig_iscs.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_iscs.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_raovseg_arch(args.out_dir)
    fig_repro_heat(args.out_dir)
    fig_iscs(args.out_dir)


if __name__ == "__main__":
    main()
