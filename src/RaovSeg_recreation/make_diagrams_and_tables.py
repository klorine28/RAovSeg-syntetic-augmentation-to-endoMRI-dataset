"""
Dissertation orientation diagrams + summary tables.

No data dependencies for the diagrams; the tables read canonical numbers
from the dissertation drafts and metrics/master_metrics.csv.

Outputs (to --out-dir, default figures/):
    fig_pipeline.png                 — generator -> RAovSeg -> DSC dataflow
    fig_ablation_2x2.png             — concat/SPADE x +/-PatchGAN matrix
    fig_conditioning_schematic.png   — how the label enters (concat vs SPADE)
    table_dataset_summary.png/.csv   — UT-EndoMRI cohort + filtering
    table_master_results.png/.csv    — every config x DSC x delta vs baseline

Run:  python -m src.RaovSeg_recreation.make_diagrams_and_tables --out-dir figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
})

C = {
    "real":   "#4C72B0", "real_bg":  "#E7EEF6",
    "gen":    "#C44E52", "gen_bg":   "#F7E4E5",
    "synth":  "#8172B3", "synth_bg": "#ECE8F3",
    "neutral":"#404040", "neutral_bg":"#EEEEEE",
    "good_bg":"#E3F0E4", "good": "#2CA02C",
    "band":   "#2CA02C",
}
BASELINE = 0.290


def _box(ax, x, y, w, h, text, fc, ec, fontsize=10, weight="normal", tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=1.2",
                 linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#404040", style="-|>", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=16, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ----------------------------------------------------------------------
def fig_pipeline(out_path: Path) -> None:
    # Single straight left-to-right row (no curved arrows).
    fig, ax = plt.subplots(figsize=(16, 3.6))
    ax.set_xlim(0, 134); ax.set_ylim(0, 34); ax.axis("off")
    y, h, w, step = 12, 16, 15, 19
    boxes = [
        ("D2 real T2FS\n30 train", C["real_bg"], C["real"]),
        ("Conditional DDPM\ngenerator\n(concat/SPADE\n±PatchGAN)", C["gen_bg"], C["gen"]),
        ("Synthetic\nlabelled volumes\n(6-channel)", C["synth_bg"], C["synth"]),
        ("Preprocessing\nalignment\n(v1→v2→v3)", C["neutral_bg"], C["neutral"]),
        ("Training pool\nreal + synth", C["neutral_bg"], C["neutral"]),
        ("RAovSeg\nsegmenter", C["neutral_bg"], C["neutral"]),
        ("Ovary DSC\nsacred 8-test", C["good_bg"], C["good"]),
    ]
    xs = []
    for i, (txt, fc, ec) in enumerate(boxes):
        x = 2 + i * step
        _box(ax, x, y, w, h, txt, fc, ec, fontsize=8.5,
             weight="bold" if ("generator" in txt or "DSC" in txt) else "normal")
        xs.append(x)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    ax.text(67, 3.0, "Baseline: real-only pool (no synth) → RAovSeg → DSC = 0.290",
            ha="center", fontsize=9, color=C["neutral"])
    ax.set_title("Synthetic augmentation pipeline: generator → RAovSeg → ovary DSC",
                 fontsize=13, fontweight="bold", pad=6)
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def fig_ablation_2x2(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # cells: (col, row) -> content. row0=concat(top), row1=SPADE(bottom)
    cells = {
        (0, 1): ("Exp 1a", "concat, −PatchGAN", "CLR ≈ 0.03", "locked out", C["gen_bg"], C["gen"]),
        (1, 1): ("Exp 1c-concat", "concat, +PatchGAN", "CLR ≈ 0.07", "DSC 0.053", C["gen_bg"], C["gen"]),
        (0, 0): ("Exp 1b", "SPADE, −PatchGAN", "CLR ≈ 0.47", "localises", C["real_bg"], C["real"]),
        (1, 0): ("Exp 1c-spade", "SPADE, +PatchGAN", "CLR ≈ 0.40", "DSC 0.178 (best)", C["real_bg"], C["real"]),
    }
    x0, y0, cw, ch, gap = 18, 14, 36, 34, 4
    for (col, row), (eid, cond, clr, note, fc, ec) in cells.items():
        x = x0 + col * (cw + gap); y = y0 + row * (ch + gap)
        _box(ax, x, y, cw, ch, "", fc, ec)
        ax.text(x + cw / 2, y + ch - 7, eid, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec)
        ax.text(x + cw / 2, y + ch - 15, cond, ha="center", va="center", fontsize=9.5)
        ax.text(x + cw / 2, y + 12, clr, ha="center", va="center", fontsize=9.5)
        ax.text(x + cw / 2, y + 5.5, note, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=ec)

    # axis labels
    ax.text(x0 + cw / 2, y0 + 2 * ch + gap + 5, "− PatchGAN", ha="center", fontsize=11, fontweight="bold")
    ax.text(x0 + cw + gap + cw / 2, y0 + 2 * ch + gap + 5, "+ PatchGAN", ha="center", fontsize=11, fontweight="bold")
    ax.text(x0 - 6, y0 + ch + gap + ch / 2, "concat", ha="center", va="center",
            rotation=90, fontsize=11, fontweight="bold", color=C["gen"])
    ax.text(x0 - 6, y0 + ch / 2, "SPADE", ha="center", va="center",
            rotation=90, fontsize=11, fontweight="bold", color=C["real"])
    ax.text(x0 - 13, y0 + ch + gap, "conditioning", ha="center", va="center",
            rotation=90, fontsize=10.5, color="#555555")
    ax.text(x0 + cw + gap / 2, y0 + 2 * ch + gap + 11, "adversarial regulariser",
            ha="center", fontsize=10.5, color="#555555")

    ax.set_title("The 2×2 conditional-DDPM ablation", fontsize=13.5, fontweight="bold", y=1.02)
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def fig_conditioning(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 140); ax.set_ylim(0, 100); ax.axis("off")

    # ===== top: concat vs SPADE conditioning contrast =====
    # -- concat (left) --
    ax.text(33, 97, "concat — label at the input only  (global · CLR ≈ 0.03)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["gen"])
    _box(ax, 4, 82, 18, 8, "noisy image xₜ", C["neutral_bg"], C["neutral"], 9)
    _box(ax, 4, 70, 18, 8, "6-ch label", C["gen_bg"], C["gen"], 9)
    _box(ax, 30, 75, 12, 9, "concat", C["gen_bg"], C["gen"], 9, weight="bold")
    _box(ax, 48, 71, 14, 17, "U-Net", C["neutral_bg"], C["neutral"], 10, weight="bold")
    _arrow(ax, 22, 86, 30, 82, color=C["neutral"])
    _arrow(ax, 22, 74, 30, 78, color=C["gen"])
    _arrow(ax, 42, 79, 48, 79, color=C["neutral"])

    # -- SPADE (right) --
    ax.text(107, 97, "SPADE — label at every block  (per-organ · CLR ≈ 0.4)",
            ha="center", fontsize=10.5, fontweight="bold", color=C["real"])
    _box(ax, 78, 82, 18, 8, "noisy image xₜ", C["neutral_bg"], C["neutral"], 9)
    _box(ax, 108, 71, 14, 17, "U-Net", C["neutral_bg"], C["neutral"], 10, weight="bold")
    _box(ax, 78, 60, 18, 8, "6-ch label", C["real_bg"], C["real"], 9)
    _arrow(ax, 96, 86, 108, 86, color=C["neutral"])
    _arrow(ax, 96, 64, 110, 71, color=C["real"], lw=1.5)
    for yb in (74, 80, 86):
        _box(ax, 126, yb - 2.5, 10, 5, "SPADE", C["real_bg"], C["real"], 7.5, weight="bold")
        _arrow(ax, 122, yb, 126, yb, color=C["real"], lw=1.3)

    # divider
    ax.plot([4, 136], [55, 55], color="#CCCCCC", lw=1.2)
    ax.text(70, 50.5, "Both conditioning variants then follow the same DDPM training + sampling:",
            ha="center", fontsize=10.5, fontweight="bold", color="#333")

    # ===== bottom: shared downstream =====
    ax.text(43, 44, "TRAINING", ha="center", fontsize=9, fontweight="bold", color="#999")
    ax.text(115, 44, "SAMPLING (inference)", ha="center", fontsize=9, fontweight="bold", color="#999")
    y, h = 27, 13
    _box(ax, 2, y, 20, h, "U-Net predicts\nnoise ε̂", C["neutral_bg"], C["neutral"], 9)
    _box(ax, 28, y, 26, h, "DDPM loss\nMSE(ε̂, ε)\n+ λ·PatchGAN (1c)", C["gen_bg"], C["gen"], 8.5)
    _box(ax, 60, y, 24, h, "EMA weight update\n(decay 0.9999)", C["synth_bg"], C["synth"], 9)
    _box(ax, 90, y, 24, h, "Sampling loop\nreverse ×100 · CFG", C["synth_bg"], C["synth"], 9)
    _box(ax, 120, y, 18, h, "Synthetic\nlabelled volume", C["good_bg"], C["good"], 9, weight="bold")
    for x0, x1 in [(22, 28), (54, 60), (84, 90), (114, 120)]:
        _arrow(ax, x0, y + h / 2, x1, y + h / 2)
    ax.text(72, 22, "EMA weights are the ones used at sampling time",
            ha="center", fontsize=8.5, style="italic", color="#666")

    fig.suptitle("The conditional generator: conditioning → U-Net → training & sampling",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def _render_table(rows, col_labels, title, out_png, col_widths=None,
                  header_color="#4C72B0", figsize=(10, 3.5), fontsize=10):
    fig, ax = plt.subplots(figsize=figsize); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(header_color); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F7FA")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.savefig(out_png); plt.close(fig); print(f"[saved] {out_png}")


def _write_csv(rows, header, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[saved] {out_csv}")


def table_dataset(out_dir: Path) -> None:
    header = ["Cohort", "Site", "Sequence", "Raw subjects", "Role", "Fat suppression"]
    rows = [
        ["D2_TCPW", "TCPW", "T2FS", "~73", "Generator + downstream (primary)", "Yes (dark fat)"],
        ["D1_MHS", "Memorial Hermann", "T2", "51", "Phase 2 generator training only", "No (bright fat)"],
    ]
    _write_csv(rows, header, out_dir / "table_dataset_summary.csv")
    _render_table(rows, header, "UT-EndoMRI cohorts (Liang et al., 2025)",
                  out_dir / "table_dataset_summary.png",
                  col_widths=[0.12, 0.18, 0.12, 0.13, 0.32, 0.18], figsize=(11, 2.2))

    # filtering funnel (D2)
    header2 = ["Filter", "Removed", "Remaining"]
    rows2 = [
        ["Raw D2 subjects", "—", "~73"],
        ["Missing T2FS image", "3", "~70"],
        ["Missing uterus/ovary mask", "~9", "~61"],
        ["Sacred RAovSeg test set (held out)", "8", "—"],
        ["Generator training pool", "—", "32"],
        ["RAovSeg train_val (excl. em/cyst)", "—", "30 (+8 test)"],
    ]
    _write_csv(rows2, header2, out_dir / "table_d2_filtering.csv")
    _render_table(rows2, header2, "D2 subject filtering funnel",
                  out_dir / "table_d2_filtering.png",
                  col_widths=[0.55, 0.2, 0.25], figsize=(8.5, 2.9))


def table_results(out_dir: Path) -> None:
    def d(v): return f"{(v - BASELINE) / BASELINE * 100:+.0f}%"
    header = ["Phase", "Configuration", "Conditioning", "Ovary DSC", "Δ vs baseline"]
    rows = [
        ["—", "Real-only baseline", "—", "0.290", "0%"],
        ["1", "v1 (no fixes)", "concat", "0.150", d(0.150)],
        ["1", "v2 (3 fixes)", "concat", "0.044", d(0.044)],
        ["1", "v3 (+Path B)", "concat", "0.053", d(0.053)],
        ["1", "v1 (no fixes)", "SPADE", "0.138", d(0.138)],
        ["1", "v2 (3 fixes)", "SPADE", "0.169", d(0.169)],
        ["1", "v3 (+Path B, n=8)", "SPADE", "0.178", d(0.178)],
        ["2", "exp2 (λ=0.01)", "SPADE cross-domain", "0.020", d(0.020)],
        ["2", "exp2 Path C", "SPADE cross-domain", "0.152", d(0.152)],
        ["2", "exp2_lam05 (λ=0.05)", "SPADE cross-domain", "0.117", d(0.117)],
    ]
    _write_csv(rows, header, out_dir / "table_master_results.csv")
    _render_table(rows, header, "Downstream ovary DSC — all configurations",
                  out_dir / "table_master_results.png",
                  col_widths=[0.08, 0.26, 0.28, 0.16, 0.22], figsize=(10.5, 4.2), fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_pipeline(args.out_dir / "fig_pipeline.png")
    fig_ablation_2x2(args.out_dir / "fig_ablation_2x2.png")
    fig_conditioning(args.out_dir / "fig_conditioning_schematic.png")
    table_dataset(args.out_dir)
    table_results(args.out_dir)


if __name__ == "__main__":
    main()
