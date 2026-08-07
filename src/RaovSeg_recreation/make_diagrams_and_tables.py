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
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "savefig.dpi": 120, "savefig.bbox": "tight",   # keeps wide diagrams <= 2000px, crisp
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
    # flatter style: thinner stroke, gentler corners
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.6",
                 linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#404040", style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ----------------------------------------------------------------------
def fig_pipeline(out_path: Path) -> None:
    # Two-row snake so box text stays legible when embedded (~6.4in wide).
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")
    h = 14
    # row 1 (left -> right)
    r1 = [(2, 20, "D2 real T2FS\n30 train", C["real_bg"], C["real"]),
          (26, 22, "Conditional DDPM\ngenerator\n(concat/SPADE ±PatchGAN)", C["gen_bg"], C["gen"]),
          (52, 20, "Synthetic\nlabelled volumes\n(6-channel)", C["synth_bg"], C["synth"]),
          (76, 20, "Preprocessing\nalignment (v1→v2→v3)", C["neutral_bg"], C["neutral"])]
    # row 2 (right -> left)
    r2 = [(76, 20, "Training pool\nreal + synth", C["neutral_bg"], C["neutral"]),
          (50, 22, "RAovSeg\nsegmenter", C["neutral_bg"], C["neutral"]),
          (26, 20, "Ovary DSC\nsacred 8-test", C["good_bg"], C["good"])]
    y1, y2 = 40, 12
    for x, w, t, fc, ec in r1:
        _box(ax, x, y1, w, h, t, fc, ec, fontsize=9.5,
             weight="bold" if "generator" in t else "normal")
    for i in range(len(r1) - 1):
        _arrow(ax, r1[i][0] + r1[i][1], y1 + h / 2, r1[i + 1][0], y1 + h / 2)
    # elbow down from preprocessing (top-right) to training pool (bottom-right)
    _arrow(ax, 86, y1, 86, y2 + h)
    for x, w, t, fc, ec in r2:
        _box(ax, x, y2, w, h, t, fc, ec, fontsize=9.5,
             weight="bold" if "DSC" in t else "normal")
    for i in range(len(r2) - 1):
        _arrow(ax, r2[i][0], y2 + h / 2, r2[i + 1][0] + r2[i + 1][1], y2 + h / 2)  # points left
    ax.text(50, 3.0, "Baseline: real-only pool (no synth) → RAovSeg → DSC = 0.290",
            ha="center", fontsize=9.5, color=C["neutral"])
    ax.set_title("Synthetic augmentation pipeline: generator → RAovSeg → ovary DSC",
                 fontsize=13, fontweight="bold", y=1.0)
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

    # ===== top: concat vs SPADE — one level deeper, inside the U-Net =====
    BW, BH = 8, 5

    def draw_unet(ox):
        b = {"e1": (ox + 4, 78), "e2": (ox + 13, 72), "b": (ox + 22, 66),
             "d2": (ox + 33, 72), "d1": (ox + 42, 78)}
        for x, y in b.values():
            _box(ax, x, y, BW, BH, "", "#F4F4F6", "#8A8A8A", 8)
        seq = ["e1", "e2", "b", "d2", "d1"]
        for a, c in zip(seq, seq[1:]):
            (xa, ya), (xc, yc) = b[a], b[c]
            _arrow(ax, xa + BW, ya + BH / 2, xc, yc + BH / 2, color="#777777", lw=1.0)
        for a, c in [("e1", "d1"), ("e2", "d2")]:
            (xa, ya), (xc, yc) = b[a], b[c]
            ax.plot([xa + BW / 2, xc + BW / 2], [ya + BH, yc + BH], ls=":", color="#B0B0B0", lw=1.0)
        ax.text(b["e1"][0] + BW / 2, b["e1"][1] - 2.4, "encoder", ha="center", fontsize=6.5, color="#999999")
        ax.text(b["d1"][0] + BW / 2, b["d1"][1] + BH + 1, "decoder", ha="center", fontsize=6.5, color="#999999")
        return b

    # -- concat (left) --
    ax.text(26, 96, "concat — label concatenated at the input only",
            ha="center", fontsize=11, fontweight="bold", color=C["gen"])
    _box(ax, 4, 88, 44, 6, "noisy xₜ  ⊕  6-ch label", C["gen_bg"], C["gen"], 9, weight="bold")
    bc = draw_unet(2)
    _arrow(ax, 26, 88, bc["e1"][0] + BW / 2, bc["e1"][1] + BH, color=C["gen"], lw=1.4)
    ax.text(24, 57, "Label enters only at the input → global influence (CLR ≈ 0.08).",
            ha="center", fontsize=8.5, color="#333333")

    # -- SPADE (right) --
    ax.text(102, 96, "SPADE — label injected at every decoder block",
            ha="center", fontsize=11, fontweight="bold", color=C["real"])
    _box(ax, 80, 88, 20, 6, "noisy xₜ", C["neutral_bg"], C["neutral"], 9)
    bs = draw_unet(76)
    _arrow(ax, 90, 88, bs["e1"][0] + BW / 2, bs["e1"][1] + BH, color=C["neutral"], lw=1.4)
    _box(ax, 99, 60.5, 26, 5, "6-ch label → resize", C["real_bg"], C["real"], 8.5, weight="bold")
    for k in ("b", "d2", "d1"):                       # SPADE at each decoder resolution
        x, y = bs[k]
        ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.6",
                     linewidth=1.2, edgecolor=C["real"], facecolor=C["real_bg"], zorder=4))
        ax.text(x + BW / 2, y + BH / 2, "SPADE", ha="center", va="center",
                fontsize=6, fontweight="bold", color=C["real"], zorder=5)
        _arrow(ax, 110, 65.5, x + BW / 2, y, color=C["real"], lw=1.0)
    ax.text(106, 57, "Label injected via SPADE at each decoder resolution (CLR ≈ 0.42).",
            ha="center", fontsize=8.5, color="#333333")

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


def _pick_ovary_slice(mask: np.ndarray) -> int:
    """Pick a representative z-slice with maximum ovary voxel count.
    Falls back to the middle of the volume if no ovary voxels present."""
    if mask.sum() == 0:
        return mask.shape[0] // 2
    per_slice = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(per_slice))


def _load_subj_arrays(test_dir: Path, pred_dir: Path, subj: str,
                      label_file: str = "label_ov.npy"
                      ) -> tuple | None:
    """Return (image, gt_mask, pred_mask) for a subject or None if missing."""
    subj_dir = test_dir / subj
    img_p = subj_dir / "image.npy"
    lbl_p = subj_dir / label_file
    pred_p = pred_dir / f"{subj}_pred.npy"
    if not (img_p.exists() and lbl_p.exists() and pred_p.exists()):
        return None
    try:
        img  = np.load(img_p)
        gt   = np.load(lbl_p).astype(np.uint8)
        pred = np.load(pred_p).astype(np.uint8)
    except Exception as e:
        print(f"  load failed for {subj}: {type(e).__name__}: {e}")
        return None
    return img, gt, pred


def _dsc_records_from_json(path: Path, mode: str = "full"
                           ) -> list[tuple[str, float]] | None:
    """Read (subject, dsc) tuples from an evaluate.py metrics JSON, if present."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            j = json.load(f)
    except Exception:
        return None
    recs = j.get("per_subject", {}).get(mode, [])
    out = []
    for r in recs:
        s = r.get("subject") or r.get("subject_id")
        d = r.get("dsc")
        if s is not None and d is not None:
            try:
                out.append((str(s), float(d)))
            except (TypeError, ValueError):
                continue
    return out or None


def fig_per_subject_dsc(out_dir: Path,
                        test_dir: Path | None = None,
                        pred_dir: Path | None = None,
                        metrics_json: Path | None = None,
                        label_file: str = "label_ov.npy") -> None:
    """Per-subject DSC waterfall on the 8-subject test set.

    Behaviour depends on which optional inputs are provided:
      - No inputs → the original bar-only figure with hard-coded recreation
        baseline DSCs (backward-compatible default).
      - test_dir + pred_dir provided → augmented layout with two thumbnail
        columns per subject: middle = ground-truth ovary contour on the
        real image; right = predicted ovary contour on the same image.
        DSCs read from `metrics_json` if given, else fall back to hard-
        coded values.
    """
    from matplotlib import cm
    # --- Get (subject, DSC) list ---
    hard_coded = [("D2-016", 0.479), ("D2-017", 0.438), ("D2-024", 0.267),
                  ("D2-015", 0.220), ("D2-026", 0.193), ("D2-023", 0.062),
                  ("D2-005", 0.003), ("D2-038", 0.000)]
    if metrics_json is not None:
        loaded = _dsc_records_from_json(metrics_json)
        if loaded:
            pairs = loaded
            print(f"[per_subject_dsc] loaded {len(pairs)} DSCs from {metrics_json}")
        else:
            print(f"[per_subject_dsc] {metrics_json} unreadable/empty — using hard-coded")
            pairs = hard_coded
    else:
        pairs = hard_coded

    # Sort ascending so best sits at the top of the horizontal bars
    pairs.sort(key=lambda t: t[1])
    subs = [p[0] for p in pairs]
    dsc  = [p[1] for p in pairs]

    norm = plt.Normalize(0.0, max(dsc) if max(dsc) > 0 else 1.0)
    cols = cm.viridis(norm(dsc))
    y = np.arange(len(subs))

    # --- Figure layout depends on whether we have image data ---
    with_thumbs = test_dir is not None and pred_dir is not None
    if with_thumbs:
        # 3 columns: bar chart | GT overlay column | pred overlay column
        # Each of the two thumbnail columns has one axis per subject stacked
        # vertically. Use gridspec so the bar chart is the fat left column.
        n = len(subs)
        fig = plt.figure(figsize=(11.5, 0.72 * n + 1.6))
        gs = fig.add_gridspec(n, 6, width_ratios=[3.2, 0.15, 1, 0.05, 1, 0.05],
                              wspace=0.05, hspace=0.05)
        ax_bar = fig.add_subplot(gs[:, 0])
    else:
        fig, ax_bar = plt.subplots(figsize=(8.6, 4.6))

    # --- Bar chart (always) ---
    ax_bar.barh(y, dsc, color=cols, edgecolor="#333333", linewidth=0.5, zorder=3)
    ax_bar.set_yticks(y); ax_bar.set_yticklabels(subs)
    ax_bar.set_xlabel("Ovary DSC"); ax_bar.set_xlim(0, max(0.56, max(dsc) * 1.15))
    ax_bar.set_title("Per-subject DSC on the 8-subject test set"
                     + ("  —  GT (red) and prediction (blue)" if with_thumbs else ""))
    m = float(np.mean(dsc))
    ax_bar.axvline(m, ls="--", color="#404040", lw=1.4, zorder=4)
    ax_bar.text(m + 0.004, len(subs) - 0.4, f"mean = {m:.3f}",
                color="#404040", fontsize=9, va="center")
    for yi, v in zip(y, dsc):
        ax_bar.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=8.5)
    for name in ("D2-005", "D2-023"):
        if name in subs:
            lbl = ax_bar.get_yticklabels()[subs.index(name)]
            lbl.set_color("#C44E52"); lbl.set_fontweight("bold")
    ax_bar.text(0.98, 0.06, "D2-005 & D2-023: persistent failures (≈ 0 across seeds)",
                transform=ax_bar.transAxes, ha="right", fontsize=8.5, color="#C44E52")
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_bar, pad=0.02)
    cb.set_label("DSC", fontsize=9)

    # --- Thumbnails (only if inputs provided) ---
    if with_thumbs:
        # y goes bottom→top in barh; reverse so worst subject sits at bottom
        for row_idx, subj in enumerate(subs[::-1]):
            arr = _load_subj_arrays(test_dir, pred_dir, subj, label_file)
            ax_gt   = fig.add_subplot(gs[row_idx, 2])
            ax_pred = fig.add_subplot(gs[row_idx, 4])
            for _a in (ax_gt, ax_pred):
                _a.set_xticks([]); _a.set_yticks([])
                for _s in _a.spines.values():
                    _s.set_visible(False)
            if arr is None:
                for _a in (ax_gt, ax_pred):
                    _a.text(0.5, 0.5, "n/a", ha="center", va="center",
                            transform=_a.transAxes, color="#888")
                continue
            img, gt, pred = arr
            z = _pick_ovary_slice(gt if gt.sum() > 0 else pred)
            ax_gt.imshow(img[z], cmap="gray", vmin=0, vmax=1)
            if gt[z].any():
                ax_gt.contour(gt[z], levels=[0.5], colors="#C44E52", linewidths=1.0)
            ax_pred.imshow(img[z], cmap="gray", vmin=0, vmax=1)
            if pred[z].any():
                ax_pred.contour(pred[z], levels=[0.5], colors="#4C72B0", linewidths=1.0)
            if row_idx == 0:
                ax_gt.set_title("GT",   fontsize=9, color="#C44E52", fontweight="bold")
                ax_pred.set_title("pred", fontsize=9, color="#4C72B0", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_per_subject_dsc.png"); plt.close(fig)
    print(f"[saved] {out_dir / 'fig_per_subject_dsc.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--per-subject-test-dir", type=Path, default=None,
                    help="Optional: path to a RAovSeg test dir with image.npy + "
                         "label_ov.npy per subject. Enables GT thumbnail column "
                         "on fig_per_subject_dsc.")
    ap.add_argument("--per-subject-pred-dir", type=Path, default=None,
                    help="Optional: path to a RAovSeg prediction dir with "
                         "<subject>_pred.npy files. Enables prediction "
                         "thumbnail column on fig_per_subject_dsc.")
    ap.add_argument("--per-subject-metrics-json", type=Path, default=None,
                    help="Optional: path to a per-subject metrics_ov.json to "
                         "load DSCs from. Falls back to hard-coded if omitted.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_pipeline(args.out_dir / "fig_pipeline.png")
    fig_ablation_2x2(args.out_dir / "fig_ablation_2x2.png")
    fig_conditioning(args.out_dir / "fig_conditioning_schematic.png")
    fig_per_subject_dsc(args.out_dir,
                        test_dir=args.per_subject_test_dir,
                        pred_dir=args.per_subject_pred_dir,
                        metrics_json=args.per_subject_metrics_json)
    table_dataset(args.out_dir)
    table_results(args.out_dir)


if __name__ == "__main__":
    main()
