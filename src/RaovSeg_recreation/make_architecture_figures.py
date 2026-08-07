"""
Background + architecture diagrams for the dissertation (no data needed):

    fig_ddpm_process.png       — forward noising / reverse denoising schematic
    fig_architecture.png       — full two-phase system architecture
    fig_label_channels.png     — the 6-channel one-hot label design
    fig_raovseg_pipeline.png   — RAovSeg 4-stage runtime pipeline (§1.1 of
                                 OVARY_INTENSITY_ISSUE.md)

Run:  python -m src.RaovSeg_recreation.make_architecture_figures --out-dir figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle

plt.rcParams.update({
    "savefig.dpi": 130, "savefig.bbox": "tight",   # keeps wide diagrams <= 2000px, crisp
    "font.family": "DejaVu Sans", "font.size": 11,
})

C = {
    "real": "#4C72B0", "real_bg": "#E7EEF6",
    "gen": "#C44E52", "gen_bg": "#F7E4E5",
    "synth": "#8172B3", "synth_bg": "#ECE8F3",
    "neutral": "#404040", "neutral_bg": "#EEEEEE",
    "good": "#2CA02C", "good_bg": "#E3F0E4",
    "amber": "#DD8452", "amber_bg": "#FBEDE3",
}


def _box(ax, x, y, w, h, text, fc, ec, fs=10, weight="normal", tc="#111"):
    # flatter style: thinner stroke, gentler corners
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.6",
                 linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#404040", style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ----------------------------------------------------------------------
def fig_ddpm(out_path: Path) -> None:
    """DDPM schematic with 3 rows:
        [1] forward diffusion  x_0  → x_T   (clean → noise, deterministic)
        [2] reverse denoising  x_T  → x_0   (noise → clean, stochastic)
        [3] "3 draws from same x_T"  — visually demonstrates the probabilistic
             nature: same seed at step T, different intermediate noise samples
             during denoising, three distinct x_0 outcomes.
    Row 3 is the answer to "how does the probabilistic nature reflect
    visually" — three different samples branching from one shared x_T.
    """
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:64, 0:64]
    # Three subtly-different "target" images to represent alternative x_0
    # samples that a stochastic denoiser could reach from the same x_T.
    def make_target(cx1, cy1, cx2, cy2, w1=200.0, w2=70.0):
        b = (np.exp(-(((xx - cx1) ** 2 + (yy - cy1) ** 2) / w1)) * 0.9
             + np.exp(-(((xx - cx2) ** 2 + (yy - cy2) ** 2) / w2)) * 0.6)
        return b / b.max()
    bases = [make_target(30, 26, 42, 42),
             make_target(28, 24, 44, 40, 220, 80),
             make_target(32, 28, 40, 44, 180, 60)]
    shared_noise = rng.standard_normal((64, 64)) * 0.5 + 0.5   # SAME x_T for row 3

    def mix(base, f, noise=None):
        n = noise if noise is not None else shared_noise
        return np.clip((1 - f) * base + f * n, 0, 1)

    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    xs = [0.085, 0.255, 0.425, 0.595, 0.765]
    wimg, himg = 0.11, 0.19
    # Reduce vertical extents so all three rows + captions fit
    yF, yR, yS = 0.72, 0.42, 0.10

    fig = plt.figure(figsize=(12.5, 8.5))
    ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off")
    ov.set_xlim(0, 1); ov.set_ylim(0, 1)

    def draw_row(y, fvals, labels, ec, base_img=None, per_step_noise=False):
        b = base_img if base_img is not None else bases[0]
        centers = []
        for i, f in enumerate(fvals):
            a = fig.add_axes([xs[i], y, wimg, himg])
            n = None
            if per_step_noise:
                # Each column gets its own noise draw so the row shows
                # the stochastic branching visibly step-by-step.
                n = rng.standard_normal((64, 64)) * 0.5 + 0.5
            a.imshow(mix(b, f, n), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(ec); s.set_linewidth(1.6)
            if labels[i]:
                ov.text(xs[i] + wimg / 2, y - 0.028, labels[i], ha="center",
                        va="top", fontsize=10.5)
            centers.append((xs[i] + wimg / 2, y + himg / 2,
                            xs[i], xs[i] + wimg))
        return centers

    fwd = draw_row(yF, steps, [r"$x_0$", r"$x_1$", r"$x_t$", r"", r"$x_T$"], "#888",
                   base_img=bases[0])
    rev = draw_row(yR, steps, [r"$x_0$", r"", r"", r"", r"$x_T$"], C["real"],
                   base_img=bases[0])

    # --- Row 3: three stochastic denoising trajectories from ONE shared x_T ---
    # Each row's x_T is IDENTICAL (shared_noise); each x_0 is a different sample.
    # The bases[k] differ subtly so the three "final images" have visually
    # distinct anatomy, illustrating the p_θ(·|x_T) distribution.
    stoch_y = [yS + 0.16, yS + 0.08, yS]
    stoch_centers = []
    for k, ry in enumerate(stoch_y):
        row = []
        for i, f in enumerate(steps):
            a = fig.add_axes([xs[i], ry, wimg * 0.7, himg * 0.5])
            # At i == last (x_T), all three see the SAME image; otherwise each
            # sample uses its own base to demonstrate divergent trajectories.
            b = bases[k]
            a.imshow(mix(b, f), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(C["real"]); s.set_linewidth(1.0)
            row.append((xs[i] + wimg * 0.35, ry + himg * 0.25,
                        xs[i], xs[i] + wimg * 0.7))
        stoch_centers.append(row)

    yc_f = yF + himg / 2
    yc_r = yR + himg / 2
    for i in range(len(xs) - 1):
        _arrow(ov, fwd[i][3], yc_f, fwd[i + 1][2], yc_f, color=C["gen"])
        _arrow(ov, rev[i + 1][2], yc_r, rev[i][3], yc_r, color=C["real"])

    # Vertical link from forward x_T down to reverse x_T (right side)
    xcol = xs[-1] + wimg + 0.010
    _arrow(ov, xcol, yF + 0.02, xcol, yR + himg, color=C["neutral"], lw=1.6)

    # --- Row-3 branching arrows: shared x_T on right fans out to 3 x_0's ---
    x_T_shared = (xs[-1] + wimg * 0.7 / 2, yR - 0.03)
    ov.text(x_T_shared[0], x_T_shared[1] + 0.008, "shared\n$x_T$",
            ha="center", va="bottom", fontsize=8, color=C["real"], fontweight="bold")
    for k in range(3):
        x0_x, x0_y = stoch_centers[k][0][0], stoch_centers[k][0][1]
        xt_x, xt_y = stoch_centers[k][-1][0], stoch_centers[k][-1][1]
        # From shared x_T (single point) to each row's x_T-position
        _arrow(ov, x_T_shared[0], x_T_shared[1] - 0.005, xt_x, xt_y + 0.01,
               color="#B0B0B0", lw=0.9)
        # Left-pointing arrow at each row's own denoising trail (visual guide)
        _arrow(ov, xt_x - 0.005, xt_y, x0_x + 0.02, x0_y,
               color=C["real"], lw=0.9)

    # --- Titles + captions ---
    ov.text(0.5, 0.965, "The denoising diffusion process (DDPM)",
            ha="center", fontsize=14, fontweight="bold")
    ov.text(0.5, yF + himg + 0.030,
            "Forward diffusion  q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1−βₜ)·xₜ₋₁, βₜI)   "
            "— fixed schedule, no learned parameters, T = 1000 steps",
            ha="center", fontsize=10.5, color=C["gen"], fontweight="bold")
    ov.text(0.5, yR + himg + 0.040,
            "Reverse denoising  p_θ(xₜ₋₁ | xₜ) = 𝒩(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))   "
            "— a 2D U-Net predicts ε̂; each step draws xₜ₋₁ from a Gaussian",
            ha="center", fontsize=10.5, color=C["real"], fontweight="bold")

    # Row 3 explanation
    ov.text(0.5, stoch_y[0] + himg * 0.5 + 0.028,
            "Same $x_T$, three stochastic draws → three different $x_0$",
            ha="center", fontsize=10.5, color=C["real"], fontweight="bold")
    ov.text(0.5, stoch_y[-1] - 0.028,
            "Because each denoising step samples from a Gaussian, the same "
            "starting noise vector can (and does) produce distinct final "
            "images. Diversity comes from the sampling stochasticity, not "
            "from a different starting point.",
            ha="center", fontsize=9.2, color="#333", style="italic")
    ov.text(0.5, 0.020,
            "U-Net conditioned on the 6-channel label + timestep t · CFG · EMA · "
            "100 inference steps (DDIM)",
            ha="center", fontsize=9.5, color="#333")
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def fig_architecture(out_path: Path) -> None:
    # Two identical pipelines (one per phase), stacked and divided, to show
    # they are distinct sequential runs of the SAME architecture (not parallel).
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 134); ax.set_ylim(0, 100); ax.axis("off")

    xs = [(2, 22), (26, 24), (52, 18), (72, 16), (90, 18), (110, 22)]

    def panel(y, data_txt, data_fc, data_ec, synth_txt, phase_title, title_color):
        cells = [
            (data_txt, data_fc, data_ec, False),
            ("Conditional DDPM generator\n2D U-Net (concat / SPADE)\n+ PatchGAN discriminator",
             C["gen_bg"], C["gen"], True),
            ("Conditional sampling\n100 · CFG · EMA", C["synth_bg"], C["synth"], False),
            (synth_txt, C["synth_bg"], C["synth"], False),
            ("RAovSeg pool\nreal + synth", C["neutral_bg"], C["neutral"], False),
            ("Ovary DSC\n(sacred 8-test)", C["good_bg"], C["good"], True),
        ]
        ax.text(2, y + 16, phase_title, ha="left", fontsize=11, fontweight="bold", color=title_color)
        prev = None
        for (x, w), (txt, fc, ec, bold) in zip(xs, cells):
            _box(ax, x, y, w, 13, txt, fc, ec, 8.2, weight="bold" if bold else "normal")
            if prev is not None:
                _arrow(ax, prev, y + 6.5, x, y + 6.5)
            prev = x + w

    panel(72, "D2 T2FS + 6-ch label\n(generator ← D2,\ndiscriminator ← D2)",
          C["real_bg"], C["real"], "Synthetic\nD2 volumes",
          "PHASE 1 — in-domain (development / testing)", C["real"])
    panel(20, "D1 T2 + D2 T2FS\n(generator ← D1,\ndiscriminator ← D2)",
          C["amber_bg"], C["amber"], "Synthetic\nD2-styled volumes",
          "PHASE 2 — cross-domain", C["amber"])

    ax.plot([2, 132], [52, 52], ls="--", color="#999999", lw=1.4)
    ax.text(67, 55, "Two sequential runs of the SAME architecture — not trained in parallel; "
                    "only the training data differs.",
            ha="center", fontsize=10, style="italic", color="#444")

    fig.suptitle("System architecture — identical model, two sequential training runs",
                 fontsize=13.5, fontweight="bold", y=0.98)
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


# ----------------------------------------------------------------------
def fig_label_channels(out_path: Path) -> None:
    chans = [
        ("0 · outside_body", "1 − body mask", "outside"),
        ("1 · uterus", "manual _ut", "uterus"),
        ("2 · ov_L", "auto-split _ov", "ovL"),
        ("3 · ov_R", "auto-split _ov", "ovR"),
        ("4 · em", "manual _em (em+)", "em"),
        ("5 · body_other", "body − organs", "other"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for ax, (title, src, key) in zip(axes.ravel(), chans):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        body = Ellipse((5, 4.6), 7.4, 6.6, facecolor="#F2F2F2", edgecolor="#AAAAAA", lw=1.2)
        ax.add_patch(body)
        hi = "#C44E52"
        # organ reference shapes
        uterus = Ellipse((5, 4.4), 1.9, 2.5, facecolor="none", edgecolor="#CCCCCC", lw=1)
        ovl = Circle((3.4, 4.9), 0.62, facecolor="none", edgecolor="#CCCCCC", lw=1)
        ovr = Circle((6.6, 4.9), 0.62, facecolor="none", edgecolor="#CCCCCC", lw=1)
        emc = Circle((6.7, 3.7), 0.5, facecolor="none", edgecolor="#CCCCCC", lw=1)
        for p in (uterus, ovl, ovr, emc):
            ax.add_patch(p)
        if key == "outside":
            ax.add_patch(FancyBboxPatch((0.2, 0.6), 9.6, 8.4, boxstyle="round,pad=0",
                         facecolor=hi, alpha=0.18, edgecolor="none", zorder=0))
            ax.add_patch(Ellipse((5, 4.6), 7.4, 6.6, facecolor="white", edgecolor="#AAAAAA", lw=1.2))
        elif key == "uterus":
            ax.add_patch(Ellipse((5, 4.4), 1.9, 2.5, facecolor=hi, alpha=0.75, edgecolor=hi))
        elif key == "ovL":
            ax.add_patch(Circle((3.4, 4.9), 0.62, facecolor=hi, alpha=0.85, edgecolor=hi))
        elif key == "ovR":
            ax.add_patch(Circle((6.6, 4.9), 0.62, facecolor=hi, alpha=0.85, edgecolor=hi))
        elif key == "em":
            ax.add_patch(Circle((6.7, 3.7), 0.5, facecolor=hi, alpha=0.85, edgecolor=hi))
        elif key == "other":
            ax.add_patch(Ellipse((5, 4.6), 7.4, 6.6, facecolor=hi, alpha=0.35, edgecolor="#AAAAAA", lw=1.2))
            for cx, cy, w, h in [(5, 4.4, 1.9, 2.5)]:
                ax.add_patch(Ellipse((cx, cy), w, h, facecolor="white", edgecolor="#CCCCCC"))
            for cx, cy in [(3.4, 4.9), (6.6, 4.9), (6.7, 3.7)]:
                ax.add_patch(Circle((cx, cy), 0.6, facecolor="white", edgecolor="#CCCCCC"))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(5, 0.0, src, ha="center", va="top", fontsize=9, color="#555", transform=ax.transData)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Six-channel one-hot label design (512×512) — every pixel in exactly one class",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


def fig_raovseg_pipeline(out_path: Path) -> None:
    # RAovSeg's four-stage runtime pipeline for OVARY_INTENSITY_ISSUE.md §1.1.
    # Two hard-coded stages (grey chip) bracket two learned stages (blue chip);
    # everything hinges on stage 1(c) landing the ovary in the enhancement band.
    fig, ax = plt.subplots(figsize=(14.5, 5.4))
    ax.set_xlim(0, 130); ax.set_ylim(0, 60); ax.axis("off")

    # Input / output pills
    _box(ax, 1, 26, 10, 8, "raw\nNIfTI", C["real_bg"], C["real"], fs=10, weight="bold")
    _box(ax, 121, 26, 10, 8, "ovary\nmask", C["good_bg"], C["good"], fs=10, weight="bold")

    # Four stage boxes: (x, w) — evenly spaced with room for arrows between.
    stages = [
        {  # 1. PREPROCESS
            "x": 13, "w": 25,
            "title": "1. PREPROCESS",
            "sub": ("• resample to 0.5³ mm\n"
                    "• percentile-clip -> [0,1]\n"
                    "• ENHANCEMENT\n"
                    "  o1=0.22, o2=0.30"),
            "kind": "hard-coded",
            "ref": "RAovSeg_tools.py:68\npreprocess.py:65",
        },
        {  # 2. RESCLASS
            "x": 40, "w": 25,
            "title": "2. RESCLASS",
            "sub": ("• per-slice binary\n"
                    "  'has ovary?'\n"
                    "• threshold 0.6\n"
                    "  (val-tuned)"),
            "kind": "learned",
            "ref": "train_resclass.py\nevaluate.py:31",
        },
        {  # 3. ATTUSEG
            "x": 67, "w": 25,
            "title": "3. ATTUSEG",
            "sub": ("• pixel-level ovary\n"
                    "  mask on flagged\n"
                    "  slices\n"
                    "  (attention U-Net)"),
            "kind": "learned",
            "ref": "train_attuseg.py",
        },
        {  # 4. POSTPROCESS
            "x": 94, "w": 25,
            "title": "4. POSTPROCESS",
            "sub": ("• binary closing\n"
                    "  (10 iterations)\n"
                    "• keep largest\n"
                    "  connected component"),
            "kind": "hard-coded",
            "ref": "RAovSeg_tools.py:150",
        },
    ]

    kind_colors = {
        "hard-coded": (C["neutral_bg"], C["neutral"], "hard-coded"),
        "learned":    (C["real_bg"],    C["real"],    "learned"),
    }

    # Draw stage boxes: title strip on top, substep body below, chip + ref beneath.
    for s in stages:
        x, w = s["x"], s["w"]
        # Title strip
        _box(ax, x, 44, w, 8, s["title"], "#F5F5F5", "#333", fs=11.5, weight="bold")
        # Substep body
        _box(ax, x, 18, w, 24, s["sub"], "#FFFFFF", "#BBB", fs=9.5)
        # Kind chip
        chip_fc, chip_ec, chip_txt = kind_colors[s["kind"]]
        _box(ax, x + w / 2 - 5, 10, 10, 5, chip_txt, chip_fc, chip_ec, fs=9, weight="bold")
        # File:line reference
        ax.text(x + w / 2, 5, s["ref"], ha="center", va="center",
                fontsize=8, color="#666", style="italic")

    # Arrows: input → 1 → 2 → 3 → 4 → output. Y-coord matches stage-body vertical centre.
    arrow_y = 30
    for x1, x2 in [(11, 13), (38, 40), (65, 67), (92, 94), (119, 121)]:
        _arrow(ax, x1, arrow_y, x2, arrow_y)

    fig.suptitle("RAovSeg's four-stage runtime pipeline — two hard-coded stages bracket two learned stages",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02,
             "The enhancement in stage 1(c) is load-bearing: it clamps voxels in [0.22, 0.30] to 1.0 "
             "and folds voxels > 0.5 to 1 − x. Feed the learned stages an input where that band never "
             "fires and the whole pipeline degrades.",
             ha="center", fontsize=9.5, style="italic", color="#444")
    fig.savefig(out_path); plt.close(fig); print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_ddpm(args.out_dir / "fig_ddpm_process.png")
    fig_architecture(args.out_dir / "fig_architecture.png")
    fig_label_channels(args.out_dir / "fig_label_channels.png")
    fig_raovseg_pipeline(args.out_dir / "fig_raovseg_pipeline.png")


if __name__ == "__main__":
    main()
