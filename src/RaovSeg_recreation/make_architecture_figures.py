"""
Background + architecture diagrams for the dissertation (no data needed):

    fig_ddpm_process.png     — forward noising / reverse denoising schematic
    fig_architecture.png     — full two-phase system architecture
    fig_label_channels.png   — the 6-channel one-hot label design

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
    "savefig.dpi": 300, "savefig.bbox": "tight",
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
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=1.0",
                 linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color="#404040", style="-|>", lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=15, lw=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ----------------------------------------------------------------------
def fig_ddpm(out_path: Path) -> None:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:64, 0:64]
    base = (np.exp(-(((xx - 30) ** 2 + (yy - 26) ** 2) / 200)) * 0.9
            + np.exp(-(((xx - 42) ** 2 + (yy - 42) ** 2) / 70)) * 0.6)
    base = base / base.max()
    noise = rng.standard_normal((64, 64)) * 0.5 + 0.5

    def mix(f):
        return np.clip((1 - f) * base + f * noise, 0, 1)

    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    xs = [0.085, 0.255, 0.425, 0.595, 0.765]
    wimg, himg = 0.135, 0.26
    yF, yR = 0.58, 0.10

    fig = plt.figure(figsize=(12.5, 6.6))
    ov = fig.add_axes([0, 0, 1, 1]); ov.axis("off")
    ov.set_xlim(0, 1); ov.set_ylim(0, 1)

    def draw_row(y, fvals, labels, ec):
        centers = []
        for i, f in enumerate(fvals):
            a = fig.add_axes([xs[i], y, wimg, himg])
            a.imshow(mix(f), cmap="gray", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(ec); s.set_linewidth(1.6)
            ov.text(xs[i] + wimg / 2, y - 0.035, labels[i], ha="center",
                    va="top", fontsize=11)
            centers.append((xs[i] + wimg / 2, y + himg / 2,
                            xs[i], xs[i] + wimg))
        return centers

    fwd = draw_row(yF, steps, [r"$x_0$", r"$x_1$", r"$x_t$", r"", r"$x_T$"], "#888")
    # reverse row: x_T on the right (under forward x_T), denoising flows right -> left
    rev = draw_row(yR, steps, [r"$x_0$", r"", r"", r"", r"$x_T$"], C["real"])

    yc_f = yF + himg / 2
    yc_r = yR + himg / 2
    for i in range(len(xs) - 1):
        _arrow(ov, fwd[i][3], yc_f, fwd[i + 1][2], yc_f, color=C["gen"])
        # reverse arrow points left (denoising: x_T -> x_0)
        _arrow(ov, rev[i + 1][2], yc_r, rev[i][3], yc_r, color=C["real"])

    # straight vertical link just right of the patches (clears the x_T labels)
    xcol = xs[-1] + wimg + 0.012
    _arrow(ov, xcol, yF + 0.02, xcol, yR + himg, color=C["neutral"], lw=1.6)

    ov.text(0.5, 0.965, "The denoising diffusion process (DDPM)",
            ha="center", fontsize=14, fontweight="bold")
    ov.text(0.5, yF + himg + 0.035,
            "Forward diffusion  q(xₜ | xₜ₋₁): add Gaussian noise over T = 1000 steps  "
            "(linear β: 1e-4 → 2e-2)",
            ha="center", fontsize=10.5, color=C["gen"], fontweight="bold")
    ov.text(0.5, yR + himg + 0.055,
            "Reverse denoising  p_θ(xₜ₋₁ | xₜ): a 2D U-Net predicts the noise ε̂ at each step",
            ha="center", fontsize=10.5, color=C["real"], fontweight="bold")
    ov.text(0.5, yR - 0.082,
            "U-Net conditioned on the 6-channel label + timestep t · classifier-free guidance · "
            "EMA weights · 100 inference steps",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_ddpm(args.out_dir / "fig_ddpm_process.png")
    fig_architecture(args.out_dir / "fig_architecture.png")
    fig_label_channels(args.out_dir / "fig_label_channels.png")


if __name__ == "__main__":
    main()
