"""
Post-hoc calibration diagnostic for a trained conditional DDPM.

Answers "does the model actually denoise well at every timestep?" — a question
the average training L_diff hides (it's a scalar over uniformly-sampled t).

For a grid of timesteps t (default 50, 100, 200, 400, 600, 800, 950), does the
round-trip test on real validation subjects:

    x0   →  add_noise(t)  →  x_t  →  predict_noise  →  eps_hat
                                            ↓
    reconstruct x0_hat via one-step DDIM from (x_t, eps_hat, t)

Reports per-t:
  * MSE(eps_hat, true_noise)             — noise-prediction accuracy
  * cosine_sim(eps_hat, true_noise)      — direction agreement
  * MSE(x0_hat, x0)                      — single-step reconstruction fidelity
  * SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))    — reference axis

Both raw JSON and a two-panel figure are produced. Runs in ~2 min on one A100
for --n 32 samples across 7 t-values.

Usage:
    python -m src.Generator.calibration \\
        --config src/Generator/exp2_lam05_fixed.yaml \\
        --ckpt   /mnt/parscratch/.../runs/exp2_lam05_fixed/ckpt/step_100000.pt \\
        --out_json runs/exp2_lam05_fixed/calibration.json \\
        --out_png  runs/exp2_lam05_fixed/calibration.png \\
        --n 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset import D2SliceDataset
from .model import build_model_from_cfg, build_train_scheduler, resolve_gen_data_cfg


DEFAULT_T_GRID = [50, 100, 200, 400, 600, 800, 950]


@torch.no_grad()
def _calibration_for_t(
    model,
    train_sched,
    x0: torch.Tensor,       # (N, 1, H, W)
    label: torch.Tensor,    # (N, C, H, W)
    t_val: int,
    device: torch.device,
) -> dict:
    """Run the round-trip test at a single timestep, averaged across the batch."""
    N = x0.shape[0]
    t = torch.full((N,), int(t_val), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = train_sched.add_noise(original_samples=x0, noise=noise, timesteps=t)

    eps_hat = model.predict_noise(x_t, label, t).float()

    # Per-sample noise-prediction MSE and cosine similarity, then batch-mean.
    noise_f = noise.float()
    eps_mse = F.mse_loss(eps_hat, noise_f, reduction="none").mean(dim=[1, 2, 3])
    cos = F.cosine_similarity(
        eps_hat.reshape(N, -1), noise_f.reshape(N, -1), dim=1
    )

    # Single-step DDIM x0 estimate:
    #   x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
    alpha_bar = train_sched.alphas_cumprod.to(device)[t]           # (N,)
    sqrt_ab = alpha_bar.sqrt().view(N, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(N, 1, 1, 1)
    x0_hat = (x_t - sqrt_1mab * eps_hat) / sqrt_ab.clamp_min(1e-8)
    x0_mse = F.mse_loss(x0_hat, x0.float(), reduction="none").mean(dim=[1, 2, 3])

    # log-SNR reference for the same t
    snr = (alpha_bar / (1.0 - alpha_bar).clamp_min(1e-12)).clamp_min(1e-12)
    log_snr = snr.log().mean().item()

    return {
        "t": int(t_val),
        "eps_mse_mean": float(eps_mse.mean().item()),
        "eps_mse_std":  float(eps_mse.std().item()),
        "cos_sim_mean": float(cos.mean().item()),
        "cos_sim_std":  float(cos.std().item()),
        "x0_mse_mean":  float(x0_mse.mean().item()),
        "x0_mse_std":   float(x0_mse.std().item()),
        "log_snr":      log_snr,
    }


def _plot(rows: list[dict], out_png: Path, title_suffix: str = "") -> None:
    ts = [r["t"] for r in rows]
    eps_mean = [r["eps_mse_mean"] for r in rows]
    eps_std  = [r["eps_mse_std"]  for r in rows]
    cos_mean = [r["cos_sim_mean"] for r in rows]
    cos_std  = [r["cos_sim_std"]  for r in rows]
    x0_mean  = [r["x0_mse_mean"]  for r in rows]
    x0_std   = [r["x0_mse_std"]   for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].errorbar(ts, eps_mean, yerr=eps_std, marker="o", capsize=3)
    axes[0].set_title("noise-prediction MSE per t")
    axes[0].set_xlabel("timestep t")
    axes[0].set_ylabel("MSE(ε̂, ε)")
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.7,
                    label="ε ~ N(0,1) baseline")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(ts, cos_mean, yerr=cos_std, marker="o", capsize=3, color="C1")
    axes[1].set_title("cosine sim per t")
    axes[1].set_xlabel("timestep t")
    axes[1].set_ylabel("cos(ε̂, ε)")
    axes[1].axhline(0.0, color="gray", ls="--", lw=0.7)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    axes[2].errorbar(ts, x0_mean, yerr=x0_std, marker="o", capsize=3, color="C2")
    axes[2].set_title("single-step reconstruction MSE per t")
    axes[2].set_xlabel("timestep t")
    axes[2].set_ylabel("MSE(x0_hat, x0)")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle(f"DDPM calibration{(' — ' + title_suffix) if title_suffix else ''}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--config", required=True, help="training YAML")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--n", type=int, default=32,
                    help="samples per t-value (drawn from D2 train split; default 32)")
    ap.add_argument("--t-grid", type=int, nargs="+", default=DEFAULT_T_GRID)
    ap.add_argument("--split", default="train",
                    help="split to draw real samples from (default: train)")
    ap.add_argument("--no-ema", action="store_true",
                    help="use training weights instead of EMA (default: EMA if present)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calib] device={device}")

    # --- Build model + scheduler --- #
    model = build_model_from_cfg(cfg).to(device)
    train_sched = build_train_scheduler(cfg["diffusion"])

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
        weight_src = "EMA"
    else:
        model.load_state_dict(ckpt["model"])
        weight_src = "training"
    model.eval()
    print(f"[calib] loaded ckpt step {ckpt['step']}, weights={weight_src}")

    # --- Draw N real samples --- #
    # Use the discriminator loader for cross-domain configs (real D2 T2FS),
    # generator loader otherwise. Both are D2SliceDataset with the same API.
    dcfg = resolve_gen_data_cfg(cfg)
    # For Phase 2 configs, dcfg is the generator (D1); the "real" reference for
    # calibration should be what the model is meant to imitate. Use the discriminator
    # data if defined, else the generator data.
    if "data" in cfg and isinstance(cfg["data"], dict) and "discriminator" in cfg["data"]:
        ref_cfg = cfg["data"]["discriminator"]
        print("[calib] using discriminator loader as real reference (Phase 2 mode)")
    else:
        ref_cfg = dcfg
        print("[calib] using generator loader as real reference (Phase 1 mode)")

    ds = D2SliceDataset(
        preprocessed_root=ref_cfg["preprocessed_root"],
        split_file=ref_cfg["split_file"],
        split=args.split,
        sequence=ref_cfg["sequence"],
        num_label_channels=int(ref_cfg["num_label_channels"]),
        image_size=int(ref_cfg["image_size"]),
    )
    print(f"[calib] dataset: {len(ds)} slices in split '{args.split}'")

    rng = np.random.default_rng(args.seed)
    # Prefer ovary-containing slices for a more representative comparison.
    ov_ix  = [i for i, s in enumerate(ds.index) if s.has_ovary]
    non_ix = [i for i, s in enumerate(ds.index) if not s.has_ovary]
    n_ov = min(args.n // 2, len(ov_ix))
    n_no = args.n - n_ov
    pick = list(rng.choice(ov_ix, size=n_ov, replace=False)) + \
           list(rng.choice(non_ix, size=n_no, replace=False))
    print(f"[calib] drew {len(pick)} slices ({n_ov} with ovary + {n_no} without)")

    # Stack into (N, 1, H, W) image + (N, C, H, W) label
    imgs, lbls = [], []
    for i in pick:
        s = ds[i]
        imgs.append(s["image"].unsqueeze(0))
        lbls.append(s["label"].unsqueeze(0))
    x0 = torch.cat(imgs, dim=0).to(device)
    label = torch.cat(lbls, dim=0).to(device)
    print(f"[calib] x0 shape: {tuple(x0.shape)}  label shape: {tuple(label.shape)}")

    # --- Run per-t --- #
    torch.manual_seed(args.seed)
    rows: list[dict] = []
    for t_val in args.t_grid:
        row = _calibration_for_t(model, train_sched, x0, label, t_val, device)
        rows.append(row)
        print(f"[calib] t={t_val:>4d}  "
              f"ε-MSE={row['eps_mse_mean']:.4f}±{row['eps_mse_std']:.4f}  "
              f"cos={row['cos_sim_mean']:.4f}±{row['cos_sim_std']:.4f}  "
              f"x0-MSE={row['x0_mse_mean']:.4f}±{row['x0_mse_std']:.4f}  "
              f"logSNR={row['log_snr']:.2f}")

    # --- Save --- #
    out = {
        "config":     args.config,
        "ckpt":       args.ckpt,
        "ckpt_step":  int(ckpt["step"]),
        "weight_src": weight_src,
        "n_samples":  int(x0.shape[0]),
        "n_ov":       int(n_ov),
        "n_no_ov":    int(n_no),
        "t_grid":     list(args.t_grid),
        "rows":       rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")

    variant = Path(args.config).stem
    _plot(rows, Path(args.out_png), title_suffix=f"{variant} @ step {ckpt['step']}")


if __name__ == "__main__":
    main()
