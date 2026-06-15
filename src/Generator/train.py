"""
Exp 1a training: 2D conditional DDPM with concat conditioning.

Single GPU. Run via SLURM (scripts/train_exp1a.sh). ~12-18h to 80k steps
on A100 80GB at batch 8.

Usage:
    python -m src.Generator.train --config configs/exp1a.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import D2SliceDataset
from .model import (
    ConcatConditionedDDPM,
    EMAModel,
    build_inference_scheduler,
    build_train_scheduler,
    build_unet,
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, *, model, optim, step: int, cfg: dict, ema=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)
    print(f"[ckpt] saved {path} @ step {step}")


def load_latest_ckpt(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    print(f"[ckpt] resuming from {latest}")
    return torch.load(latest, map_location="cpu")


def save_sample_grid(samples: torch.Tensor, labels: torch.Tensor, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = samples.shape[0]
    img = (samples.detach().cpu().numpy() + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    lbl = labels.detach().cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title("synthetic")
        axes[i, 0].axis("off")

        rgb = np.stack([img[i, 0]] * 3, axis=-1)
        if lbl.shape[1] >= 4:
            uterus_mask = lbl[i, 1] > 0.5
            ov_l_mask = lbl[i, 2] > 0.5
            ov_r_mask = lbl[i, 3] > 0.5
            rgb[uterus_mask] = [1.0, 1.0, 0.0]   # yellow
            rgb[ov_l_mask] = [1.0, 0.0, 0.0]     # red — left ovary
            rgb[ov_r_mask] = [0.0, 0.5, 1.0]     # blue — right ovary
            if lbl.shape[1] >= 5:
                em_mask = lbl[i, 4] > 0.5
                rgb[em_mask] = [0.0, 1.0, 0.0]   # green — endometrioma
        axes[i, 1].imshow(rgb)
        axes[i, 1].set_title("overlay (Y=ut, R=L-ov, B=R-ov, G=em)")
        axes[i, 1].axis("off")

        argmax_lbl = lbl[i].argmax(axis=0)
        axes[i, 2].imshow(argmax_lbl, cmap="tab10", vmin=0, vmax=lbl.shape[1] - 1)
        axes[i, 2].set_title("input label (argmax)")
        axes[i, 2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    sample_dir = out_dir / "samples"
    tb_dir = out_dir / "tb"

    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    if device.type == "cuda":
        print(f"[setup] GPU={torch.cuda.get_device_name(0)}")

    # --- decide resume vs fresh start, clean previous run artefacts ---
    # If resume is enabled AND there's a checkpoint, we keep everything.
    # Otherwise we wipe samples/ ckpt/ tb/ so a fresh run never mixes two
    # training histories in the same output dir.
    resume_payload = None
    if cfg["experiment"].get("resume", False):
        resume_payload = load_latest_ckpt(ckpt_dir)
    if resume_payload is None:
        for sub, label in [(sample_dir, "samples"), (ckpt_dir, "ckpt"), (tb_dir, "tb")]:
            if sub.exists():
                print(f"[clean] removing previous {label} dir at {sub}")
                shutil.rmtree(sub)

    # --- data --- #
    dcfg = cfg["data"]
    ds = D2SliceDataset(
        preprocessed_root=dcfg["preprocessed_root"],
        split_file=dcfg["split_file"],
        split="train",
        sequence=dcfg["sequence"],
        num_label_channels=dcfg["num_label_channels"],
        image_size=dcfg["image_size"],
    )
    sampler = ds.make_weighted_sampler(dcfg["ovary_oversample_weight"])
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=dcfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=dcfg["num_workers"] > 0,
    )

    # --- model --- #
    unet = build_unet(cfg["model"])
    model = ConcatConditionedDDPM(unet).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] U-Net params: {n_params/1e6:.1f}M")

    train_sched = build_train_scheduler(cfg["diffusion"])
    infer_sched = build_inference_scheduler(
        cfg["diffusion"], cfg["sampling"]["num_inference_steps"]
    )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # --- EMA (created BEFORE resume so we can load saved EMA state) --- #
    ema_decay = float(cfg["training"].get("ema_decay", 0.0))
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        print(f"[setup] EMA enabled, decay={ema_decay}")

    # --- resume --- #
    start_step = 0
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        optim.load_state_dict(resume_payload["optim"])
        start_step = resume_payload["step"] + 1
        if ema is not None and "ema" in resume_payload:
            ema.load_state_dict(resume_payload["ema"])
            print(f"[ckpt] resumed EMA state from step {start_step - 1}")

    writer = SummaryWriter(log_dir=str(tb_dir))

    # --- train --- #
    total_steps = cfg["training"]["total_steps"]
    log_every = cfg["training"]["log_every"]
    sample_every = cfg["training"]["sample_every"]
    ckpt_every = cfg["training"]["ckpt_every"]
    grad_clip = cfg["training"]["grad_clip"]
    use_amp = cfg["training"]["amp"]
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    cfg_dropout_prob = float(cfg["training"].get("cfg_dropout_prob", 0.0))
    guidance_scale = float(cfg["sampling"].get("guidance_scale", 1.0))
    print(f"[setup] CFG dropout_prob={cfg_dropout_prob}, "
          f"sampling guidance_scale={guidance_scale}")

    # --- Pick a fixed batch with anatomy for the periodic visualisation grid.
    # Without this, the first batch from the loader has ~24% chance of being
    # all background-only slices, leaving the in-training grids unable to
    # show whether CFG/EMA are doing anything (blank labels = blank overlays).
    # We resample up to 20 batches looking for one with at least half the
    # samples carrying foreground voxels.
    n_grid = cfg["sampling"]["num_samples_per_grid"]
    required_fg = max(2, n_grid // 2)
    chosen_batch = None
    chosen_fg = -1
    best_batch = None
    best_fg = -1
    for attempt in range(20):
        cand = next(iter(loader))
        # Score on target organs only (channels 1-4 = uterus, L-ov, R-ov, em).
        # Excludes channel 0 (outside_body) and channel 5 (body_other) — those
        # are always non-zero somewhere and don't signal "interesting anatomy."
        fg_per_sample = (cand["label"][:, 1:5].sum(dim=(1, 2, 3)) > 0).long()
        fg = int(fg_per_sample[:n_grid].sum())
        if fg > best_fg:
            best_fg, best_batch = fg, cand
        if fg >= required_fg:
            chosen_batch, chosen_fg = cand, fg
            print(f"[setup] fixed_labels chosen at attempt {attempt + 1}: "
                  f"{fg}/{n_grid} samples have foreground")
            break
    if chosen_batch is None:
        chosen_batch, chosen_fg = best_batch, best_fg
        print(f"[setup] WARNING: no batch met required_fg={required_fg} after 20 "
              f"attempts; using best ({best_fg}/{n_grid} with foreground)")
    fixed_labels = chosen_batch["label"][:n_grid].to(device)

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x0 = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)

        b = x0.shape[0]
        # CFG dropout: with prob cfg_dropout_prob per sample, replace label
        # with all-zeros so the model learns the unconditional distribution
        # alongside the conditional one. Drops are independent per sample.
        if cfg_dropout_prob > 0.0:
            keep = (torch.rand(b, device=device) >= cfg_dropout_prob).float()
            lbl = lbl * keep.view(-1, 1, 1, 1)

        t = torch.randint(
            0, cfg["diffusion"]["num_train_timesteps"], (b,), device=device, dtype=torch.long
        )
        noise = torch.randn_like(x0)
        x_t = train_sched.add_noise(original_samples=x0, noise=noise, timesteps=t)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            eps_pred = model.predict_noise(x_t, lbl, t)
            loss = F.mse_loss(eps_pred.float(), noise.float())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        if ema is not None:
            ema.update(model)

        if step % log_every == 0:
            dt = time.time() - t0
            it_per_s = (step - start_step + 1) / max(dt, 1e-6)
            print(f"[step {step:>6d}/{total_steps}] L_diff={loss.item():.4f} "
                  f"({it_per_s:.2f} it/s)")
            writer.add_scalar("loss/L_diff", loss.item(), step)
            writer.add_scalar("speed/it_per_s", it_per_s, step)

        if step > 0 and step % sample_every == 0:
            # Use EMA model for sampling if available — cleaner samples.
            sample_model = ema.ema_model if ema is not None else model
            sample_model.eval()
            with torch.no_grad():
                samples = sample_model.sample(fixed_labels, infer_sched, device,
                                              guidance_scale=guidance_scale)
            save_sample_grid(samples, fixed_labels, sample_dir / f"step_{step:06d}.png")
            if ema is None:
                model.train()  # EMA model stays in eval; training model returns to train

        if step > 0 and step % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"step_{step:06d}.pt",
                      model=model, optim=optim, step=step, cfg=cfg, ema=ema)

        step += 1

    save_ckpt(ckpt_dir / f"step_{step:06d}.pt",
              model=model, optim=optim, step=step, cfg=cfg, ema=ema)
    sample_model = ema.ema_model if ema is not None else model
    sample_model.eval()
    with torch.no_grad():
        samples = sample_model.sample(fixed_labels, infer_sched, device,
                                      guidance_scale=guidance_scale)
    save_sample_grid(samples, fixed_labels, sample_dir / f"step_{step:06d}_final.png")
    writer.close()
    print(f"[done] total wall time: {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
