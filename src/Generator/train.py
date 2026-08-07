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
    EMAModel,
    build_inference_scheduler,
    build_model_from_cfg,
    build_train_scheduler,
)
from .patchgan import (
    PatchGAN,
    discriminator_accuracy,
    discriminator_loss,
    estimate_x0_from_eps,
    generator_adv_loss,
    lambda_schedule,
)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(
    path: Path, *, model, optim, step: int, cfg: dict,
    ema=None, discriminator=None, optim_d=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if optim_d is not None:
        payload["optim_d"] = optim_d.state_dict()
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


def save_sample_grid(
    samples: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    real_images: torch.Tensor | None = None,
):
    """Render one PNG with one row per sample. Columns are:

        [real]  synthetic  overlay  input-label-argmax

    where the leading 'real' column is shown ONLY if `real_images` is given
    — i.e. when the caller has the actual MRI slice that the label was
    extracted from (the case during training where labels are taken from
    the dataset's first batch and during inference_validate when labels
    are picked directly from the dataset index).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = samples.shape[0]
    img = (samples.detach().cpu().numpy() + 1.0) / 2.0
    img = np.clip(img, 0.0, 1.0)
    lbl = labels.detach().cpu().numpy()

    has_real = real_images is not None
    if has_real:
        real = (real_images.detach().cpu().numpy() + 1.0) / 2.0
        real = np.clip(real, 0.0, 1.0)

    n_cols = 4 if has_real else 3
    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[None, :]

    col = 0
    if has_real:
        for i in range(n):
            axes[i, col].imshow(real[i, 0], cmap="gray", vmin=0, vmax=1)
            axes[i, col].set_title("real (source of the label)")
            axes[i, col].axis("off")
        col += 1

    for i in range(n):
        axes[i, col].imshow(img[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, col].set_title("synthetic")
        axes[i, col].axis("off")

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
        axes[i, col + 1].imshow(rgb)
        axes[i, col + 1].set_title("overlay (Y=ut, R=L-ov, B=R-ov, G=em)")
        axes[i, col + 1].axis("off")

        argmax_lbl = lbl[i].argmax(axis=0)
        axes[i, col + 2].imshow(argmax_lbl, cmap="tab10", vmin=0, vmax=lbl.shape[1] - 1)
        axes[i, col + 2].set_title("input label (argmax)")
        axes[i, col + 2].axis("off")
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
    # Two schemas supported:
    #   Phase 1 (single cohort): `data:` has preprocessed_root/split_file at
    #     top level. The one loader supplies both DDPM samples and D's real
    #     path.
    #   Phase 2 (cross-domain): `data.generator:` and `data.discriminator:`
    #     hold two separate cohort configs. Generator loader supplies DDPM
    #     samples (D1 T2). Discriminator loader supplies D's real path only
    #     (D2 T2FS). The generated fake is critiqued against D2's style.
    dcfg = cfg["data"]
    cross_domain = "generator" in dcfg and "discriminator" in dcfg

    def _make_loader(sub_dcfg: dict, tag: str) -> DataLoader:
        ds_ = D2SliceDataset(
            preprocessed_root=sub_dcfg["preprocessed_root"],
            split_file=sub_dcfg["split_file"],
            split="train",
            sequence=sub_dcfg["sequence"],
            num_label_channels=sub_dcfg["num_label_channels"],
            image_size=sub_dcfg["image_size"],
        )
        sampler_ = ds_.make_weighted_sampler(sub_dcfg["ovary_oversample_weight"])
        print(f"[setup] {tag} loader: {len(ds_)} slices from "
              f"{sub_dcfg['preprocessed_root']} ({sub_dcfg['sequence']})")
        return DataLoader(
            ds_,
            batch_size=cfg["training"]["batch_size"],
            sampler=sampler_,
            num_workers=sub_dcfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=sub_dcfg["num_workers"] > 0,
        )

    if cross_domain:
        print("[setup] cross-domain mode: separate generator + discriminator loaders")
        loader = _make_loader(dcfg["generator"], "generator (D1 T2)")
        disc_loader = _make_loader(dcfg["discriminator"], "discriminator (D2 T2FS)")
        num_label_channels = dcfg["generator"]["num_label_channels"]
    else:
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
        disc_loader = None
        num_label_channels = dcfg["num_label_channels"]

    # --- model --- #
    model = build_model_from_cfg(cfg).to(device)
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

    # --- Discriminator (Exp 1c only) --- #
    # Only constructed when the YAML has a `discriminator:` block. Keeps 1a/1b
    # training paths unchanged.
    dcfg_disc = cfg.get("discriminator")
    discriminator: torch.nn.Module | None = None
    optim_d = None
    if dcfg_disc is not None:
        if dcfg_disc.get("type", "patchgan") != "patchgan":
            raise ValueError(f"Unknown discriminator type: {dcfg_disc.get('type')}")
        discriminator = PatchGAN(
            image_channels=1,
            label_channels=num_label_channels,
            base_channels=int(dcfg_disc.get("base_channels", 64)),
            use_spectral_norm=bool(dcfg_disc.get("use_spectral_norm", True)),
        ).to(device)
        d_unconditional = bool(dcfg_disc.get("unconditional", False))
        if d_unconditional:
            print("[setup] D is UNCONDITIONAL: label channel zeroed before D forward")
        d_params = sum(p.numel() for p in discriminator.parameters())
        print(f"[setup] PatchGAN params: {d_params/1e6:.1f}M")
        optim_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(dcfg_disc["lr"]),
            weight_decay=float(dcfg_disc.get("weight_decay", 0.0)),
        )
        print(f"[setup] D optimiser: AdamW lr={dcfg_disc['lr']}")
        print(f"[setup] λ schedule: warmup_end={dcfg_disc['lambda_warmup_end']} "
              f"ramp_end={dcfg_disc['lambda_ramp_end']} "
              f"peak={dcfg_disc['lambda_peak']}")

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
        if discriminator is not None and "discriminator" in resume_payload:
            discriminator.load_state_dict(resume_payload["discriminator"])
            if optim_d is not None and "optim_d" in resume_payload:
                optim_d.load_state_dict(resume_payload["optim_d"])
            print(f"[ckpt] resumed discriminator state from step {start_step - 1}")

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
    # Keep the real MRI slices that those labels came from. They show up as
    # the leftmost column in every periodic sample grid, so you can compare
    # the synthetic against the real source slice at a glance.
    fixed_real_images = chosen_batch["image"][:n_grid].to(device)

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)
    disc_iter = iter(disc_loader) if disc_loader is not None else None

    def _next_from(loader_, iter_ref):
        try:
            return next(iter_ref[0])
        except StopIteration:
            iter_ref[0] = iter(loader_)
            return next(iter_ref[0])

    disc_iter_ref = [disc_iter] if disc_iter is not None else None

    # --- Per-timestep loss buckets for calibration ---
    # Diffusion training samples t uniformly, so the scalar L_diff averages
    # across all noise levels. That hides whether the model has converged on
    # low-t (near-clean, texture) vs high-t (near-noise, layout) equally.
    # We track an EMA of the per-sample MSE bucketed into 10 timestep bins,
    # log each bin as its own TensorBoard scalar, and print a compact row
    # every N log intervals.
    num_train_timesteps = int(cfg["diffusion"]["num_train_timesteps"])
    NUM_T_BUCKETS = 10
    bucket_ema = torch.zeros(NUM_T_BUCKETS, device=device)
    bucket_seen = torch.zeros(NUM_T_BUCKETS, device=device)
    BUCKET_EMA_DECAY = 0.99

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
            # Per-sample MSE first, then batch-mean. Same scalar as before,
            # but the per-sample tensor is needed for per-t calibration bucketing.
            per_sample_mse = F.mse_loss(
                eps_pred.float(), noise.float(), reduction="none"
            ).mean(dim=[1, 2, 3])
            loss_diff = per_sample_mse.mean()

        # Update per-timestep calibration EMA (no grad, negligible cost).
        with torch.no_grad():
            bucket_ids = (t.float() * NUM_T_BUCKETS / num_train_timesteps).long().clamp_(0, NUM_T_BUCKETS - 1)
            for i in range(b):
                bi = int(bucket_ids[i])
                v = per_sample_mse[i].detach()
                if bucket_seen[bi] == 0:
                    bucket_ema[bi] = v
                    bucket_seen[bi] = 1
                else:
                    bucket_ema[bi] = BUCKET_EMA_DECAY * bucket_ema[bi] + (1 - BUCKET_EMA_DECAY) * v

        # --- Conditional PatchGAN block (Exp 1c only) --- #
        # Stays identity-no-op when discriminator is None (1a/1b paths).
        loss_g_adv = torch.tensor(0.0, device=device)
        loss_d = torch.tensor(0.0, device=device)
        d_real_acc, d_fake_acc = 0.0, 0.0
        lam = 0.0
        if discriminator is not None:
            lam = lambda_schedule(
                step,
                warmup_end=int(dcfg_disc["lambda_warmup_end"]),
                ramp_end=int(dcfg_disc["lambda_ramp_end"]),
                peak=float(dcfg_disc["lambda_peak"]),
            )
            # The label fed to D uses the ORIGINAL (non-CFG-dropped) version
            # so D always sees a meaningful label-image consistency signal,
            # regardless of whether the generator was given a dropped label
            # this step. We grab it from the batch again.
            lbl_d = batch["label"].to(device, non_blocking=True)
            # Single-step x̂_0 estimate — see patchgan.estimate_x0_from_eps docstring.
            # NOTE: do NOT detach eps_pred here — we need the graph intact so
            # that the G-adversarial path (line further down) can propagate
            # gradient back through discriminator → x0_hat → eps_pred → G params.
            # The D update below re-uses x0_hat.detach() to sever the graph for D.
            # (Historical bug: this used to be eps_pred.detach(); that severed the
            #  graph at creation time, so all λ ablation runs had zero adversarial
            #  gradient into G. Confirmed by GRAD_DIAG showing |grad_lam·L_adv|=0.)
            x0_hat = estimate_x0_from_eps(x_t, eps_pred.float(),
                                          train_sched, t)

            # D real path: single-cohort mode uses the same batch (x0);
            # cross-domain mode pulls a fresh batch from the D2 disc loader
            # so D's "real" is authentic D2 T2FS.
            if disc_iter_ref is not None:
                disc_batch = _next_from(disc_loader, disc_iter_ref)
                x0_real = disc_batch["image"].to(device, non_blocking=True)
            else:
                x0_real = x0

            # Unconditional D (Phase 2): zero out label to prevent D from
            # short-circuiting on D1↔D2 label-distribution differences and
            # ignoring image style.
            if d_unconditional:
                lbl_d_real = torch.zeros_like(lbl_d)
                lbl_d_fake = torch.zeros_like(lbl_d)
            else:
                lbl_d_real = lbl_d
                lbl_d_fake = lbl_d

            if lam > 0.0:
                # --- D step: real vs fake.detach() ---
                d_real_logits = discriminator(x0_real, lbl_d_real)
                d_fake_logits = discriminator(x0_hat.detach(), lbl_d_fake)
                loss_d = discriminator_loss(d_real_logits, d_fake_logits)
                optim_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                optim_d.step()
                d_real_acc, d_fake_acc = discriminator_accuracy(d_real_logits, d_fake_logits)

                # --- G adversarial loss: D should call our fakes 'real' ---
                # x0_hat WITHOUT detach so gradient flows back into G.
                d_fake_logits_for_g = discriminator(x0_hat, lbl_d_fake)
                loss_g_adv = generator_adv_loss(d_fake_logits_for_g)

        loss_g_total = loss_diff + lam * loss_g_adv

        # --- GRAD_DIAG (opt-in gradient-contribution logging) --- #
        # When env GRAD_DIAG=1, print separate gradient norms for L_diff and
        # (lam * L_adv) at every log_every step where lam > 0. Answers the
        # LAMBDA_ABLATION_COLLAPSE.md hypothesis 5.1 question ("is the
        # adversarial gradient being AMP-underflowed?"). Cost is ~2× a
        # normal step because we split gradients — only fired at log steps.
        import os as _os
        if _os.environ.get("GRAD_DIAG") == "1" and step % log_every == 0 and lam > 0.0:
            from torch.autograd import grad as _autograd_grad
            _params = [p for p in model.parameters() if p.requires_grad]
            _g_diff = _autograd_grad(loss_diff,          _params,
                                     retain_graph=True, allow_unused=True)
            _g_adv  = _autograd_grad(lam * loss_g_adv,   _params,
                                     retain_graph=True, allow_unused=True)
            _n_diff = sum(g.detach().float().norm().item()**2
                          for g in _g_diff if g is not None) ** 0.5
            _n_adv  = sum(g.detach().float().norm().item()**2
                          for g in _g_adv  if g is not None) ** 0.5
            _ratio  = _n_adv / max(_n_diff, 1e-12)
            print(f"[GRAD_DIAG] step={step} lam={lam:.4e} "
                  f"|grad_L_diff|={_n_diff:.6e} "
                  f"|grad_lam_L_adv|={_n_adv:.6e} ratio={_ratio:.6e}",
                  flush=True)

        optim.zero_grad(set_to_none=True)
        loss_g_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        if ema is not None:
            ema.update(model)

        if step % log_every == 0:
            dt = time.time() - t0
            it_per_s = (step - start_step + 1) / max(dt, 1e-6)
            if discriminator is not None:
                print(f"[step {step:>6d}/{total_steps}] L_diff={loss_diff.item():.4f} "
                      f"L_adv={loss_g_adv.item():.4f} L_D={loss_d.item():.4f} "
                      f"λ={lam:.4f} D_acc(r/f)={d_real_acc:.2f}/{d_fake_acc:.2f} "
                      f"({it_per_s:.2f} it/s)")
                writer.add_scalar("loss/L_diff", loss_diff.item(), step)
                writer.add_scalar("loss/L_adv_g", loss_g_adv.item(), step)
                writer.add_scalar("loss/L_d", loss_d.item(), step)
                writer.add_scalar("loss/lambda", lam, step)
                writer.add_scalar("disc/acc_real", d_real_acc, step)
                writer.add_scalar("disc/acc_fake", d_fake_acc, step)
            else:
                print(f"[step {step:>6d}/{total_steps}] L_diff={loss_diff.item():.4f} "
                      f"({it_per_s:.2f} it/s)")
                writer.add_scalar("loss/L_diff", loss_diff.item(), step)
            writer.add_scalar("speed/it_per_s", it_per_s, step)

            # Per-timestep calibration: one TB scalar per bucket, plus a
            # compact console row every 10 log intervals so the reader can
            # see the full curve without opening TensorBoard.
            for _bi in range(NUM_T_BUCKETS):
                if bucket_seen[_bi] > 0:
                    writer.add_scalar(f"loss_by_t/bucket_{_bi:02d}",
                                      bucket_ema[_bi].item(), step)
            if step % (log_every * 10) == 0 and step > 0:
                _row = " ".join(
                    f"{bucket_ema[_bi].item():.3f}" if bucket_seen[_bi] > 0 else "  ---"
                    for _bi in range(NUM_T_BUCKETS)
                )
                print(f"[t-calib] step={step:>6d} buckets t=[0,100)..[900,1000): {_row}")

        if step > 0 and step % sample_every == 0:
            # Use EMA model for sampling if available — cleaner samples.
            sample_model = ema.ema_model if ema is not None else model
            sample_model.eval()
            with torch.no_grad():
                samples = sample_model.sample(fixed_labels, infer_sched, device,
                                              guidance_scale=guidance_scale)
            save_sample_grid(samples, fixed_labels,
                             sample_dir / f"step_{step:06d}.png",
                             real_images=fixed_real_images)
            if ema is None:
                model.train()  # EMA model stays in eval; training model returns to train

        if step > 0 and step % ckpt_every == 0:
            save_ckpt(ckpt_dir / f"step_{step:06d}.pt",
                      model=model, optim=optim, step=step, cfg=cfg, ema=ema,
                      discriminator=discriminator, optim_d=optim_d)

        step += 1

    save_ckpt(ckpt_dir / f"step_{step:06d}.pt",
              model=model, optim=optim, step=step, cfg=cfg, ema=ema)
    sample_model = ema.ema_model if ema is not None else model
    sample_model.eval()
    with torch.no_grad():
        samples = sample_model.sample(fixed_labels, infer_sched, device,
                                      guidance_scale=guidance_scale)
    save_sample_grid(samples, fixed_labels,
                     sample_dir / f"step_{step:06d}_final.png",
                     real_images=fixed_real_images)
    writer.close()
    print(f"[done] total wall time: {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
