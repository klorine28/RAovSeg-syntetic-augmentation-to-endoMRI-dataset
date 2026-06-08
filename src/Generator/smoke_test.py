"""
Smoke test: build dataset + model, run 20 steps, sample once.
Catches CUDA / MONAI / dataloader issues before submitting a real job.

Run inside an interactive GPU session:

    srun --partition=gpu --qos=gpu --gres=gpu:1 \
         --mem=82G --cpus-per-task=4 --time=01:00:00 --pty bash
    module load Anaconda3/2024.02-1
    module load cuDNN/8.9.2.26-CUDA-12.1.1
    set +u; source activate synth_mri; set -u
    cd /mnt/parscratch/users/$USER/synth_mri/EndometriosisDataset
    python -m src.Generator.smoke_test --config src/Generator/exp1a.yaml
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .dataset import D2SliceDataset
from .model import (
    ConcatConditionedDDPM,
    build_inference_scheduler,
    build_train_scheduler,
    build_unet,
)


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Smoke test requires a GPU."
    print(f"[smoke] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[smoke] free mem: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    ds = D2SliceDataset(
        preprocessed_root=cfg["data"]["preprocessed_root"],
        split_file=cfg["data"]["split_file"],
        split="train",
        sequence=cfg["data"]["sequence"],
        num_label_channels=cfg["data"]["num_label_channels"],
        image_size=cfg["data"]["image_size"],
    )
    sampler = ds.make_weighted_sampler(cfg["data"]["ovary_oversample_weight"])
    loader = DataLoader(
        ds, batch_size=cfg["training"]["batch_size"], sampler=sampler, num_workers=2
    )

    batch = next(iter(loader))
    print(f"[smoke] image batch: {batch['image'].shape}, "
          f"range [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
    print(f"[smoke] label batch: {batch['label'].shape}, sum per channel: "
          f"{batch['label'].sum(dim=(0, 2, 3)).tolist()}")
    print(f"[smoke] has_ovary in batch: {batch['has_ovary'].tolist()}")

    unet = build_unet(cfg["model"])
    model = ConcatConditionedDDPM(unet).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] U-Net params: {n_params/1e6:.1f}M")

    train_sched = build_train_scheduler(cfg["diffusion"])
    infer_sched = build_inference_scheduler(
        cfg["diffusion"], cfg["sampling"]["num_inference_steps"]
    )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    model.train()
    for step in range(20):
        x0 = batch["image"].to(device)
        lbl = batch["label"].to(device)
        b = x0.shape[0]
        t = torch.randint(0, cfg["diffusion"]["num_train_timesteps"], (b,), device=device)
        noise = torch.randn_like(x0)
        x_t = train_sched.add_noise(original_samples=x0, noise=noise, timesteps=t)

        eps_pred = model.predict_noise(x_t, lbl, t)
        loss = F.mse_loss(eps_pred, noise)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if step % 5 == 0:
            print(f"[smoke] step {step:2d}  L_diff={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        samples = model.sample(lbl[:2], infer_sched, device, progress=True)
    print(f"[smoke] sample shape: {samples.shape}, "
          f"range [{samples.min():.3f}, {samples.max():.3f}]")
    print("[smoke] OK — pipeline is wired correctly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
