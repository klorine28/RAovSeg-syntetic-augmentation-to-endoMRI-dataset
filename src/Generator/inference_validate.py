"""
Post-hoc validation of the concat-conditioned DDPM.

The training script freezes 4 labels at startup and reuses them for every
visualisation grid (so progress is comparable across steps). If those 4 happen
to land on background-only slices (~24% chance with our weighted sampler),
the in-training grids look unconditioned — even though the model is fine.

This script side-steps that issue:
  - Loads a checkpoint
  - Picks the top-N slices by total foreground voxel count from the train set
    (so labels are guaranteed to carry uterus/ovary/endometrioma)
  - Runs DDIM sampling
  - Saves a sample grid identical in layout to the training script's

If the resulting overlay column lights up with yellow uterus / red L-ovary /
blue R-ovary / green endo on the synthetic image, conditioning works. If the
overlay stays blank even on these clearly-non-empty labels, there's a real
conditioning bug to chase.

Usage on Stanage (after grabbing an interactive GPU node):
    SYNTH_PY=/mnt/parscratch/users/ijp25lg/anaconda/.envs/synth_mri/bin/python
    $SYNTH_PY -m src.Generator.inference_validate \\
        --config src/Generator/exp1a.yaml \\
        --ckpt   /mnt/parscratch/users/$USER/synth_mri/runs/exp1a/ckpt/step_050000.pt \\
        --out    /mnt/parscratch/users/$USER/synth_mri/runs/exp1a/samples/validate_step_050000.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from .dataset import D2SliceDataset
from .model import (
    build_inference_scheduler,
    build_model_from_cfg,
)
from .train import save_sample_grid


def pick_top_n_by_foreground(
    dataset: D2SliceDataset, n: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pick the n slices with the highest target-organ voxel count.

    Scoring uses channels 1-4 only (uterus, L-ov, R-ov, em) — the body
    silhouette at channel 5 (if present) is excluded from the score
    because it's true everywhere inside the body and would dominate the
    ranking, defeating the purpose of picking "labels with anatomy."

    Returns:
        (labels, real_images) where:
          labels      — list of (C, H, W) one-hot label arrays
          real_images — list of (1, H, W) real MRI slices in [-1, 1] range,
                        matching the DDPM convention so save_sample_grid
                        can convert them to [0, 1] for display the same
                        way it does for synthetic samples
    """
    scored: list[tuple[int, np.ndarray, np.ndarray, str, int]] = []
    for si in dataset.index:
        lbl = dataset.labels[si.subject][:, si.z]            # (C, H, W) uint8
        # Slice [1:5] = channels 1, 2, 3, 4 (uterus, L-ov, R-ov, em).
        # Works for both 5-channel (legacy) and 6-channel (with body) layouts.
        fg = int(lbl[1:5].sum())
        if fg == 0:
            continue
        # Real image in [0,1] from dataset cache → [-1,1] for DDPM convention
        real_slice = dataset.images[si.subject][si.z]        # (H, W) float32
        real_slice = (real_slice.astype(np.float32) * 2.0 - 1.0)[None, ...]  # (1,H,W)
        scored.append((fg, lbl, real_slice, si.subject, si.z))
    scored.sort(key=lambda x: -x[0])
    picked = scored[:n]
    print(f"[infer] picked {len(picked)} labels by target-organ voxel count:")
    for i, (fg, lbl, _, subj, z) in enumerate(picked):
        per_ch = {c: int((lbl[c] > 0).sum()) for c in range(lbl.shape[0])}
        print(f"  [{i}] {subj} z={z}: target_fg={fg}, per_channel={per_ch}")
    labels = [lbl for _, lbl, _, _, _ in picked]
    reals = [img for _, _, img, _, _ in picked]
    return labels, reals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of sample slices to generate (default 4)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG guidance scale (overrides YAML; 1.0=disable, 3-5=typical)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="DDIM steps (overrides YAML; typical 50, more is smoother)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use training weights instead of EMA (default: EMA if present in ckpt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}")
    if device.type == "cuda":
        print(f"[infer] GPU={torch.cuda.get_device_name(0)}")

    # --- dataset (only to pull good labels — no actual training data is loaded
    #     onto the GPU) ---
    dcfg = cfg["data"]
    ds = D2SliceDataset(
        preprocessed_root=dcfg["preprocessed_root"],
        split_file=dcfg["split_file"],
        split="train",
        sequence=dcfg["sequence"],
        num_label_channels=dcfg["num_label_channels"],
        image_size=dcfg["image_size"],
    )

    labels_np, reals_np = pick_top_n_by_foreground(ds, args.n)
    if not labels_np:
        raise RuntimeError("No slices with any foreground voxels found in train split.")
    labels = torch.from_numpy(np.stack(labels_np)).float().to(device)
    reals = torch.from_numpy(np.stack(reals_np)).float().to(device)

    # --- model + scheduler ---
    model = build_model_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Prefer EMA weights if present in the checkpoint — they produce cleaner
    # samples. Override with --no-ema to use training weights instead.
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
        weight_source = "EMA"
    else:
        model.load_state_dict(ckpt["model"])
        weight_source = "training (no EMA in ckpt)" if "ema" not in ckpt else "training (--no-ema)"
    model.eval()
    print(f"[infer] loaded ckpt from step {ckpt['step']}, weights={weight_source}")

    num_inference_steps = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else cfg["sampling"]["num_inference_steps"]
    )
    infer_sched = build_inference_scheduler(cfg["diffusion"], num_inference_steps)
    print(f"[infer] num_inference_steps={num_inference_steps}")

    # Resolve guidance scale: CLI overrides YAML; default 1.0 (no CFG) if absent.
    if args.guidance_scale is not None:
        guidance = args.guidance_scale
    else:
        guidance = float(cfg["sampling"].get("guidance_scale", 1.0))
    print(f"[infer] guidance_scale={guidance} "
          f"({'CFG enabled' if guidance != 1.0 else 'conditional only'})")

    # --- sample ---
    with torch.no_grad():
        samples = model.sample(labels, infer_sched, device,
                               guidance_scale=guidance, progress=True)

    out_path = Path(args.out)
    save_sample_grid(samples, labels, out_path, real_images=reals)
    print(f"[infer] wrote {out_path}")


if __name__ == "__main__":
    main()
