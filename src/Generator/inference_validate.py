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
    ConcatConditionedDDPM,
    build_inference_scheduler,
    build_unet,
)
from .train import save_sample_grid


def pick_top_n_by_foreground(dataset: D2SliceDataset, n: int) -> list[np.ndarray]:
    """Pick the n slices with the highest total foreground voxel count.

    Returns a list of (C, H, W) label arrays.
    """
    scored: list[tuple[int, np.ndarray, str, int]] = []
    for si in dataset.index:
        lbl = dataset.labels[si.subject][:, si.z]   # (C, H, W)
        fg = int(lbl[1:].sum())                     # exclude background (ch 0)
        if fg == 0:
            continue
        scored.append((fg, lbl, si.subject, si.z))
    scored.sort(key=lambda x: -x[0])
    picked = scored[:n]
    print(f"[infer] picked {len(picked)} labels by foreground voxel count:")
    for i, (fg, lbl, subj, z) in enumerate(picked):
        per_ch = {c: int((lbl[c] > 0).sum()) for c in range(lbl.shape[0])}
        print(f"  [{i}] {subj} z={z}: total_fg={fg}, per_channel={per_ch}")
    return [lbl for _, lbl, _, _ in picked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of sample slices to generate (default 4)")
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

    labels_np = pick_top_n_by_foreground(ds, args.n)
    if not labels_np:
        raise RuntimeError("No slices with any foreground voxels found in train split.")
    labels = torch.from_numpy(np.stack(labels_np)).float().to(device)

    # --- model + scheduler ---
    unet = build_unet(cfg["model"])
    model = ConcatConditionedDDPM(unet).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] loaded ckpt from step {ckpt['step']}")

    infer_sched = build_inference_scheduler(
        cfg["diffusion"], cfg["sampling"]["num_inference_steps"]
    )

    # --- sample ---
    with torch.no_grad():
        samples = model.sample(labels, infer_sched, device, progress=True)

    out_path = Path(args.out)
    save_sample_grid(samples, labels, out_path)
    print(f"[infer] wrote {out_path}")


if __name__ == "__main__":
    main()
