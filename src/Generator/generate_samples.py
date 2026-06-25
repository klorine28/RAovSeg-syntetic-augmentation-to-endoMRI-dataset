"""
Generate N synthetic samples for downstream review (e.g. by a radiologist).

For each sample, writes:
  synth_NNN.png       — synthetic image alone, grayscale 512×512
  overlay_NNN.png     — synthetic + organ label overlay (yellow uterus, red L-ov,
                        blue R-ov, green em), for "does the anatomy match" review
  real_NNN.png        — the real source MRI slice that the label was drawn from,
                        for direct head-to-head comparison
  label_NNN.png       — label argmax visualisation
And:
  manifest.csv        — one row per sample with source subject ID, z-slice index,
                        organ presence flags, and which file path goes with which
  README.txt          — generation settings + organ color key + viewing tips

Same `--seed` across two runs selects the SAME labels, so 1a and 1b outputs can
be compared side-by-side at matched anatomy.

Usage:
    python -m src.Generator.generate_samples \\
        --config src/Generator/exp1a.yaml \\
        --ckpt   .../runs/exp1a/ckpt/step_080000.pt \\
        --out_dir .../runs/exp1a/radiologist_review \\
        --n 50 --seed 0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from .dataset import D2SliceDataset
from .model import build_inference_scheduler, build_model_from_cfg


ORGAN_COLORS = {
    1: (255, 255, 0),    # uterus = yellow
    2: (255, 0, 0),      # L-ov = red
    3: (0, 102, 255),    # R-ov = blue
    4: (0, 255, 0),      # em = green
}
ORGAN_NAMES = {1: "uterus", 2: "ov_L", 3: "ov_R", 4: "em"}


def save_grayscale_png(arr_minus1_1: np.ndarray, path: Path) -> None:
    """Save (H, W) float in [-1, 1] as 8-bit grayscale PNG."""
    arr = np.clip((arr_minus1_1 + 1.0) / 2.0, 0.0, 1.0)
    img = (arr * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def save_overlay_png(
    synth_minus1_1: np.ndarray,
    label: np.ndarray,
    path: Path,
    alpha: float = 0.4,
) -> None:
    """Save synth-with-label-overlay as RGB PNG."""
    base = np.clip((synth_minus1_1 + 1.0) / 2.0, 0.0, 1.0)
    rgb = np.stack([base, base, base], axis=-1)  # (H, W, 3) in [0, 1]
    rgb_u8 = (rgb * 255).astype(np.float32)
    for ch, color in ORGAN_COLORS.items():
        if ch >= label.shape[0]:
            continue
        mask = label[ch] > 0
        for k in range(3):
            rgb_u8[..., k][mask] = (1 - alpha) * rgb_u8[..., k][mask] + alpha * color[k]
    Image.fromarray(np.clip(rgb_u8, 0, 255).astype(np.uint8)).save(path)


def save_label_png(label: np.ndarray, path: Path) -> None:
    """Save label argmax as an RGB PNG using the same organ colors."""
    h, w = label.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Body_other in medium gray, outside in near-black
    if label.shape[0] >= 6:
        rgb[label[5] > 0] = (128, 128, 128)
    if label.shape[0] >= 1:
        rgb[label[0] > 0] = (20, 20, 20)
    for ch, color in ORGAN_COLORS.items():
        if ch < label.shape[0]:
            rgb[label[ch] > 0] = color
    Image.fromarray(rgb).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out_dir", required=True,
                        help="Directory for synth_NNN.png + overlay/real/label/manifest")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of samples to generate (default 50)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for label selection AND initial noise. "
                             "Use the SAME seed across variants for matched-anatomy comparison.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Sampling batch size on GPU (default 16). "
                             "Training used 4 due to gradient memory; at inference "
                             "(no grads) the A100 fits much more.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Override YAML guidance scale")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override YAML DDIM step count")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use training weights instead of EMA")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay PNGs (saves disk)")
    parser.add_argument("--no-real", action="store_true",
                        help="Skip real source PNGs (saves disk)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] device={device}")
    if device.type == "cuda":
        print(f"[gen] GPU={torch.cuda.get_device_name(0)}")

    # --- Dataset (for label selection + real source images) ---
    dcfg = cfg["data"]
    ds = D2SliceDataset(
        preprocessed_root=dcfg["preprocessed_root"],
        split_file=dcfg["split_file"],
        split="train",
        sequence=dcfg["sequence"],
        num_label_channels=dcfg["num_label_channels"],
        image_size=dcfg["image_size"],
    )

    # --- Deterministic label selection by seed ---
    rng = np.random.default_rng(args.seed)
    # Pick a mix: 40% of samples must contain ovary (the rare class), 60% can be
    # anything with foreground. This gives the radiologist a clinically relevant mix.
    ovary_indices = []
    other_indices = []
    for k, si in enumerate(ds.index):
        lbl = ds.labels[si.subject][:, si.z]
        # ovary if either L-ov (ch2) or R-ov (ch3) present
        has_ovary = bool(lbl[2:4].sum() > 0)
        has_any_organ = bool(lbl[1:5].sum() > 0) if lbl.shape[0] >= 5 else bool(lbl.sum() > 0)
        if has_ovary:
            ovary_indices.append(k)
        elif has_any_organ:
            other_indices.append(k)
    n_ovary = min(int(args.n * 0.4), len(ovary_indices))
    n_other = args.n - n_ovary
    chosen_ovary = rng.choice(ovary_indices, size=n_ovary, replace=False).tolist()
    chosen_other = rng.choice(other_indices, size=min(n_other, len(other_indices)),
                              replace=False).tolist()
    chosen = chosen_ovary + chosen_other
    rng.shuffle(chosen)  # interleave so the radiologist sees mixed-anatomy mid-review
    chosen = chosen[: args.n]
    print(f"[gen] selected {len(chosen)} labels: "
          f"{n_ovary} ovary-containing + {len(chosen_other)} other-anatomy")

    # Pre-load labels + real images onto host
    label_arrays: list[np.ndarray] = []
    real_arrays: list[np.ndarray] = []
    meta: list[dict] = []
    for sample_idx, k in enumerate(chosen):
        si = ds.index[int(k)]
        lbl = ds.labels[si.subject][:, si.z]   # (C, H, W) uint8
        img = ds.images[si.subject][si.z].astype(np.float32) * 2.0 - 1.0  # [0,1] → [-1,1]
        label_arrays.append(lbl)
        real_arrays.append(img)
        has = {1: int((lbl[1] > 0).any()), 2: int((lbl[2] > 0).any()),
               3: int((lbl[3] > 0).any()), 4: int((lbl[4] > 0).any())}
        meta.append({
            "sample_idx": sample_idx, "subject": si.subject, "z": int(si.z),
            "has_uterus": has[1], "has_ov_L": has[2],
            "has_ov_R": has[3], "has_em": has[4],
        })

    # --- Model + scheduler ---
    model = build_model_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    weight_src = "EMA"
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
        weight_src = "training" if "ema" not in ckpt else "training (--no-ema)"
    model.eval()

    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None
        else cfg["sampling"]["num_inference_steps"]
    )
    scheduler = build_inference_scheduler(cfg["diffusion"], num_inference_steps)
    guidance = (
        args.guidance_scale if args.guidance_scale is not None
        else float(cfg["sampling"].get("guidance_scale", 1.0))
    )
    print(f"[gen] ckpt step {ckpt['step']}, weights={weight_src}, "
          f"guidance={guidance}, steps={num_inference_steps}")

    # --- Output paths ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Batched sampling ---
    n = len(label_arrays)
    bsz = args.batch_size
    print(f"[gen] generating {n} samples in batches of {bsz} ...")

    for b in range(0, n, bsz):
        e = min(n, b + bsz)
        batch_labels = torch.from_numpy(np.stack(label_arrays[b:e])).float().to(device)
        # Per-batch noise seeded from --seed + batch index so the same --seed across
        # variants produces the same initial noise per sample.
        g = torch.Generator(device=device).manual_seed(int(args.seed) + b)
        h, w = batch_labels.shape[-2:]
        noise = torch.randn(e - b, 1, h, w, device=device, generator=g)

        # Replicate model.sample()'s DDIM loop here so we can inject our own noise
        x = noise.clone()
        use_cfg = guidance != 1.0
        null_label = torch.zeros_like(batch_labels) if use_cfg else None
        with torch.no_grad():
            for t in scheduler.timesteps:
                t_batch = torch.full((e - b,), int(t), device=device, dtype=torch.long)
                eps_cond = model.predict_noise(x, batch_labels, t_batch)
                if use_cfg:
                    eps_uncond = model.predict_noise(x, null_label, t_batch)
                    eps = eps_uncond + guidance * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x, _ = scheduler.step(model_output=eps, timestep=int(t), sample=x)
            samples = x.clamp(-1.0, 1.0).cpu().float().numpy()

        for j in range(e - b):
            sample_idx = b + j
            synth = samples[j, 0]                   # (H, W) in [-1, 1]
            lbl = label_arrays[sample_idx]
            real = real_arrays[sample_idx]

            save_grayscale_png(synth, out_dir / f"synth_{sample_idx:03d}.png")
            if not args.no_overlay:
                save_overlay_png(synth, lbl, out_dir / f"overlay_{sample_idx:03d}.png")
            if not args.no_real:
                save_grayscale_png(real, out_dir / f"real_{sample_idx:03d}.png")
            save_label_png(lbl, out_dir / f"label_{sample_idx:03d}.png")

        print(f"  [gen] saved {e}/{n}")

    # --- Manifest CSV ---
    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_idx", "subject", "z",
            "has_uterus", "has_ov_L", "has_ov_R", "has_em",
            "synth_png", "overlay_png", "real_png", "label_png",
        ])
        w.writeheader()
        for m in meta:
            idx = m["sample_idx"]
            w.writerow({
                **m,
                "synth_png": f"synth_{idx:03d}.png",
                "overlay_png": f"overlay_{idx:03d}.png" if not args.no_overlay else "",
                "real_png": f"real_{idx:03d}.png" if not args.no_real else "",
                "label_png": f"label_{idx:03d}.png",
            })
    print(f"[gen] wrote {manifest_path}")

    # --- README ---
    readme_path = out_dir / "README.txt"
    variant = cfg["experiment"]["name"]
    cond_type = cfg["model"]["type"]
    with open(readme_path, "w") as f:
        f.write(
            f"Synthetic MRI samples — {variant}\n"
            f"=================================\n\n"
            f"Generated by src/Generator/generate_samples.py\n\n"
            f"Settings:\n"
            f"  Conditioning      : {cond_type}\n"
            f"  Checkpoint step   : {ckpt['step']}\n"
            f"  Weights           : {weight_src}\n"
            f"  Guidance scale    : {guidance}\n"
            f"  DDIM steps        : {num_inference_steps}\n"
            f"  Selection seed    : {args.seed}\n"
            f"  Samples generated : {n} "
            f"({n_ovary} ovary-containing, {n - n_ovary} other-anatomy)\n\n"
            f"File naming (NNN = 000..{n-1:03d}):\n"
            f"  synth_NNN.png    — synthetic T2FS pelvic slice (the model's output)\n"
            f"  overlay_NNN.png  — synthetic + label overlay (organ colors)\n"
            f"  real_NNN.png     — real MRI slice that the label was extracted from\n"
            f"  label_NNN.png    — label argmax visualisation\n\n"
            f"Label color key:\n"
            f"  Yellow = uterus\n"
            f"  Red    = left ovary\n"
            f"  Blue   = right ovary\n"
            f"  Green  = endometrioma\n"
            f"  Gray   = body (non-target tissue)\n"
            f"  Dark   = outside body\n\n"
            f"manifest.csv lists each sample's source subject ID, z-slice index,\n"
            f"and organ presence flags.\n\n"
            f"Cross-variant comparison:\n"
            f"  Use the SAME --seed across runs (e.g. 1a and 1b) to produce\n"
            f"  matched-anatomy samples — synth_000.png will correspond to the\n"
            f"  same label in both variants. Useful for direct head-to-head\n"
            f"  comparison by a radiologist.\n"
        )
    print(f"[gen] wrote {readme_path}")
    print(f"\n[gen] done. Output in {out_dir}")


if __name__ == "__main__":
    main()
