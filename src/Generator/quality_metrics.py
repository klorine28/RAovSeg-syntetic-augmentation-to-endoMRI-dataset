"""
Quality metrics for synthetic MRI: FID, LPIPS-NN, intensity histogram divergence.

Category 2 from TIER1_TUNING_AND_EXPLAINABILITY.md §11. These answer "is the
synthetic data realistic" rather than "what is the model doing." Standard
literature metrics — compared across variants (1a, 1b, 1c_concat, 1c_spade)
they give a quantitative ranking.

Outputs a single JSON per run:
{
    "variant": "exp1a",
    "ckpt_step": 80000,
    "n_synth": 256, "n_real": 200,
    "fid": 38.4,
    "lpips_nn": {"min": 0.18, "mean": 0.31, "max": 0.55},
    "hist_kl": 0.043,
    "guidance_scale": 3.0,
    "num_inference_steps": 100
}

Dependencies (install on Stanage env):
    pip install pytorch-fid lpips

Usage:
    python -m src.Generator.quality_metrics \\
        --config src/Generator/exp1a.yaml \\
        --ckpt   /mnt/.../runs/exp1a/ckpt/step_080000.pt \\
        --out_json /mnt/.../runs/exp1a/quality.json \\
        --n 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dataset import D2SliceDataset
from .model import build_inference_scheduler, build_model_from_cfg


def _save_images_to_dir(tensor: torch.Tensor, out_dir: Path) -> None:
    """Save (N, 1, H, W) tensor in [-1, 1] as N PNG files for pytorch-fid.

    pytorch-fid reads images from disk via its own ImageFolderDataset. We
    convert to uint8 RGB (replicating the grayscale channel) since Inception
    expects 3 channels.
    """
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = ((tensor.cpu().float() + 1.0) / 2.0).clamp(0, 1).numpy()  # [0,1]
    arr = (arr * 255).astype(np.uint8)
    for i in range(arr.shape[0]):
        img = arr[i, 0]  # (H, W)
        rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        Image.fromarray(rgb).save(out_dir / f"{i:05d}.png")


@torch.no_grad()
def generate_synth_samples(
    model, ds, n: int, scheduler, device, guidance_scale: float,
    batch_size: int = 16,
) -> torch.Tensor:
    """Generate n synthetic samples using random labels drawn from the train
    set. Returns (n, 1, H, W) in [-1, 1]. Default batch_size=16 at inference;
    training used 4 only because of gradient memory. On an A100 80GB the
    UNet at batch 16 at 512² uses ~6 GiB."""
    # Sample n random label indices from the dataset
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds.index), size=n, replace=(n > len(ds.index)))
    labels = []
    for k in indices:
        si = ds.index[int(k)]
        lbl = ds.labels[si.subject][:, si.z]
        labels.append(torch.from_numpy(lbl).float())
    labels = torch.stack(labels, dim=0).to(device)

    out = []
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min(n, (b + 1) * batch_size)
        batch_lbl = labels[s:e]
        samples = model.sample(batch_lbl, scheduler, device,
                               guidance_scale=guidance_scale, progress=False)
        out.append(samples.cpu())
        if (b + 1) % 16 == 0 or (b + 1) == n_batches:
            print(f"  [synth] generated {e}/{n}")
    return torch.cat(out, dim=0)


def collect_real_samples(ds, n: int, device) -> torch.Tensor:
    """Pick n random real T2FS slices from the train set. Returns (n, 1, H, W)
    in [-1, 1] (DDPM convention)."""
    rng = np.random.default_rng(1)
    indices = rng.choice(len(ds.index), size=n, replace=(n > len(ds.index)))
    out = []
    for k in indices:
        si = ds.index[int(k)]
        img = ds.images[si.subject][si.z].astype(np.float32)
        img = img * 2.0 - 1.0  # [0,1] → [-1,1]
        out.append(torch.from_numpy(img)[None, None])
    return torch.cat(out, dim=0).to(device)


def compute_fid(synth: torch.Tensor, real: torch.Tensor, work_dir: Path) -> float:
    """FID via pytorch-fid CLI (writes images to disk, runs InceptionV3)."""
    from pytorch_fid.fid_score import calculate_fid_given_paths

    synth_dir = work_dir / "fid_synth"
    real_dir = work_dir / "fid_real"
    _save_images_to_dir(synth, synth_dir)
    _save_images_to_dir(real, real_dir)

    fid = calculate_fid_given_paths(
        [str(real_dir), str(synth_dir)],
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims=2048,
        num_workers=2,
    )
    return float(fid)


def compute_lpips_nn(
    synth: torch.Tensor, real: torch.Tensor, device, batch_size: int = 32,
) -> dict[str, float]:
    """LPIPS nearest-neighbour distance for each synth → min over real pool.
    Returns {min, mean, max, p10, p90}.

    Memorisation check: very low values (e.g. < 0.05) indicate the model has
    copied real images. Mid-range values (0.2-0.5) are healthy.

    Performance: O(N_synth × N_real) AlexNet forwards. Move both pools to GPU
    once, then process synth in larger batches. Default batch_size=32 (was 8);
    on A100 batch=32 AlexNet pairs use ~1 GiB."""
    import lpips
    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    def to_rgb(x: torch.Tensor) -> torch.Tensor:
        # LPIPS expects 3-channel input; replicate grayscale.
        return x.expand(x.shape[0], 3, x.shape[2], x.shape[3])

    # Move BOTH pools to device once. Was: synth on CPU + per-i .to(device)
    # transfer for each of N_synth outer iterations = N_synth redundant copies.
    synth_dev = synth.to(device)
    real_dev = real.to(device)
    synth_rgb = to_rgb(synth_dev)
    real_rgb = to_rgb(real_dev)
    n_synth = synth_rgb.shape[0]
    n_real = real_rgb.shape[0]

    nn_dists = []
    with torch.no_grad():
        for i in range(n_synth):
            s_i = synth_rgb[i:i + 1]   # already on device
            # Process the real pool in batches of `batch_size`.
            # Each batch tiles the synth image to match the real batch shape.
            row_min = torch.tensor(float("inf"), device=device)
            for b in range(0, n_real, batch_size):
                e = min(n_real, b + batch_size)
                s_tile = s_i.expand(e - b, -1, -1, -1)
                d = loss_fn(s_tile, real_rgb[b:e]).view(-1)
                row_min = torch.minimum(row_min, d.min())
            nn_dists.append(float(row_min.item()))
            if (i + 1) % 32 == 0:
                print(f"  [lpips] {i+1}/{n_synth} done")

    arr = np.array(nn_dists)
    return {
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "mean": float(arr.mean()),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def compute_hist_kl(synth: torch.Tensor, real: torch.Tensor,
                     n_bins: int = 100) -> float:
    """Intensity histogram KL divergence between synth and real. Both
    tensors in [-1, 1]; histograms computed on [0, 1] after rescaling."""
    s = ((synth + 1.0) / 2.0).cpu().numpy().flatten()
    r = ((real + 1.0) / 2.0).cpu().numpy().flatten()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    p_s, _ = np.histogram(s, bins=edges, density=True)
    p_r, _ = np.histogram(r, bins=edges, density=True)
    p_s = p_s + 1e-10
    p_r = p_r + 1e-10
    p_s /= p_s.sum()
    p_r /= p_r.sum()
    return float(np.sum(p_r * np.log(p_r / p_s)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--n", type=int, default=256,
                        help="Number of synthetic samples to generate (default 256)")
    parser.add_argument("--n-real", type=int, default=200,
                        help="Number of real samples to compare against (default 200)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size for DDIM sampling (default 16). "
                             "Training used 4 due to gradient memory; at inference "
                             "the A100 fits much more.")
    parser.add_argument("--lpips-batch-size", type=int, default=32,
                        help="Batch size for LPIPS pairwise computation (default 32)")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--work_dir", default=None,
                        help="Scratch dir for FID image dumps (default: out_json's parent /fid_work)")
    parser.add_argument("--skip-lpips", action="store_true",
                        help="Skip LPIPS-NN (slow on large pools)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[quality] device={device}")

    dcfg = cfg["data"]
    ds = D2SliceDataset(
        preprocessed_root=dcfg["preprocessed_root"],
        split_file=dcfg["split_file"],
        split="train",
        sequence=dcfg["sequence"],
        num_label_channels=dcfg["num_label_channels"],
        image_size=dcfg["image_size"],
    )

    model = build_model_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    weights_src = "EMA"
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
        weights_src = "training" if "ema" not in ckpt else "training (--no-ema)"
    model.eval()
    print(f"[quality] loaded ckpt step {ckpt['step']}, weights={weights_src}")

    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None
        else cfg["sampling"]["num_inference_steps"]
    )
    scheduler = build_inference_scheduler(cfg["diffusion"], num_inference_steps)
    guidance = (
        args.guidance_scale if args.guidance_scale is not None
        else float(cfg["sampling"].get("guidance_scale", 1.0))
    )
    print(f"[quality] guidance={guidance}, steps={num_inference_steps}, "
          f"n_synth={args.n}, n_real={args.n_real}, "
          f"sample_batch={args.batch_size}, lpips_batch={args.lpips_batch_size}")

    print("\n[quality] generating synthetic samples ...")
    synth = generate_synth_samples(model, ds, args.n, scheduler, device,
                                    guidance_scale=guidance,
                                    batch_size=args.batch_size)
    print(f"[quality] synth shape: {tuple(synth.shape)}")

    print("\n[quality] collecting real samples ...")
    real = collect_real_samples(ds, args.n_real, device).cpu()
    print(f"[quality] real shape: {tuple(real.shape)}")

    out_json = Path(args.out_json)
    work_dir = Path(args.work_dir) if args.work_dir else out_json.parent / "fid_work"

    print("\n[quality] computing FID ...")
    fid = compute_fid(synth, real, work_dir)
    print(f"  FID = {fid:.3f}")

    print("\n[quality] computing intensity histogram KL ...")
    hist_kl = compute_hist_kl(synth, real)
    print(f"  hist_KL = {hist_kl:.4f}")

    lpips_nn = None
    if not args.skip_lpips:
        print("\n[quality] computing LPIPS-NN ...")
        lpips_nn = compute_lpips_nn(synth, real, device,
                                     batch_size=args.lpips_batch_size)
        print(f"  LPIPS-NN min={lpips_nn['min']:.3f} mean={lpips_nn['mean']:.3f} max={lpips_nn['max']:.3f}")

    result = {
        "variant": cfg["experiment"]["name"],
        "ckpt_step": int(ckpt["step"]),
        "weights": weights_src,
        "n_synth": int(args.n),
        "n_real": int(args.n_real),
        "guidance_scale": guidance,
        "num_inference_steps": num_inference_steps,
        "fid": fid,
        "hist_kl": hist_kl,
        "lpips_nn": lpips_nn,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[quality] wrote {out_json}")


if __name__ == "__main__":
    main()
