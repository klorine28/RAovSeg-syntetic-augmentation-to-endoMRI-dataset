"""
Assemble synthetic 3D NIfTI volumes for RAovSeg augmentation.

For each of the 30 train subjects in the generator split:
  1. Load the subject's preprocessed 6-channel label volume (body-centered).
  2. Per-slice: run ISCS-coherent DDIM sampling conditioned on that slice's label.
  3. Stack the synthetic slices into a 3D NIfTI matching the source spacing.
  4. Save the synthetic image + the binarised ovary label (ov_L ∪ ov_R) in the
     UT-EndoMRI raw-input layout so RAovSeg's preprocess.py can ingest them:

       SYNTH_DIR/
         D2-901/
           D2-901_T2FS.nii.gz      ← synthetic image (this script's output)
           D2-901_ov.nii.gz        ← binary ovary mask (ov_L ∪ ov_R from gen label)
         D2-902/ ...

We use the D2-9XX naming so preprocess.py's `SUBJECT_RE = ^D2-\\d{3}$` accepts
them; combined with the new `--extra-train-dir` flag we'll add to preprocess.py,
these get forced into the train_val split without disturbing the real 30/8.

Inter-Slice Consistent Stochasticity (ISCS, Kwon & Ye ICLR 2025) is used to
keep adjacent slices coherent. Per volume:
  ε_shared ~ N(0, I) (one shared base noise tensor, used for all slices)
  per slice z: ε_z = α·ε_shared + √(1-α²)·ε_z_independent
  default α = 0.8

Usage:
    python -m src.Generator.assemble_synthetic_volumes \\
        --config src/Generator/exp1c_concat.yaml \\
        --ckpt   .../runs/exp1c_concat/ckpt/step_100000.pt \\
        --gen-preprocessed-root .../data/processed_generator/D2 \\
        --raw-data-dir .../UT-EndoMRI/D2_TCPW \\
        --gen-split-file .../data/splits/d2_generator_split.json \\
        --out-dir .../synth_volumes/exp1c_concat \\
        --iscs-alpha 0.8 \\
        --noise-seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from scipy.ndimage import gaussian_filter1d

from .model import build_inference_scheduler, build_model_from_cfg


def _gaussian_kernel_2d(sigma_pixels: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a normalised 2D Gaussian kernel for a torch.nn.functional.conv2d call.

    Kernel size is 2 * ceil(3 sigma) + 1 — enough to capture ~99.7% of the mass.
    Returned shape: (1, 1, k, k).
    """
    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be > 0 to build a Gaussian kernel")
    radius = int(math.ceil(3.0 * sigma_pixels))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma_pixels ** 2))
    g1 = g1 / g1.sum()
    g2 = g1.unsqueeze(0) * g1.unsqueeze(1)
    return g2.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)


def _gaussian_lowpass_2d(x: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    """Apply a 2D Gaussian lowpass to a (B, C, H, W) tensor. Reflect padding."""
    if sigma_pixels <= 0:
        return x
    k = _gaussian_kernel_2d(sigma_pixels, x.device, x.dtype)
    r = k.shape[-1] // 2
    return F.conv2d(F.pad(x, (r, r, r, r), mode="reflect"), k)


def _load_label_nifti(label_path: Path, num_channels: int) -> np.ndarray:
    """Load the generator's 6-channel vector NIfTI as (C, Z, H, W) uint8.

    Mirrors dataset._load_label_nifti — same logic, kept local to avoid the
    import cycle through the Generator/dataset module.
    """
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)
    if arr.ndim == 4 and arr.shape[-1] == num_channels:
        return np.transpose(arr, (3, 0, 1, 2)).astype(np.uint8)
    if arr.ndim == 4 and arr.shape[0] == num_channels:
        return arr.astype(np.uint8)
    raise ValueError(f"Unexpected label shape {arr.shape} at {label_path}")


@torch.no_grad()
def sample_volume_iscs(
    model, label_zchw: torch.Tensor, scheduler, device: torch.device,
    guidance_scale: float, iscs_alpha: float, noise_seed: int,
    iscs_lowpass_sigma: float = 0.0,
) -> torch.Tensor:
    """ISCS-coherent slice-by-slice DDIM sampling for an entire volume.

    Args:
        label_zchw: (Z, C, H, W) one-hot label volume on device.
        iscs_alpha: ISCS coherence factor in [0, 1]. 0 = fully independent slices,
                    1 = fully shared noise (all slices identical given same label).
                    0.8 is the paper default.
        iscs_lowpass_sigma: Gaussian lowpass std (in pixels) applied to the
                    shared noise ONLY. 0 = original single-scale ISCS
                    (default). > 0 = multi-scale ISCS — α applies to the
                    low-frequency (structural) component only; high-frequency
                    (texture) is always per-slice-independent. Independent
                    noise is variance-corrected so the combined initial
                    latent still has unit variance in expectation. Typical
                    values: 2-8 pixels.

    Returns:
        (Z, 1, H, W) synthetic image volume on CPU, values in [-1, 1].
    """
    Z, C, H, W = label_zchw.shape

    # Shared noise base for this volume — coherence across slices
    g_shared = torch.Generator(device=device).manual_seed(noise_seed)
    eps_shared = torch.randn(1, 1, H, W, device=device, generator=g_shared)

    alpha = float(iscs_alpha)
    coeff_shared = alpha
    coeff_indep = math.sqrt(max(0.0, 1.0 - alpha * alpha))

    # Multi-scale ISCS: replace shared noise with its lowpass so the shared
    # component carries only structural (low-frequency) content. Rescale to
    # preserve unit variance in the shared component (lowpass reduces variance
    # by the sum-of-squared-kernel factor).
    if iscs_lowpass_sigma > 0.0:
        eps_shared_lp = _gaussian_lowpass_2d(eps_shared, iscs_lowpass_sigma)
        # Empirical variance restoration so the initial latent stays ~N(0, 1)
        eps_shared = eps_shared_lp / (eps_shared_lp.std().clamp_min(1e-6))

    use_cfg = guidance_scale != 1.0

    out = torch.empty(Z, 1, H, W, device="cpu", dtype=torch.float32)

    for z in range(Z):
        # Per-slice independent noise (seeded so reruns reproduce)
        g_indep = torch.Generator(device=device).manual_seed(noise_seed * 100003 + z)
        eps_indep = torch.randn(1, 1, H, W, device=device, generator=g_indep)

        # ISCS initial latent
        x = coeff_shared * eps_shared + coeff_indep * eps_indep

        label = label_zchw[z:z + 1].float()
        null_label = torch.zeros_like(label) if use_cfg else None

        for t in scheduler.timesteps:
            t_batch = torch.full((1,), int(t), device=device, dtype=torch.long)
            eps_cond = model.predict_noise(x, label, t_batch)
            if use_cfg:
                eps_uncond = model.predict_noise(x, null_label, t_batch)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = eps_cond
            x, _ = scheduler.step(model_output=eps_pred, timestep=int(t), sample=x)

        out[z] = x.clamp(-1.0, 1.0).cpu().float()[0]
    return out


def _label_aware_ovary_rescale(
    synth_arr_raw: np.ndarray,           # synth in raw intensity range
    ovary_mask: np.ndarray,              # (Z, H, W) bool — where ovary is
    raw_p1: float, raw_p99: float,       # source real's percentile range
    target_normalized: float = 0.26,     # middle of RAovSeg's [0.22, 0.30] window
) -> np.ndarray:
    """Apply a per-volume additive offset to the ovary region only, so that
    the synth ovary pixels land at `target_normalized` after RAovSeg's
    percentile-clip + min-max normalization.

    RAovSeg's normalization is: clip to [p1, p99] then minmax → [0, 1].
    So the post-normalization value of a raw intensity X is approximately
    (X - p1) / (p99 - p1). For ovary to land at `target_normalized`, we
    need ovary_raw = p1 + target_normalized × (p99 − p1).

    Histogram matching (rank-based) doesn't guarantee this — it aligns
    distributions but doesn't guarantee that the labelled ovary pixels
    are at the right rank. This function uses the explicit label to push
    the ovary tissue to the right raw intensity.

    Side note: this changes the ovary region's local appearance (it gets
    brighter to land in the enhancement window). Since the ovary is a
    small fraction of total pixels, p1/p99 of the whole image barely
    shift, so the rest of the body keeps its post-normalization
    distribution.
    """
    if not ovary_mask.any():
        return synth_arr_raw  # No ovary in this volume — nothing to do

    target_raw = float(target_normalized) * (raw_p99 - raw_p1) + raw_p1
    current_ovary_mean = float(synth_arr_raw[ovary_mask].mean())
    offset = target_raw - current_ovary_mean

    synth_arr_raw = synth_arr_raw.copy()
    synth_arr_raw[ovary_mask] = synth_arr_raw[ovary_mask] + offset
    return synth_arr_raw


def _histogram_match(source_01: np.ndarray, reference_01: np.ndarray) -> np.ndarray:
    """Histogram-match `source` to `reference`. Both arrays expected in [0, 1].

    Standard sorted-CDF approach: for each unique value in source, find the
    reference value with the matching cumulative probability. Pure numpy,
    no skimage dependency.

    Returns array same shape as source, but with the intensity *distribution*
    of reference.
    """
    src_flat = source_01.ravel()
    ref_flat = reference_01.ravel()

    src_values, src_inverse, src_counts = np.unique(
        src_flat, return_inverse=True, return_counts=True
    )
    ref_values, ref_counts = np.unique(ref_flat, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64) / src_flat.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / ref_flat.size

    # For each source CDF value, find the reference value at that CDF
    interp = np.interp(src_cdf, ref_cdf, ref_values)
    matched = interp[src_inverse].reshape(source_01.shape).astype(np.float32)
    return matched


def save_synth_subject_nifti(
    synth_zchw: torch.Tensor,                # (Z, 1, H, W) in [-1, 1]
    label_zchw: np.ndarray,                  # (C, Z, H, W) one-hot, body-centered frame
    source_image_sitk: sitk.Image,           # generator's preprocessed (body-centered) image
    out_dir: Path,
    subject_id: str,
    *,
    raw_real_path: Path | None = None,       # raw real subject T2FS (for resample + hist match)
    raw_real_ov_path: Path | None = None,    # raw real subject ovary label (for label resample reference)
    apply_body_mask: bool = True,            # zero synth outside body silhouette
    histogram_match: bool = True,            # match synth intensity dist to raw real
    resample_to_source: bool = True,         # resample synth NIfTI into raw real subject's frame
    ovary_target_intensity: float | None = 0.26,  # Path B: explicit ovary rescale to target post-norm intensity
) -> None:
    """Write synthetic subject directory in UT-EndoMRI raw-input layout, with
    optional preprocessing fixes:

        out_dir/subject_id/
          subject_id_T2FS.nii.gz   ← synthetic image, float32
          subject_id_ov.nii.gz     ← binary ovary mask, uint8

    Three fixes applied by default (each individually toggleable):

    1. **apply_body_mask** — zero out the synth outside the body silhouette
       (kills outside-body hallucinations that survive RAovSeg's normalization).

    2. **histogram_match** — match the synth's intensity distribution to the
       raw real subject's. After RAovSeg's percentile-clip + min-max, the
       synth ends up at the same intensity profile as real, so the o1=0.22 /
       o2=0.30 ovary enhancement step fires consistently for both.

    3. **resample_to_source** — resample the synth NIfTI from the generator's
       body-centered frame (per-subject spacing, body fills 90% of 512²)
       into the raw real subject's frame (image-centered, body fills ~60%
       of frame). Kills the FOV mismatch between train (mixed) and test (real).

    When (3) is enabled, the saved synth NIfTI has the raw real subject's
    spacing/origin/direction — so RAovSeg's preprocess.py produces output
    indistinguishable from a real subject in framing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subj_dir = out_dir / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Synth image: [-1, 1] → [0, 1], shape (Z, H, W) -----
    img_arr = ((synth_zchw + 1.0) / 2.0).clamp(0.0, 1.0).squeeze(1).numpy().astype(np.float32)

    # ----- 2) Apply body silhouette mask (kill outside-body hallucinations) -----
    if apply_body_mask:
        # label channel 0 = outside_body (1 where outside body)
        outside_body = label_zchw[0] > 0   # (Z, H, W) bool
        # Set outside-body pixels to 0 (will be dark after RAovSeg normalization)
        img_arr[outside_body] = 0.0

    # ----- 3) Build the binary ovary label (always needed) -----
    if label_zchw.shape[0] >= 4:
        ov_arr = ((label_zchw[2] > 0) | (label_zchw[3] > 0)).astype(np.uint8)
    elif label_zchw.shape[0] >= 3:
        ov_arr = (label_zchw[2] > 0).astype(np.uint8)
    else:
        raise ValueError(f"Label has {label_zchw.shape[0]} channels; need >=4")

    # ----- 4) Load raw real subject if we need it (for hist-match or resample) -----
    raw_real_sitk = None
    raw_real_arr = None
    raw_p1 = raw_p99 = None
    if (histogram_match or resample_to_source) and raw_real_path is not None:
        if not raw_real_path.exists():
            raise FileNotFoundError(f"Raw real subject T2FS not found: {raw_real_path}")
        raw_real_sitk = sitk.ReadImage(str(raw_real_path), sitk.sitkFloat32)
        raw_real_arr = sitk.GetArrayFromImage(raw_real_sitk).astype(np.float32)
        raw_p1, raw_p99 = np.percentile(raw_real_arr, [1, 99])
    elif histogram_match or resample_to_source:
        raise ValueError("histogram_match and resample_to_source require --raw-data-dir to be supplied")

    # ----- 5) Histogram-match synth (in [0, 1]) to raw real's [0, 1] distribution -----
    if histogram_match:
        # Reference = raw real after percentile clip + minmax (mirrors what RAovSeg will compute)
        real_01 = np.clip(raw_real_arr, raw_p1, raw_p99)
        real_01 = (real_01 - raw_p1) / max(raw_p99 - raw_p1, 1e-6)
        img_arr = _histogram_match(img_arr, real_01)

        # Map back to the raw real's intensity range, so when RAovSeg's preprocess
        # re-applies percentile-clip + minmax it gives an identical [0, 1] distribution.
        img_arr_raw = img_arr * (raw_p99 - raw_p1) + raw_p1
    else:
        # No matching: just save in [0, 1] range (RAovSeg's preprocess will normalize anyway)
        img_arr_raw = img_arr

    # ----- 5b) Path B: label-aware ovary rescaling -----
    # After histogram matching the overall distribution shape, explicitly push
    # the labelled ovary tissue to the intensity that lands inside RAovSeg's
    # [0.22, 0.30] enhancement window. Fixes the rank-based-match-doesn't-
    # know-which-pixels-are-ovary problem.
    if ovary_target_intensity is not None and raw_p1 is not None and raw_p99 is not None:
        if label_zchw.shape[0] >= 4:
            ovary_mask = (label_zchw[2] > 0) | (label_zchw[3] > 0)   # (Z, H, W) bool
        elif label_zchw.shape[0] >= 3:
            ovary_mask = label_zchw[2] > 0
        else:
            ovary_mask = np.zeros(img_arr_raw.shape, dtype=bool)
        img_arr_raw = _label_aware_ovary_rescale(
            img_arr_raw, ovary_mask,
            raw_p1=raw_p1, raw_p99=raw_p99,
            target_normalized=float(ovary_target_intensity),
        )

    # ----- 6) Build SimpleITK images in the SOURCE preprocessed (body-centered) frame -----
    img_sitk = sitk.GetImageFromArray(img_arr_raw)
    img_sitk.CopyInformation(source_image_sitk)

    ov_sitk = sitk.GetImageFromArray(ov_arr)
    ov_sitk.CopyInformation(source_image_sitk)

    # ----- 7) Resample to the raw real subject's frame if requested -----
    if resample_to_source:
        # Image: linear interpolation, background = raw_p1 (matches real outside-body intensity)
        img_resampler = sitk.ResampleImageFilter()
        img_resampler.SetReferenceImage(raw_real_sitk)
        img_resampler.SetInterpolator(sitk.sitkLinear)
        img_resampler.SetDefaultPixelValue(float(raw_p1))
        img_sitk = img_resampler.Execute(img_sitk)

        # Label: nearest-neighbour to preserve binary values
        ov_resampler = sitk.ResampleImageFilter()
        ov_resampler.SetReferenceImage(raw_real_sitk)
        ov_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        ov_resampler.SetDefaultPixelValue(0)
        ov_sitk = ov_resampler.Execute(ov_sitk)
        # Ensure dtype stays uint8 after resample
        ov_sitk = sitk.Cast(ov_sitk, sitk.sitkUInt8)

    # ----- 8) Save -----
    sitk.WriteImage(img_sitk, str(subj_dir / f"{subject_id}_T2FS.nii.gz"))
    sitk.WriteImage(ov_sitk, str(subj_dir / f"{subject_id}_ov.nii.gz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--config", required=True, help="Generator training YAML")
    parser.add_argument("--ckpt", required=True, help="Generator checkpoint .pt")
    parser.add_argument("--gen-preprocessed-root", required=True,
                        help="Path to data/processed_generator/D2 (the generator's preprocessed input)")
    parser.add_argument("--gen-split-file", required=True,
                        help="Path to data/splits/d2_generator_split.json")
    parser.add_argument("--out-dir", required=True,
                        help="Output dir for synth NIfTI subjects (e.g. synth_volumes/exp1c_concat)")
    parser.add_argument("--raw-data-dir", default=None,
                        help="Path to UT-EndoMRI/D2_TCPW (raw real subjects). Required when "
                             "--histogram-match or --resample-to-source is enabled.")
    parser.add_argument("--no-body-mask", action="store_true",
                        help="Skip the body-silhouette mask step (synth outside-body NOT zeroed)")
    parser.add_argument("--no-histogram-match", action="store_true",
                        help="Skip histogram matching to the raw real subject's intensity distribution")
    parser.add_argument("--no-resample-to-source", action="store_true",
                        help="Skip resampling synth NIfTI into the raw real subject's frame")
    parser.add_argument("--ovary-target-intensity", type=float, default=0.26,
                        help="Path B: rescale synth ovary pixels so they land at this "
                             "post-normalization intensity (default 0.26 = middle of "
                             "RAovSeg's [0.22, 0.30] enhancement window). Set to negative "
                             "to disable, e.g. --ovary-target-intensity -1.")
    parser.add_argument("--subject-id-prefix", default="D2-9",
                        help="Prefix for synth subject IDs. Combined with a 2-digit index → D2-901, D2-902, …")
    parser.add_argument("--iscs-alpha", type=float, default=0.8,
                        help="ISCS coherence factor in [0,1] (default 0.8 per paper)")
    parser.add_argument("--iscs-lowpass-sigma", type=float, default=0.0,
                        help="Gaussian lowpass sigma (pixels) on shared noise. "
                             "0 = single-scale ISCS (default). >0 = multi-scale ISCS "
                             "where alpha applies to structural (low-freq) noise only; "
                             "texture (high-freq) is per-slice independent. Typical: 2-8.")
    parser.add_argument("--z-smooth-sigma", type=float, default=0.0,
                        help="Post-hoc Gaussian smoothing sigma (in units of slices) "
                             "applied along z after DDIM sampling but before all other "
                             "post-processing. 0 = no smoothing (default). Typical: "
                             "0.3-0.8 for light coherence, up to 1.5 for heavy.")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="RNG seed for the shared noise base + per-slice noise")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="Override YAML guidance scale (default uses YAML)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override YAML DDIM steps (default uses YAML)")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[assemble] device={device}")
    if device.type == "cuda":
        print(f"[assemble] GPU={torch.cuda.get_device_name(0)}")

    # --- Read split: only train subjects get synthetic volumes (never test) ---
    with open(args.gen_split_file) as f:
        split = json.load(f)
    train_subjects = split.get("train", []) or split.get("train_val", [])
    if not train_subjects:
        raise RuntimeError(f"No train subjects found in {args.gen_split_file}")
    print(f"[assemble] {len(train_subjects)} train subjects from generator split")

    # --- Model + scheduler ---
    model = build_model_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
        weight_src = "EMA"
    else:
        model.load_state_dict(ckpt["model"])
        weight_src = "training" if "ema" not in ckpt else "training (--no-ema)"
    model.eval()
    print(f"[assemble] loaded ckpt step {ckpt['step']}, weights={weight_src}")

    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None
        else cfg["sampling"]["num_inference_steps"]
    )
    scheduler = build_inference_scheduler(cfg["diffusion"], num_inference_steps)
    guidance = (
        args.guidance_scale if args.guidance_scale is not None
        else float(cfg["sampling"].get("guidance_scale", 1.0))
    )
    print(f"[assemble] guidance={guidance}, steps={num_inference_steps}, "
          f"iscs_alpha={args.iscs_alpha}, "
          f"iscs_lowpass_sigma={args.iscs_lowpass_sigma}, "
          f"z_smooth_sigma={args.z_smooth_sigma}")

    # --- Output ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_root = Path(args.gen_preprocessed_root)
    raw_root = Path(args.raw_data_dir) if args.raw_data_dir else None
    # Phase 2 configs put data under `data.generator` / `data.discriminator`;
    # Phase 1 puts fields at top level. resolve_gen_data_cfg picks whichever.
    from .model import resolve_gen_data_cfg
    gen_dcfg = resolve_gen_data_cfg(cfg)
    num_label_channels = int(gen_dcfg["num_label_channels"])
    gen_sequence = str(gen_dcfg["sequence"])
    manifest_rows = []

    apply_body_mask = not args.no_body_mask
    histogram_match = not args.no_histogram_match
    resample_to_source = not args.no_resample_to_source
    ovary_target = float(args.ovary_target_intensity) if args.ovary_target_intensity >= 0 else None
    if (histogram_match or resample_to_source or ovary_target is not None) and raw_root is None:
        raise ValueError("--raw-data-dir is required when histogram matching / resampling / "
                         "ovary rescaling is enabled (defaults all ON).")
    print(f"[assemble] fixes — body_mask={apply_body_mask}, "
          f"histogram_match={histogram_match}, resample_to_source={resample_to_source}, "
          f"ovary_target_intensity={ovary_target}")

    # --- Iterate train subjects ---
    for i, src_subj in enumerate(train_subjects):
        synth_subj = f"{args.subject_id_prefix}{i:02d}"     # D2-900, D2-901, …
        print(f"\n[assemble] [{i+1}/{len(train_subjects)}] {src_subj} → {synth_subj}")

        # Load source preprocessed image (for spacing) + label
        src_image_path = gen_root / src_subj / f"image_{gen_sequence}.nii.gz"
        src_label_path = gen_root / src_subj / f"label_{gen_sequence}.nii.gz"
        if not src_image_path.exists() or not src_label_path.exists():
            print(f"  SKIP — missing preprocessed files at {gen_root / src_subj}")
            continue

        src_image_sitk = sitk.ReadImage(str(src_image_path))
        label_czhw = _load_label_nifti(src_label_path, num_label_channels)   # (C, Z, H, W)
        label_zchw = torch.from_numpy(np.transpose(label_czhw, (1, 0, 2, 3))).to(device)
        Z = label_zchw.shape[0]
        print(f"  Z={Z}, H×W={label_zchw.shape[-2]}×{label_zchw.shape[-1]}, "
              f"spacing={src_image_sitk.GetSpacing()}")

        # ISCS-coherent sampling
        synth_zchw = sample_volume_iscs(
            model, label_zchw, scheduler, device,
            guidance_scale=guidance, iscs_alpha=args.iscs_alpha,
            noise_seed=int(args.noise_seed),
            iscs_lowpass_sigma=float(args.iscs_lowpass_sigma),
        )

        # Post-hoc z-Gaussian smoothing (softens residual inter-slice jitter).
        # Applied on the (Z, 1, H, W) tensor in [-1, 1] before all other
        # post-processing so downstream steps see the smoothed volume.
        if args.z_smooth_sigma > 0.0:
            arr = synth_zchw.numpy()  # (Z, 1, H, W)
            arr = gaussian_filter1d(arr, sigma=float(args.z_smooth_sigma), axis=0, mode="reflect")
            synth_zchw = torch.from_numpy(arr).clamp(-1.0, 1.0)

        # Locate raw real subject (if fixes are enabled)
        raw_real_path = None
        if raw_root is not None:
            candidate = raw_root / src_subj / f"{src_subj}_{gen_sequence}.nii.gz"
            if candidate.exists():
                raw_real_path = candidate
            else:
                print(f"  WARNING — raw real subject not found at {candidate}; "
                      f"fixes that need it will be skipped for this subject")

        # Save in UT-EndoMRI raw-input layout with the v2 fixes + Path B ovary rescale
        save_synth_subject_nifti(
            synth_zchw=synth_zchw,
            label_zchw=label_czhw,
            source_image_sitk=src_image_sitk,
            out_dir=out_dir,
            subject_id=synth_subj,
            raw_real_path=raw_real_path,
            apply_body_mask=apply_body_mask,
            histogram_match=histogram_match and raw_real_path is not None,
            resample_to_source=resample_to_source and raw_real_path is not None,
            ovary_target_intensity=(ovary_target if raw_real_path is not None else None),
        )
        manifest_rows.append({
            "synth_subject_id": synth_subj,
            "source_subject_id": src_subj,
            "z_slices": int(Z),
            "iscs_alpha": float(args.iscs_alpha),
            "iscs_lowpass_sigma": float(args.iscs_lowpass_sigma),
            "z_smooth_sigma": float(args.z_smooth_sigma),
            "noise_seed": int(args.noise_seed),
            "guidance_scale": float(guidance),
            "num_inference_steps": int(num_inference_steps),
        })
        print(f"  wrote {out_dir / synth_subj}")

    # Manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "config": str(args.config),
            "ckpt": str(args.ckpt),
            "ckpt_step": int(ckpt["step"]),
            "weights": weight_src,
            "variant": cfg["experiment"]["name"],
            "n_subjects": len(manifest_rows),
            "subjects": manifest_rows,
        }, f, indent=2)
    print(f"\n[assemble] wrote {manifest_path}")
    print(f"[assemble] done — {len(manifest_rows)} synth volumes in {out_dir}")


if __name__ == "__main__":
    main()
