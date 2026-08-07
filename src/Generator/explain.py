"""
Explainability for the conditional DDPM generators (Exp 1a concat, Exp 1b SPADE).

Produces one multi-panel PNG per sample combining five different views of
"what is the model doing":

  1. Attention-like activation magnitude map at the deepest U-Net level —
     hooked from `mid_block_2` (1b) or `middle_block` (1a, MONAI). Averaged
     across denoising timesteps. Answers "where is the model carrying signal
     at its most abstract layer."

  2. GradientSHAP attribution per label channel — manual implementation of
     SHAP's gradient-based path integral (no Captum dep). Target: predicted
     noise magnitude at a single mid-range timestep. Baseline: zero label
     (matches the CFG null pathway). Answers "which label pixels drove the
     output, per channel." Works on both 1a and 1b → apples-to-apples.

  3. SPADE γ magnitude maps (1b only) — per-module |γ| averaged across
     decoder levels and timesteps. Native SPADE interpretation of "where
     each layer thinks the label is most important." 1a has no comparable
     mechanism; that asymmetry IS the point.

  4. Counterfactual label ablation — zero each organ channel in turn, re-sample
     with the same initial noise, and compare. Answers "is the model actually
     using all four organ channels, or does the output collapse to one
     conditioning signal?"

  5. Per-timestep snapshots — capture x_t at equally-spaced points through
     the denoising chain. Tells the "body silhouette first, then organ
     positions, then texture" story visually.

The CLI mirrors `inference_validate.py` for consistency.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from .dataset import D2SliceDataset
from .inference_validate import pick_top_n_by_foreground
from .model import (
    SPADEConditionedDDPM,
    build_inference_scheduler,
    build_model_from_cfg,
)
from .spade import SPADE


LABEL_NAMES = ["outside_body", "uterus", "ov_L", "ov_R", "em", "body_other"]
ORGAN_CHANNELS = (1, 2, 3, 4)
ORGAN_NAMES = {1: "uterus", 2: "ov_L", 3: "ov_R", 4: "em"}
ORGAN_COLORS = {
    1: (1.0, 1.0, 0.0),   # uterus = yellow
    2: (1.0, 0.0, 0.0),   # L ovary = red
    3: (0.0, 0.4, 1.0),   # R ovary = blue
    4: (0.0, 1.0, 0.0),   # endometrioma = green
}


def is_spade_model(model) -> bool:
    return isinstance(model, SPADEConditionedDDPM)


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #
def to_disp(image_minus1_1: np.ndarray) -> np.ndarray:
    """Convert (..., H, W) image in [-1, 1] to display range [0, 1]."""
    return np.clip((image_minus1_1 + 1.0) / 2.0, 0.0, 1.0)


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """(C, H, W) one-hot → (H, W, 3) display RGB. Organs over body_other
    over outside_body."""
    h, w = label.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    if label.shape[0] >= 6:
        rgb[label[5] > 0] = (0.5, 0.5, 0.5)
    if label.shape[0] >= 1:
        rgb[label[0] > 0] = (0.08, 0.08, 0.08)
    for ch, color in ORGAN_COLORS.items():
        if ch < label.shape[0]:
            rgb[label[ch] > 0] = color
    return rgb


def overlay_label_on_image(image: np.ndarray, label: np.ndarray,
                            alpha: float = 0.45) -> np.ndarray:
    """image: (H, W); label: (C, H, W). Returns (H, W, 3) overlay."""
    img = to_disp(image) if image.min() < 0 else np.clip(image, 0, 1)
    rgb = np.stack([img, img, img], axis=-1)
    for ch, color in ORGAN_COLORS.items():
        if ch < label.shape[0]:
            mask = label[ch] > 0
            for k in range(3):
                rgb[..., k][mask] = (1 - alpha) * rgb[..., k][mask] + alpha * color[k]
    return rgb


def robust_vmax(heat: np.ndarray, pct: float = 99.0) -> float:
    """99th-percentile cap so a few outlier pixels don't blow out the colormap."""
    if heat.size == 0 or float(heat.max()) <= 0:
        return 1.0
    return float(np.percentile(heat, pct))


# --------------------------------------------------------------------------- #
# Combined sampling pass — runs ONE DDIM chain with all hooks attached.
# Replaces 4 separate chains per label (attention, SPADE γ, per-timestep,
# counterfactual-full) with a single batched chain. Plus a noise seed shared
# with the ablation pass so the "synthetic" in TEST 1 matches "full" in TEST 3.
# --------------------------------------------------------------------------- #
def _find_deepest_module(model) -> tuple[str, torch.nn.Module]:
    """Locate a deep target module to hook.

    1b: model.unet.mid_block_2 (after bottleneck SPADE+attention).
    1a (MONAI): model.unet.middle_block (its bottleneck Sequential).
    Fallback for 1a: last module whose name contains 'attention' or 'attn'.
    """
    if is_spade_model(model):
        return "unet.mid_block_2", model.unet.mid_block_2

    unet = model.unet
    if hasattr(unet, "middle_block"):
        return "unet.middle_block", unet.middle_block

    # Fallback: walk named_modules, pick the last attention-like module
    candidate: tuple[str, torch.nn.Module] | None = None
    for name, mod in unet.named_modules():
        n_lower = name.lower()
        if "attn" in n_lower or "attention" in n_lower:
            candidate = (f"unet.{name}", mod)
    if candidate is None:
        raise RuntimeError("Could not locate a deep module to hook for attention map.")
    return candidate


def combined_explain_sampling(
    model, labels: torch.Tensor, scheduler, device: torch.device,
    guidance_scale: float, n_snapshots: int = 6, noise_seed: int = 0,
) -> dict:
    """Run ONE batched DDIM chain with attention + SPADE γ hooks and capture
    per-timestep snapshots. Replaces 4 separate sampling chains.

    Online running means for attention/γ keep memory bounded to one (B, H, W)
    tensor per hook target rather than (T_passes, B, H, W).

    Returns dict with:
      samples           — (B, 1, H, W) in [-1, 1], GPU
      attn_map          — (B, H, W) CPU mean |activation| at deepest module
      attn_hook_target  — str name of the hooked module
      spade_gamma       — dict {module_name: (B, H, W) CPU mean |γ|}, empty for concat
      snapshots         — dict {step_idx: (B, 1, H, W) CPU x_t at that step}
    """
    B, _, H, W = labels.shape
    target_name, target = _find_deepest_module(model)

    # Online accumulators on CPU
    attn_state = {"sum": None, "count": 0}
    spade_state: dict[str, dict] = {}  # name -> {"sum": tensor, "count": int}

    def attn_hook(_m, _in, output):
        t = output[0] if isinstance(output, (list, tuple)) and len(output) > 0 else output
        if not isinstance(t, torch.Tensor):
            return
        with torch.no_grad():
            a = t.detach().abs().mean(dim=1, keepdim=True)   # (B, 1, h, w)
            a_up = F.interpolate(a, size=(H, W), mode="bilinear",
                                 align_corners=False).squeeze(1).cpu().float()
        if attn_state["sum"] is None:
            attn_state["sum"] = a_up
        else:
            attn_state["sum"] += a_up
        attn_state["count"] += 1

    def make_spade_hook(name: str):
        def spade_hook(module, inputs, _out):
            x_in, lbl_in = inputs[0], inputs[1]
            with torch.no_grad():
                lbl_r = F.interpolate(lbl_in, size=x_in.shape[-2:], mode="nearest")
                actv = module.mlp_shared(lbl_r)
                gamma = module.mlp_gamma(actv)
                g_abs = gamma.detach().abs().mean(dim=1, keepdim=True)
                g_up = F.interpolate(g_abs, size=(H, W), mode="bilinear",
                                     align_corners=False).squeeze(1).cpu().float()
            s = spade_state.setdefault(name, {"sum": None, "count": 0})
            if s["sum"] is None:
                s["sum"] = g_up
            else:
                s["sum"] += g_up
            s["count"] += 1
        return spade_hook

    handles = [target.register_forward_hook(attn_hook)]
    if is_spade_model(model):
        for name, mod in model.unet.named_modules():
            if isinstance(mod, SPADE):
                handles.append(mod.register_forward_hook(make_spade_hook(name)))

    # Deterministic initial noise — shared with ablation pass so "synthetic"
    # in TEST 1 matches "counterfactual: full" in TEST 3.
    g = torch.Generator(device=device).manual_seed(noise_seed)
    x = torch.randn(B, 1, H, W, device=device, generator=g)

    # Snapshot indices through the denoising chain
    n_steps = len(scheduler.timesteps)
    snapshot_steps = sorted({
        int(round(i * n_steps / (n_snapshots - 1))) for i in range(n_snapshots)
    })
    snapshots: dict[int, torch.Tensor] = {}
    if 0 in snapshot_steps:
        snapshots[0] = x.clamp(-1.0, 1.0).cpu().float()

    use_cfg = guidance_scale != 1.0
    null_label = torch.zeros_like(labels) if use_cfg else None

    try:
        with torch.no_grad():
            for i, t in enumerate(scheduler.timesteps):
                t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
                eps_cond = model.predict_noise(x, labels, t_batch)
                if use_cfg:
                    eps_uncond = model.predict_noise(x, null_label, t_batch)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x, _ = scheduler.step(model_output=eps, timestep=int(t), sample=x)
                step_count = i + 1
                if step_count in snapshot_steps:
                    snapshots[step_count] = x.clamp(-1.0, 1.0).cpu().float()
    finally:
        for h in handles:
            h.remove()

    samples = x.clamp(-1.0, 1.0)

    attn_map = (
        attn_state["sum"] / max(1, attn_state["count"])
        if attn_state["sum"] is not None
        else torch.zeros(B, H, W)
    )
    spade_gamma = {
        name: s["sum"] / max(1, s["count"]) for name, s in spade_state.items()
    }

    return {
        "samples": samples,
        "attn_map": attn_map,
        "attn_hook_target": target_name,
        "spade_gamma": spade_gamma,
        "snapshots": snapshots,
    }


# --------------------------------------------------------------------------- #
# (2) GradientSHAP attribution on label channels
# --------------------------------------------------------------------------- #
def gradient_shap_label(
    model, label: torch.Tensor, t_value: int = 500,
    n_samples: int = 10, noise_seed: int = 42,
) -> torch.Tensor:
    """Manual GradientSHAP — gradient × (input − baseline), averaged across
    random interpolation points between baseline (zero label) and input.

    Target: scalar mean over the squared predicted noise at a single timestep.
    Baseline: zero label tensor (matches the CFG null condition).

    Returns: (B, C, H, W) absolute attribution map per label channel.
    """
    device = label.device
    b, _, h, w = label.shape

    g = torch.Generator(device=device).manual_seed(noise_seed)
    x_t = torch.randn(b, 1, h, w, device=device, generator=g)
    t = torch.full((b,), int(t_value), device=device, dtype=torch.long)

    baseline = torch.zeros_like(label)
    attributions = torch.zeros_like(label)

    model.eval()
    for _ in range(n_samples):
        alpha = torch.rand(1, device=device, generator=g).item()
        interp = (baseline + alpha * (label - baseline)).detach().requires_grad_(True)
        eps = model.predict_noise(x_t, interp, t)
        target = eps.pow(2).mean(dim=[1, 2, 3]).sum()
        grad = torch.autograd.grad(target, interp,
                                   retain_graph=False, create_graph=False)[0]
        attributions = attributions + (grad * (label - baseline)).detach()
    attributions = attributions / n_samples
    return attributions.abs().cpu().float()


# --------------------------------------------------------------------------- #
# Counterfactual label ablation (4 ablation chains; the "full" comes for free
# from combined_explain_sampling so we don't recompute it here)
# --------------------------------------------------------------------------- #
def _sample_with_noise(model, label_in, init_noise, scheduler, device, guidance_scale):
    """DDIM sampling starting from an externally-supplied noise tensor.
    Mirrors model.sample() but lets us re-use the same noise across runs."""
    b = label_in.shape[0]
    x = init_noise.clone()
    use_cfg = guidance_scale != 1.0
    null_label = torch.zeros_like(label_in) if use_cfg else None
    for t in scheduler.timesteps:
        t_batch = torch.full((b,), int(t), device=device, dtype=torch.long)
        eps_cond = model.predict_noise(x, label_in, t_batch)
        if use_cfg:
            eps_uncond = model.predict_noise(x, null_label, t_batch)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        x, _ = scheduler.step(model_output=eps, timestep=int(t), sample=x)
    return x.clamp(-1.0, 1.0)


def compute_ablations(
    model, label: torch.Tensor, scheduler, device: torch.device,
    guidance_scale: float, ablate_channels: tuple[int, ...] = ORGAN_CHANNELS,
    noise_seed: int = 0,
) -> dict[int, torch.Tensor]:
    """Generate one sample per ablated organ channel. All runs share the same
    initial noise (seeded by noise_seed) so per-pixel diffs against the
    combined-pass `samples` are meaningful — provided that combined_explain
    _sampling was called with the SAME noise_seed.

    Returns {channel_index: (B, 1, H, W) in [-1, 1]}.
    """
    B, _, H, W = label.shape
    g = torch.Generator(device=device).manual_seed(noise_seed)
    base_noise = torch.randn(B, 1, H, W, device=device, generator=g)

    ablated: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for ch in ablate_channels:
            if ch >= label.shape[1]:
                continue
            label_ab = label.clone()
            label_ab[:, ch] = 0
            ablated[ch] = _sample_with_noise(model, label_ab, base_noise,
                                             scheduler, device, guidance_scale)
    return ablated


# --------------------------------------------------------------------------- #
# Category 1 — quantitative interpretability metrics
# --------------------------------------------------------------------------- #
# All metrics are per-sample, channel-resolved where applicable. They turn the
# visual TEST 2/3/5 panels into numbers that can be aggregated across samples
# and compared across model variants (1a, 1b, 1c_concat, 1c_spade).

def compute_clr(
    synth_full: np.ndarray,                  # (1, H, W) in [-1, 1]
    synth_ablated: dict[int, np.ndarray],    # ch -> (1, H, W)
    label: np.ndarray,                        # (C, H, W) one-hot
) -> dict[str, float]:
    """Counterfactual Localisation Ratio (CLR).

    For each ablated channel, what fraction of the L² change between
    full-label sample and ablated-label sample falls *inside* that channel's
    label region?

    CLR(ch) = ‖diff‖² inside ch_mask / ‖diff‖² overall

    High (→1.0) = surgical per-channel conditioning.
    Low (→0.3 in 6 channels) = diffuse conditioning, change spread everywhere.
    Equal-distribution baseline ≈ fraction of pixels covered by ch_mask.
    """
    out: dict[str, float] = {}
    full = synth_full[0]
    for ch, synth_ab in synth_ablated.items():
        diff = (synth_ab[0] - full) ** 2
        mask = label[ch] > 0
        if mask.sum() == 0 or diff.sum() == 0:
            out[ORGAN_NAMES.get(ch, f"ch{ch}")] = float("nan")
            continue
        ratio = float(diff[mask].sum() / diff.sum())
        out[ORGAN_NAMES.get(ch, f"ch{ch}")] = ratio
    return out


def compute_ailm(
    grad_attr: np.ndarray,    # (C, H, W) per-channel attribution
    label: np.ndarray,        # (C, H, W) one-hot
) -> dict[str, float]:
    """Attribution Inside-Label Mass (AILM).

    For each channel, what fraction of its GradientSHAP magnitude falls
    inside its own label region?

    AILM(ch) = sum |grad_attr[ch]| inside ch_mask / sum |grad_attr[ch]| total

    High = attribution stays inside its semantic region (clean conditioning).
    Low = attribution leaks across the image.
    """
    out: dict[str, float] = {}
    for ch in range(grad_attr.shape[0]):
        attr = grad_attr[ch]
        mask = label[ch] > 0
        total = attr.sum()
        if total <= 0 or mask.sum() == 0:
            out[LABEL_NAMES[ch]] = float("nan")
            continue
        inside = attr[mask].sum()
        out[LABEL_NAMES[ch]] = float(inside / total)
    return out


def compute_attribution_sparsity(grad_attr: np.ndarray) -> dict[str, float]:
    """Gini coefficient of attribution magnitude per channel.
    0 = uniform attribution (diffuse), 1 = all attribution at one pixel."""
    out: dict[str, float] = {}
    for ch in range(grad_attr.shape[0]):
        vals = np.sort(grad_attr[ch].flatten())
        n = vals.shape[0]
        if vals.sum() == 0:
            out[LABEL_NAMES[ch]] = float("nan")
            continue
        idx = np.arange(1, n + 1)
        gini = (2 * (idx * vals).sum() - (n + 1) * vals.sum()) / (n * vals.sum())
        out[LABEL_NAMES[ch]] = float(gini)
    return out


def compute_osi(
    spade_gamma: dict[str, np.ndarray],   # module_name -> (H, W) |γ|
    label: np.ndarray,                    # (C, H, W) one-hot
) -> dict[str, dict[str, float]]:
    """SPADE γ Organ-Specificity Index (OSI), 1b/1c_spade only.

    For each SPADE module, Pearson correlation between |γ| and each organ
    mask. Reports max-organ correlation AND body-mask correlation per module:
      - High max-organ corr → module specialises per organ (SPADE-as-designed)
      - High body corr only → module encodes body shape only (degenerate)
    """
    out: dict[str, dict[str, float]] = {}
    # Build per-organ masks
    organ_masks = {LABEL_NAMES[ch]: (label[ch] > 0).astype(np.float32).flatten()
                   for ch in ORGAN_CHANNELS if ch < label.shape[0]}
    # "Body" mask = inside body (any of body_other, uterus, ovaries, em)
    if label.shape[0] >= 6:
        body_mask = ((label[1:6].sum(axis=0)) > 0).astype(np.float32).flatten()
    else:
        body_mask = ((label[1:].sum(axis=0)) > 0).astype(np.float32).flatten()

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    for name, gamma in spade_gamma.items():
        g = gamma.flatten()
        corrs_organ = {organ: _pearson(g, m) for organ, m in organ_masks.items()}
        # Filter NaNs (empty organ channels) and take max
        valid = {k: v for k, v in corrs_organ.items() if not np.isnan(v)}
        max_organ = max(valid.values()) if valid else float("nan")
        body_corr = _pearson(g, body_mask)
        out[name] = {
            "max_organ_corr": max_organ,
            "body_corr": float(body_corr),
            "per_organ": corrs_organ,
        }
    return out


def compute_all_interpretability_metrics(
    grad_attr: np.ndarray,
    label: np.ndarray,
    counterfactual_full: Optional[np.ndarray] = None,
    counterfactual_ablated: Optional[dict[int, np.ndarray]] = None,
    spade_gamma: Optional[dict[str, np.ndarray]] = None,
) -> dict:
    """Combine all per-sample metrics into one nested dict ready for JSON dump."""
    result: dict = {
        "AILM_per_channel": compute_ailm(grad_attr, label),
        "attribution_sparsity_per_channel": compute_attribution_sparsity(grad_attr),
    }
    if counterfactual_full is not None and counterfactual_ablated:
        result["CLR_per_channel"] = compute_clr(
            counterfactual_full, counterfactual_ablated, label
        )
    if spade_gamma:
        result["OSI_per_module"] = compute_osi(spade_gamma, label)
        # Convenience aggregates across modules
        if result["OSI_per_module"]:
            max_organs = [v["max_organ_corr"] for v in result["OSI_per_module"].values()
                          if not np.isnan(v["max_organ_corr"])]
            body_corrs = [v["body_corr"] for v in result["OSI_per_module"].values()
                          if not np.isnan(v["body_corr"])]
            result["OSI_summary"] = {
                "mean_max_organ_corr": float(np.mean(max_organs)) if max_organs else float("nan"),
                "mean_body_corr": float(np.mean(body_corrs)) if body_corrs else float("nan"),
            }
    return result


# --------------------------------------------------------------------------- #
# Figure assembly — annotated, lay-reader-friendly
# --------------------------------------------------------------------------- #
def _wrap(text: str, width: int = 175) -> str:
    return textwrap.fill(text, width=width)


def _add_colorbar(fig, left: float, bottom_frac: float, height_frac: float,
                   cmap_name: str, vmin: float, vmax: float, label: str) -> None:
    """Vertical colorbar at fractional figure coords."""
    cbar_ax = fig.add_axes([left, bottom_frac + 0.005, 0.012, max(0.001, height_frac - 0.01)])
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(label, fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)


def plot_explanation_figure(
    sample_idx: int,
    real: np.ndarray,
    synthetic: np.ndarray,
    label: np.ndarray,
    attn_map: np.ndarray,
    attn_hook_target: str,
    grad_attr: np.ndarray,
    counterfactual_full: np.ndarray,
    counterfactual_ablated: dict[int, np.ndarray],
    snapshots: dict[int, np.ndarray],
    n_total_steps: int,
    spade_gamma: Optional[dict[str, np.ndarray]] = None,
    out_path: Optional[Path] = None,
    subject_id: str = "",
    inference_settings: str = "",
) -> None:
    """Annotated multi-section figure: titled tests, descriptions, colorbars,
    organ legend, footer. Designed so a paper reader can interpret each panel
    without external documentation."""
    has_spade = spade_gamma is not None and len(spade_gamma) > 0
    n_cols = 6

    # ------------------------------------------------------------------ #
    # Test definitions: (name, description, colorbar_cmap, colorbar_label)
    # colorbar_cmap=None means no colorbar for this test row.
    # ------------------------------------------------------------------ #
    tests = [
        (
            "TEST 1 — Bottleneck Activation Map",
            f"Where the model carries signal at its deepest U-Net layer ({attn_hook_target.split('.')[-1]}). "
            f"Activation magnitude (|values|) averaged across feature channels and across all denoising timesteps. "
            f"Black = low activation, yellow/white = high. "
            f"The first 4 panels are reference (real, label, synthetic, overlay).",
            "hot", "|activation| (0 = low, 1 = high, normalized)",
        ),
        (
            "TEST 2 — GradientSHAP (per label channel)",
            "Which label pixels drove the model's predicted noise. Manual GradientSHAP with the zero-label as baseline, "
            "computed at timestep t=500 (mid-range). One panel per label channel. "
            "Dark purple = no influence, bright yellow = strong influence. "
            "Empty channels (e.g. an absent R-ovary in a label without one) should be uniformly dark.",
            "viridis", "attribution magnitude (normalized per channel)",
        ),
        (
            "TEST 3 — Counterfactual Label Ablation",
            "What changes in the generated image when one organ channel is removed from the label. "
            "All 5 sampling runs share the SAME initial noise so per-pixel diffs are meaningful. "
            "First panel: full label (all organs). Next 4: each organ removed in turn. "
            "Last panel: diff (synthetic with uterus removed − synthetic with full label). "
            "Red = brighter after removal, blue = darker after removal, white = no change. "
            "A near-empty diff means the model barely uses that channel.",
            "bwr", "synth(−uterus) − synth(full)  (red>0, blue<0)",
        ),
        (
            "TEST 4 — Denoising Trajectory",
            "How the synthetic image is built across the DDIM steps. "
            "Leftmost: pure Gaussian noise (step 0, ~100% noise). Rightmost: final clean image (~0% noise). "
            "Body silhouette should emerge first (high noise), then organ positions (mid noise), then texture (low noise). "
            "This is a sanity check on the diffusion process.",
            None, None,
        ),
    ]
    if has_spade:
        tests.append((
            "TEST 5 — SPADE γ Magnitude (1b SPADE only)",
            "Magnitude of the SPADE γ modulation at 6 decoder modules (deepest → shallowest, left → right). "
            "Dark purple/blue = no modulation, bright yellow/green = strong modulation. "
            "Interpretive reading: bright IN ORGAN REGIONS = SPADE is using its per-organ conditioning capacity. "
            "Bright ONLY along the body silhouette = SPADE is encoding inside/outside body only "
            "(information already given by the body_other and outside_body channels — the SPADE-specific "
            "advantage is not being exploited).",
            "viridis", "|γ| (normalized per panel)",
        ))

    n_tests = len(tests)

    # ------------------------------------------------------------------ #
    # Figure sizing
    # ------------------------------------------------------------------ #
    # Top: title (0.45) + legend (0.55) = 1.0 in
    # Per test: header (0.30) + 4-line description (0.85) + images (2.50) + pad (0.10) = 3.75 in
    # Bottom: pad (0.20) in
    top_h = 1.0
    per_test_h = 3.75
    bottom_h = 0.2
    fig_h = top_h + n_tests * per_test_h + bottom_h
    fig_w = n_cols * 2.55 + 0.55  # +0.55 reserves space for colorbars + margins

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    def to_frac_y(y_in: float) -> float:
        """Convert inches-from-top to fractional y (0=bottom, 1=top)."""
        return 1.0 - (y_in / fig_h)

    # ------------------------------------------------------------------ #
    # Top: title + organ legend
    # ------------------------------------------------------------------ #
    fig.text(
        0.5, to_frac_y(0.22),
        f"Explainability Report — Sample {sample_idx}" + (f"  ({subject_id})" if subject_id else ""),
        ha="center", fontsize=15, fontweight="bold",
    )
    if inference_settings:
        fig.text(
            0.5, to_frac_y(0.45),
            inference_settings,
            ha="center", fontsize=9, style="italic", color="dimgray",
        )

    # Organ color legend strip
    legend_handles = [
        Patch(facecolor=(1.0, 1.0, 0.0), edgecolor="black", label="uterus (channel 1)"),
        Patch(facecolor=(1.0, 0.0, 0.0), edgecolor="black", label="L-ovary (channel 2)"),
        Patch(facecolor=(0.0, 0.4, 1.0), edgecolor="black", label="R-ovary (channel 3)"),
        Patch(facecolor=(0.0, 1.0, 0.0), edgecolor="black", label="endometrioma (channel 4)"),
        Patch(facecolor=(0.5, 0.5, 0.5), edgecolor="black", label="body_other (channel 5)"),
        Patch(facecolor=(0.08, 0.08, 0.08), edgecolor="gray", label="outside_body (channel 0)"),
    ]
    legend_ax = fig.add_axes([0.05, to_frac_y(0.95), 0.9, 0.30 / fig_h])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles, ncol=6, loc="center", frameon=True, fontsize=9,
        title="Label channel color key (used in 'label argmax', 'overlay', and counterfactual panels)",
        title_fontsize=9,
    )

    # ------------------------------------------------------------------ #
    # Pre-compute display arrays
    # ------------------------------------------------------------------ #
    real_disp = to_disp(real[0])
    synth_disp = to_disp(synthetic[0])
    label_rgb = label_to_rgb(label)
    overlay = overlay_label_on_image(synthetic[0], label)

    # ------------------------------------------------------------------ #
    # Iterate tests, render each section
    # ------------------------------------------------------------------ #
    panel_area_left = 0.035
    panel_area_right_with_cbar = 0.92
    panel_area_right_no_cbar = 0.965
    cbar_x = 0.945

    y_in = top_h  # inches-from-top cursor

    for t_idx, (test_name, description, cbar_cmap, cbar_label) in enumerate(tests):
        # --- Section header ---
        header_h = 0.30
        fig.text(
            0.03, to_frac_y(y_in + header_h * 0.55),
            test_name, fontsize=12.5, fontweight="bold", color="#1f4e79",
        )
        y_in += header_h

        # --- Description (wrap manually to fit figure width) ---
        desc_h = 0.85
        wrapped_desc = _wrap(description, width=175)
        fig.text(
            0.035, to_frac_y(y_in + 0.05),
            wrapped_desc, fontsize=8.5, color="#333333", va="top",
            linespacing=1.25,
        )
        y_in += desc_h

        # --- Image row ---
        img_h = 2.50
        img_top_frac = to_frac_y(y_in)
        img_bottom_frac = to_frac_y(y_in + img_h)
        img_height_frac = img_top_frac - img_bottom_frac
        right_edge = panel_area_right_with_cbar if cbar_cmap is not None else panel_area_right_no_cbar
        avail_width = right_edge - panel_area_left
        panel_w = avail_width / n_cols * 0.96
        panel_gap = avail_width / n_cols * 0.04

        # Build the panel list for THIS test
        panels: list[tuple[str, Optional[np.ndarray], Optional[str], Optional[float], Optional[tuple[float, float]]]]
        if t_idx == 0:  # Bottleneck activation
            panels = [
                ("real (source slice)", real_disp, "gray", None, None),
                ("label argmax (color key above)", label_rgb, None, None, None),
                ("synthetic (model output)", synth_disp, "gray", None, None),
                ("overlay (synth + label)", overlay, None, None, None),
                (f"activation @ {attn_hook_target.split('.')[-1]}", attn_map, "hot",
                 robust_vmax(attn_map), None),
                ("", None, None, None, None),
            ]
        elif t_idx == 1:  # GradientSHAP
            panels = []
            for i in range(n_cols):
                if i < grad_attr.shape[0]:
                    heat = grad_attr[i]
                    panels.append((
                        f"GradSHAP[{LABEL_NAMES[i]}]",
                        heat, "viridis", robust_vmax(heat), None,
                    ))
                else:
                    panels.append(("", None, None, None, None))
        elif t_idx == 2:  # Counterfactual
            panels = [("full label (all organs)", to_disp(counterfactual_full[0]), "gray", None, None)]
            for ch in ORGAN_CHANNELS:
                if ch in counterfactual_ablated:
                    panels.append((
                        f"minus {ORGAN_NAMES[ch]}",
                        to_disp(counterfactual_ablated[ch][0]), "gray", None, None,
                    ))
                else:
                    panels.append((f"minus {ORGAN_NAMES[ch]} (n/a)", None, None, None, None))
            if 1 in counterfactual_ablated:
                diff = to_disp(counterfactual_ablated[1][0]) - to_disp(counterfactual_full[0])
                mag = float(np.abs(diff).max()) or 1e-3
                panels.append((
                    "diff: (− uterus) minus (full)",
                    diff, "bwr", mag, (-mag, mag),
                ))
            else:
                panels.append(("", None, None, None, None))
            panels = panels[:n_cols]
        elif t_idx == 3:  # Per-timestep
            snap_counts = sorted(snapshots.keys())
            if len(snap_counts) > n_cols:
                idx_take = np.round(np.linspace(0, len(snap_counts) - 1, n_cols)).astype(int)
                snap_counts = [snap_counts[k] for k in idx_take]
            panels = []
            for sc in snap_counts:
                frac_remaining = max(0.0, (n_total_steps - sc) / n_total_steps)
                panels.append((
                    f"step {sc}/{n_total_steps}\n(~{frac_remaining:.0%} noise)",
                    to_disp(snapshots[sc][0]), "gray", None, None,
                ))
            while len(panels) < n_cols:
                panels.append(("", None, None, None, None))
        else:  # SPADE γ
            names = sorted(spade_gamma.keys())
            if len(names) <= n_cols:
                chosen = names
            else:
                chosen = [names[int(round((len(names) - 1) * k / (n_cols - 1)))]
                          for k in range(n_cols)]
            panels = []
            for name in chosen:
                heat = spade_gamma[name]
                short = name.replace("up_resblocks.", "up").replace("down_resblocks.", "dn")
                if len(short) > 24:
                    short = "…" + short[-22:]
                panels.append((
                    f"|γ| {short}",
                    heat, "viridis", robust_vmax(heat), None,
                ))
            while len(panels) < n_cols:
                panels.append(("", None, None, None, None))

        for i, (panel_title, img, cmap_name, vmax, vrange) in enumerate(panels):
            left = panel_area_left + i * (panel_w + panel_gap)
            ax = fig.add_axes([left, img_bottom_frac, panel_w, img_height_frac])
            if img is None:
                ax.axis("off")
            else:
                if cmap_name is None:
                    ax.imshow(img)
                elif vrange is not None:
                    ax.imshow(img, cmap=cmap_name, vmin=vrange[0], vmax=vrange[1])
                else:
                    kw = {"vmin": 0, "vmax": vmax} if vmax is not None else {}
                    ax.imshow(img, cmap=cmap_name, **kw)
            ax.set_title(panel_title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        # --- Colorbar (if this test has one) ---
        if cbar_cmap is not None:
            if cbar_cmap == "bwr":
                _add_colorbar(fig, cbar_x, img_bottom_frac, img_height_frac,
                              cbar_cmap, vmin=-1.0, vmax=1.0, label=cbar_label)
            else:
                _add_colorbar(fig, cbar_x, img_bottom_frac, img_height_frac,
                              cbar_cmap, vmin=0.0, vmax=1.0, label=cbar_label)

        # Move cursor past this test row
        y_in += img_h + 0.10

    # ------------------------------------------------------------------ #
    # Footer
    # ------------------------------------------------------------------ #
    fig.text(
        0.5, to_frac_y(fig_h - 0.07),
        "Generated by src/Generator/explain.py — see TIER1_TUNING_AND_EXPLAINABILITY.md for method details.",
        ha="center", fontsize=7, color="gray", style="italic",
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[explain] wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--config", required=True, help="Training YAML")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out_dir", required=True, help="Directory for per-sample PNGs")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of high-foreground samples to explain (default 4)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG guidance scale (overrides YAML)")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="DDIM steps (overrides YAML)")
    parser.add_argument("--gradshap-t", type=int, default=500,
                        help="Timestep at which to compute GradientSHAP (default 500)")
    parser.add_argument("--gradshap-samples", type=int, default=10,
                        help="Interpolation samples for GradientSHAP (default 10)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use training weights instead of EMA")
    parser.add_argument("--skip-counterfactual", action="store_true",
                        help="Skip the counterfactual row (saves 4 sampling runs)")
    parser.add_argument("--n-snapshots", type=int, default=6,
                        help="Snapshots through the denoising chain (default 6)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[explain] device={device}")
    if device.type == "cuda":
        print(f"[explain] GPU={torch.cuda.get_device_name(0)}")

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
        raise RuntimeError("No slices with any foreground voxels in the train split.")
    labels = torch.from_numpy(np.stack(labels_np)).float().to(device)
    reals = torch.from_numpy(np.stack(reals_np)).float().to(device)

    model = build_model_from_cfg(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "ema" in ckpt and not args.no_ema:
        model.load_state_dict(ckpt["ema"])
        weight_source = "EMA"
    else:
        model.load_state_dict(ckpt["model"])
        weight_source = "training" if "ema" not in ckpt else "training (--no-ema)"
    model.eval()
    print(f"[explain] loaded ckpt step {ckpt['step']}, weights={weight_source}")
    print(f"[explain] model type: {'SPADE (1b)' if is_spade_model(model) else 'concat (1a)'}")

    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None
        else cfg["sampling"]["num_inference_steps"]
    )
    scheduler = build_inference_scheduler(cfg["diffusion"], num_inference_steps)
    print(f"[explain] num_inference_steps={num_inference_steps}")

    guidance = (
        args.guidance_scale if args.guidance_scale is not None
        else float(cfg["sampling"].get("guidance_scale", 1.0))
    )
    print(f"[explain] guidance_scale={guidance}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = labels.shape[0]

    # ===== Refactored: all sampling done in 2 batched chains (was 8 × N) =====
    # ONE combined chain with hooks → samples + attn + spade_gamma + snapshots
    print(f"\n[explain] (1) combined sampling chain (attn + SPADE γ + snapshots) "
          f"for all {N} labels at once ...")
    combined = combined_explain_sampling(
        model, labels, scheduler, device,
        guidance_scale=guidance, n_snapshots=args.n_snapshots, noise_seed=0,
    )
    samples_all = combined["samples"]                    # (N, 1, H, W) GPU
    attn_map_all = combined["attn_map"]                  # (N, H, W) CPU
    hook_target = combined["attn_hook_target"]
    spade_gamma_all = combined["spade_gamma"]            # {name: (N, H, W) CPU}
    snapshots_all = combined["snapshots"]                # {step: (N, 1, H, W) CPU}

    # GradientSHAP — batched across all labels
    print(f"[explain] (2) GradientSHAP @ t={args.gradshap_t}, "
          f"n_samples={args.gradshap_samples} (batched across all labels) ...")
    grad_attr_all = gradient_shap_label(
        model, labels, t_value=args.gradshap_t, n_samples=args.gradshap_samples,
    )                                                    # (N, C, H, W) CPU

    # Counterfactual ablations — share noise_seed=0 with combined pass so the
    # "full" panel in TEST 3 IS the same image as the "synthetic" in TEST 1.
    if not args.skip_counterfactual:
        print(f"[explain] (3) counterfactual ablations "
              f"({len(ORGAN_CHANNELS)} channels × {N} labels, batched) ...")
        cf_ablated_all = compute_ablations(
            model, labels, scheduler, device,
            guidance_scale=guidance, noise_seed=0,
        )                                                # {ch: (N, 1, H, W)}
    else:
        cf_ablated_all = {}

    # ===== Per-sample figure rendering + JSON dump (CPU-only) =====
    inference_settings = (
        f"guidance = {guidance}    |    DDIM steps = {num_inference_steps}    |    "
        f"weights = {weight_source}    |    conditioning = "
        f"{'SPADE (Exp 1b)' if is_spade_model(model) else 'concat (Exp 1a)'}"
    )

    samples_cpu = samples_all.cpu().numpy()
    grad_attr_cpu = grad_attr_all.numpy()
    cf_ablated_cpu = {ch: t.cpu().numpy() for ch, t in cf_ablated_all.items()}
    snapshots_cpu = {sc: t.numpy() for sc, t in snapshots_all.items()}
    spade_gamma_cpu = (
        {n: t.numpy() for n, t in spade_gamma_all.items()} if spade_gamma_all else None
    )
    labels_cpu = labels.cpu().numpy()
    reals_cpu = reals.cpu().numpy()
    attn_cpu = attn_map_all.numpy()

    for i in range(N):
        print(f"\n[explain] rendering sample {i+1}/{N} ...")
        out_path = out_dir / f"sample_{i:02d}.png"
        json_path = out_dir / f"sample_{i:02d}_metrics.json"

        cf_full_i = samples_cpu[i]                                # (1, H, W) — same as TEST 1 synth
        cf_ablated_i = {ch: t[i] for ch, t in cf_ablated_cpu.items()}
        spade_gamma_i = (
            {n: t[i] for n, t in spade_gamma_cpu.items()} if spade_gamma_cpu else None
        )
        snapshots_i = {sc: t[i] for sc, t in snapshots_cpu.items()}

        # Category 1 metrics + JSON
        metrics = compute_all_interpretability_metrics(
            grad_attr=grad_attr_cpu[i],
            label=labels_cpu[i],
            counterfactual_full=cf_full_i if not args.skip_counterfactual else None,
            counterfactual_ablated=cf_ablated_i if not args.skip_counterfactual else None,
            spade_gamma=spade_gamma_i,
        )
        metrics["meta"] = {
            "sample_idx": int(i),
            "variant": cfg["experiment"]["name"],
            "ckpt_step": int(ckpt["step"]),
            "weights": weight_source,
            "guidance_scale": guidance,
            "num_inference_steps": num_inference_steps,
            "attn_hook_target": hook_target,
        }
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[explain] wrote {json_path}")

        plot_explanation_figure(
            sample_idx=i,
            real=reals_cpu[i],
            synthetic=samples_cpu[i],
            label=labels_cpu[i],
            attn_map=attn_cpu[i],
            attn_hook_target=hook_target,
            grad_attr=grad_attr_cpu[i],
            counterfactual_full=cf_full_i,
            counterfactual_ablated=cf_ablated_i,
            snapshots=snapshots_i,
            n_total_steps=num_inference_steps,
            spade_gamma=spade_gamma_i,
            out_path=out_path,
            inference_settings=inference_settings,
        )

    print(f"\n[explain] done. {N} figures in {out_dir}")


if __name__ == "__main__":
    main()
