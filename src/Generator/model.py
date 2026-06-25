"""
Concat-conditioned 2D DDPM for Exp 1a.

The conditioning strategy is the simplest possible: concatenate the 5-channel
label map with the noisy image along the channel dimension, giving a 6-channel
input to the U-Net. The U-Net predicts noise on a 1-channel output.

This is the *baseline* for the Phase 1 ablation. Exp 1b will replace this with
SPADE in the decoder; Exp 1c will add a PatchGAN. The interface here stays
minimal and avoids hiding any conditioning machinery.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from .unet_spade import DiffusionUNetSPADE


class EMAModel:
    """Exponential moving average of a wrapped nn.Module's parameters.

    Standard practice in DDPM literature (Ho et al. 2020, ADM, EDM, Imagen,
    Stable Diffusion). After each optimisation step, the EMA copy is updated:
        ema_params = decay * ema_params + (1 - decay) * train_params

    Decay = 0.9999 averages effectively over the last ~10k steps. The EMA copy
    smooths out high-frequency optimisation noise and tends to produce visibly
    cleaner samples — particularly the textured graininess you see in raw
    DDPM samples. It is used for inference, never for backprop.

    Buffers (e.g. BatchNorm running stats, though our UNet uses GroupNorm so
    this is moot in practice) are copied verbatim, not averaged.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, train_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(train_p.data, alpha=1.0 - self.decay)
        for ema_b, train_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(train_b.data)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.ema_model.load_state_dict(sd)


def build_unet(model_cfg: dict) -> DiffusionModelUNet:
    return DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_cfg["in_channels"],          # 7 = 1 image + 6 labels
        out_channels=model_cfg["out_channels"],        # 1 = noise on image
        num_channels=tuple(model_cfg["num_channels"]),
        attention_levels=tuple(model_cfg["attention_levels"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        num_head_channels=tuple(model_cfg["num_head_channels"]),
        norm_num_groups=model_cfg["norm_num_groups"],
    )


def build_model_from_cfg(cfg: dict) -> "_BaseConditionedDDPM":
    """Dispatch on cfg['model']['type'] (default 'concat' for 1a).

    Returns a wrapped DDPM ready to call .predict_noise / .sample.
    """
    model_type = cfg["model"].get("type", "concat")
    if model_type == "concat":
        return ConcatConditionedDDPM(build_unet(cfg["model"]))
    if model_type == "spade":
        num_label_channels = cfg["data"]["num_label_channels"]
        return SPADEConditionedDDPM(build_unet_spade(cfg["model"], num_label_channels))
    raise ValueError(
        f"Unknown model.type={model_type!r}; supported: 'concat' (1a), 'spade' (1b)"
    )


def build_unet_spade(model_cfg: dict, num_label_channels: int) -> DiffusionUNetSPADE:
    """Builds the 1b backbone — pure-SPADE U-Net.

    Pure SPADE means in_channels is 1 (noisy image only); the label tensor
    enters the network through SPADE modules at the bottleneck and every
    decoder ResBlock, not via input concatenation.
    """
    return DiffusionUNetSPADE(
        in_channels=model_cfg["in_channels"],            # 1 for pure SPADE
        out_channels=model_cfg["out_channels"],          # 1 = noise on image
        label_channels=num_label_channels,               # 6 for our setup
        channels=tuple(model_cfg["num_channels"]),
        attention_levels=tuple(model_cfg["attention_levels"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        spade_hidden=model_cfg.get("spade_hidden", 64),
    )


def build_train_scheduler(diff_cfg: dict) -> DDPMScheduler:
    # MONAI Generative's scheduler API names the schedule type `schedule`
    # (not `beta_schedule`); the surplus kwarg gets forwarded to the schedule
    # function (e.g. `_linear_beta`) which rejects it. We keep the YAML key
    # name as `beta_schedule` for readability and map it here.
    return DDPMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        schedule=diff_cfg["beta_schedule"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        prediction_type=diff_cfg["prediction_type"],
    )


def build_inference_scheduler(diff_cfg: dict, num_inference_steps: int) -> DDIMScheduler:
    sched = DDIMScheduler(
        num_train_timesteps=diff_cfg["num_train_timesteps"],
        schedule=diff_cfg["beta_schedule"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        prediction_type=diff_cfg["prediction_type"],
    )
    sched.set_timesteps(num_inference_steps)
    return sched


class _BaseConditionedDDPM(nn.Module):
    """Common sampling logic for both ConcatConditionedDDPM and
    SPADEConditionedDDPM. Subclasses implement `predict_noise`."""

    def predict_noise(
        self,
        x_t: torch.Tensor,
        label: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        label: torch.Tensor,
        scheduler: DDIMScheduler,
        device: torch.device,
        guidance_scale: float = 1.0,
        progress: bool = False,
    ) -> torch.Tensor:
        """DDIM sampling with optional Classifier-Free Guidance.

        guidance_scale (w):
            1.0  → conditional only (no CFG)
            0.0  → unconditional only
            >1.0 → CFG: ε_guided = ε_uncond + w·(ε_cond − ε_uncond)

        At w=1.0 we skip the second forward pass — same compute as before.
        At w≠1.0 each denoising step is 2× the compute.
        """
        b = label.shape[0]
        h, w_img = label.shape[-2:]
        x = torch.randn(b, 1, h, w_img, device=device)
        iterator = scheduler.timesteps
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="sampling")

        use_cfg = guidance_scale != 1.0
        null_label = torch.zeros_like(label) if use_cfg else None

        for t in iterator:
            t_batch = torch.full((b,), int(t), device=device, dtype=torch.long)
            eps_cond = self.predict_noise(x, label, t_batch)
            if use_cfg:
                eps_uncond = self.predict_noise(x, null_label, t_batch)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond
            x, _ = scheduler.step(model_output=eps, timestep=int(t), sample=x)
        return x.clamp(-1.0, 1.0)


class ConcatConditionedDDPM(_BaseConditionedDDPM):
    """Wraps the U-Net so the training/inference loops don't need to know about
    the channel-concat trick."""

    def __init__(self, unet: DiffusionModelUNet):
        super().__init__()
        self.unet = unet

    def predict_noise(
        self,
        x_t: torch.Tensor,         # (B, 1, H, W) noisy image
        label: torch.Tensor,       # (B, C, H, W) condition
        timesteps: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        x_in = torch.cat([x_t, label], dim=1)
        return self.unet(x=x_in, timesteps=timesteps)


class SPADEConditionedDDPM(_BaseConditionedDDPM):
    """Exp 1b wrapper. Same external interface as ConcatConditionedDDPM —
    train.py and inference_validate.py don't need to know which conditioning
    mechanism is underneath.

    The difference from ConcatConditionedDDPM:
      - The U-Net's input is the noisy image alone (1 channel)
      - The label is passed through to the U-Net as a separate argument,
        which routes it to the SPADE modules at the bottleneck + decoder
    """

    def __init__(self, unet: DiffusionUNetSPADE):
        super().__init__()
        self.unet = unet

    def predict_noise(
        self,
        x_t: torch.Tensor,         # (B, 1, H, W) noisy image
        label: torch.Tensor,       # (B, C, H, W) condition
        timesteps: torch.Tensor,   # (B,) int64
    ) -> torch.Tensor:
        return self.unet(x=x_t, timesteps=timesteps, label=label)


