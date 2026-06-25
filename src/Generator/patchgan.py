"""
Conditional PatchGAN discriminator for Exp 1c.

Architecture: 70×70 receptive field on 512×512 input. Standard pix2pix
discriminator (Isola et al. 2017) with spectral normalisation (Miyato 2018)
on all weight layers for training stability.

The discriminator is *conditional*: input is `concat(image, label_map)` where
label_map is the 6-channel one-hot used by the generator. This forces the
discriminator to judge both texture realism AND image-label consistency
simultaneously — the latter is what makes adversarial loss useful here
versus a pure unconditional GAN.

Used in train.py when the config has a `discriminator:` block. Same
discriminator architecture is used for both Exp 1c-concat and Exp 1c-spade
(only the generator side differs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sn(layer: nn.Module) -> nn.Module:
    """Spectral normalisation wrapper. Centralised so we can toggle for
    debugging if D collapses or saturates."""
    return nn.utils.spectral_norm(layer)


class PatchGAN(nn.Module):
    """Conditional PatchGAN — 70×70 receptive field, 5 conv blocks.

    Input shape:  (B, 1 + label_channels, H, W)   — concat of image + label
    Output shape: (B, 1, H/16, W/16)               — patch logits
                                                     For 512² input → 32² patches.

    Args:
        image_channels: 1 (grayscale T2FS slice)
        label_channels: 6 (outside_body, uterus, L-ov, R-ov, em, body_other)
        base_channels: feature width of the first conv (default 64).
            Total params ≈ 2.7 M at 64.
        use_spectral_norm: apply spectral norm to all weight layers (default True).
            Strongly recommended — without it D often saturates at >99% accuracy
            within a few hundred steps and the generator stops getting useful
            gradient.
    """

    def __init__(
        self,
        image_channels: int = 1,
        label_channels: int = 6,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        in_ch = image_channels + label_channels
        c = base_channels
        wrap = _sn if use_spectral_norm else (lambda m: m)

        # Block 1: 512² → 256², no norm (standard PatchGAN doesn't norm the first block)
        self.block1 = nn.Sequential(
            wrap(nn.Conv2d(in_ch, c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Block 2: 256² → 128²
        self.block2 = nn.Sequential(
            wrap(nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Block 3: 128² → 64²
        self.block3 = nn.Sequential(
            wrap(nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Block 4: 64² → 32² (stride 2 here for additional downsampling vs standard
        # 256² pix2pix; our 512² input needs one more level to keep patch count tractable)
        self.block4 = nn.Sequential(
            wrap(nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Block 5: stride-1 + 1x1 head, 32² patches out
        self.block5 = nn.Sequential(
            wrap(nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out = wrap(nn.Conv2d(c * 8, 1, kernel_size=4, stride=1, padding=1))

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, label], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.out(x)  # raw logits, BCEWithLogits applied externally


def discriminator_loss(
    d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor,
) -> torch.Tensor:
    """Standard non-saturating BCE on real (target=1) and fake (target=0)."""
    real_loss = F.binary_cross_entropy_with_logits(
        d_real_logits, torch.ones_like(d_real_logits)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        d_fake_logits, torch.zeros_like(d_fake_logits)
    )
    return 0.5 * (real_loss + fake_loss)


def generator_adv_loss(d_fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator's adversarial loss — wants D to call fakes 'real'."""
    return F.binary_cross_entropy_with_logits(
        d_fake_logits, torch.ones_like(d_fake_logits)
    )


def discriminator_accuracy(
    d_real_logits: torch.Tensor, d_fake_logits: torch.Tensor,
) -> tuple[float, float]:
    """Accuracy on real and fake patches (for monitoring D not saturating)."""
    with torch.no_grad():
        real_acc = (torch.sigmoid(d_real_logits) > 0.5).float().mean().item()
        fake_acc = (torch.sigmoid(d_fake_logits) < 0.5).float().mean().item()
    return real_acc, fake_acc


def estimate_x0_from_eps(
    x_t: torch.Tensor, eps_pred: torch.Tensor, scheduler, timesteps: torch.Tensor,
) -> torch.Tensor:
    """Single-step estimate x̂_0 from current x_t and predicted noise.

    Used to give the discriminator a "synthetic image" to judge each step,
    without running the full reverse diffusion chain (which would be far too
    expensive per training step).

    x̂_0 = (x_t − √(1−ᾱ_t) · ε̂) / √(ᾱ_t)

    The scheduler holds `alphas_cumprod` indexed by timestep.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(x_t.device)
    a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(x_t.dtype)
    sqrt_a_t = a_t.sqrt()
    sqrt_one_minus_a_t = (1.0 - a_t).sqrt()
    x0_hat = (x_t - sqrt_one_minus_a_t * eps_pred) / sqrt_a_t
    return x0_hat.clamp(-1.0, 1.0)


def lambda_schedule(step: int, warmup_end: int, ramp_end: int, peak: float) -> float:
    """λ-warmup schedule for the adversarial loss term.

    step ∈ [0, warmup_end)         → λ = 0 (pure DDPM training)
    step ∈ [warmup_end, ramp_end)  → λ ramps linearly 0 → peak
    step ∈ [ramp_end, ∞)           → λ = peak
    """
    if step < warmup_end:
        return 0.0
    if step >= ramp_end:
        return peak
    frac = (step - warmup_end) / max(1, ramp_end - warmup_end)
    return peak * frac
