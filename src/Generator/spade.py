"""
SPADE (Spatially-Adaptive Normalization) module.

Reference: Park, Liu, Wang, Zhu (2019),
  "Semantic Image Synthesis with Spatially-Adaptive Normalization", CVPR.

Replaces standard normalization with a label-conditioned modulation:
    y = ParamFreeNorm(x)
    γ, β = SmallConvNet(label_downsampled_to_xy)
    out = y * (1 + γ) + β

The label tensor (one-hot or near one-hot, [B, C_label, H_label, W_label])
is nearest-neighbour resized to the feature-map's H×W before the modulation
is computed. Nearest-neighbour preserves the categorical structure of the
label; bilinear would blur class boundaries.

This module is the building block injected at every decoder ResBlock and at
the bottleneck of the SPADE-conditioned diffusion U-Net (Exp 1b).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Spatially-Adaptive Normalization for label-conditioned features.

    Args:
        norm_channels: number of feature channels we are normalising
        label_channels: number of channels in the (one-hot) label tensor
        hidden: width of the shared MLP that produces γ and β
        num_groups: GroupNorm groups (must divide norm_channels)
    """

    def __init__(
        self,
        norm_channels: int,
        label_channels: int,
        hidden: int = 64,
        num_groups: int = 32,
    ):
        super().__init__()
        if norm_channels % num_groups != 0:
            # Fall back to the largest divisor ≤ requested num_groups
            for g in (32, 16, 8, 4, 2, 1):
                if norm_channels % g == 0:
                    num_groups = g
                    break
        # Affine OFF — SPADE provides spatially-varying γ and β instead.
        self.param_free_norm = nn.GroupNorm(num_groups, norm_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden, norm_channels, kernel_size=3, padding=1)

        # Zero-init BOTH γ and β heads so SPADE starts as pure GroupNorm
        # (output = GroupNorm(x) * 1 + 0 = identity for an affine-free norm).
        # Without zero-init γ has random non-zero values from step 0 and
        # feature scaling is chaotic before the model learns what γ should
        # be; without zero-init β the additive term still injects random
        # noise even with γ=0. Standard practice for modulation networks in
        # diffusion (DiT's AdaLN-Zero, Imagen, SDM).
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        normalised = self.param_free_norm(x)
        # Resize the label to the feature-map resolution. Nearest neighbour
        # preserves the categorical structure of the one-hot encoding.
        label_resized = F.interpolate(label, size=x.shape[-2:], mode="nearest")
        actv = self.mlp_shared(label_resized)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalised * (1.0 + gamma) + beta
