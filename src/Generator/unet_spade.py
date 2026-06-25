"""
SPADE-conditioned 2D diffusion U-Net for Exp 1b.

Structure mirrors 1a's MONAI DiffusionModelUNet closely so the 1a-vs-1b
ablation isolates the conditioning mechanism rather than the architecture:

    Levels        4 (resolution 512² → 256² → 128² → 64²)
    Channel widths [64, 128, 256, 256]
    Res blocks    2 per encoder level, 2 per decoder level
    Attention     self-attention at the deepest level only (64²)
                  — level-2 attention OOMs the A100 80GB even at batch 4
    Norm          GroupNorm(32). Encoder uses standard affine GroupNorm;
                  decoder + bottleneck use SPADE GroupNorm conditioned on
                  the 6-channel label tensor.
    Time emb      Sinusoidal positional encoding → 2-layer MLP, added to
                  every ResBlock via Linear projection
    Conditioning  Label enters ONLY through SPADE. The image input is the
                  noisy MRI (1 channel) — no concat conditioning. Setting
                  the label tensor to zero in training (CFG dropout) collapses
                  γ→learned-baseline, β→learned-baseline, so the unconditional
                  pathway is jointly learned.

Pure SPADE (not hybrid concat+SPADE): in_channels = 1 by design.

Parameter count is roughly 25-30 M with the default config — comparable to 1a.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spade import SPADE


# --------------------------------------------------------------------------- #
# Sinusoidal time-step embedding
# --------------------------------------------------------------------------- #
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Diffusion-standard sinusoidal embedding for discrete timesteps."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """sinusoidal(t) → Linear → SiLU → Linear → emb_dim"""

    def __init__(self, in_dim: int, emb_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.proj1 = nn.Linear(in_dim, emb_dim)
        self.proj2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        x = sinusoidal_time_embedding(timesteps, self.in_dim)
        x = F.silu(self.proj1(x))
        x = self.proj2(x)
        return x


# --------------------------------------------------------------------------- #
# Residual block, with optional SPADE normalisation
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    """2D residual block: norm → silu → conv → +t_emb → norm → silu → conv → +skip.

    If `label_channels` is given, both norms are SPADE (label-conditioned).
    Otherwise they are standard affine GroupNorm.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        label_channels: int | None = None,
        spade_hidden: int = 64,
    ):
        super().__init__()
        self.use_spade = label_channels is not None
        if self.use_spade:
            self.norm1 = SPADE(in_ch, label_channels, hidden=spade_hidden)
            self.norm2 = SPADE(out_ch, label_channels, hidden=spade_hidden)
        else:
            self.norm1 = nn.GroupNorm(32, in_ch)
            self.norm2 = nn.GroupNorm(32, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x, label) if self.use_spade else self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.norm2(h, label) if self.use_spade else self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


# --------------------------------------------------------------------------- #
# 2D self-attention (deep-layer use only)
# --------------------------------------------------------------------------- #
class SelfAttention2D(nn.Module):
    """Multi-head self-attention over flattened spatial positions.

    Memory cost: B × heads × N × N where N = H·W. We only use this at the
    deepest level (64²) where N = 4096 — score tensor is ~2 GiB at batch 4,
    fp32. Adding it at level 2 (128² → N = 16384) would OOM the A100.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        normed = self.norm(x)
        q, k, v = self.qkv(normed).chunk(3, dim=1)
        # (B, heads, head_dim, N)
        q = q.view(b, self.num_heads, self.head_dim, n)
        k = k.view(b, self.num_heads, self.head_dim, n)
        v = v.view(b, self.num_heads, self.head_dim, n)
        scale = self.head_dim ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        # einsum can return a non-contiguous tensor; .reshape handles both
        # cases (calls .contiguous() internally when needed).
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


# --------------------------------------------------------------------------- #
# Down / Up sampling
# --------------------------------------------------------------------------- #
class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # Nearest-neighbour upsample + 3x3 conv: avoids checkerboarding of
        # ConvTranspose2d.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Full U-Net
# --------------------------------------------------------------------------- #
class DiffusionUNetSPADE(nn.Module):
    """SPADE-conditioned 2D diffusion U-Net.

    Args:
        in_channels: noisy-image channel count (1 for our pure-SPADE 1b)
        out_channels: output channel count (1 — predicts noise on image)
        label_channels: label tensor channel count (6 for 1b)
        channels: per-level feature widths (length = num_levels)
        attention_levels: bool per level — True activates self-attention at
            the bottleneck (and the corresponding decoder block) for that
            level. For our config only the deepest level is True.
        num_res_blocks: ResBlocks per encoder / decoder level
        spade_hidden: hidden width inside the SPADE MLP
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        label_channels: int = 6,
        channels: tuple[int, ...] = (64, 128, 256, 256),
        attention_levels: tuple[bool, ...] = (False, False, False, True),
        num_res_blocks: int = 2,
        spade_hidden: int = 64,
        time_emb_dim: int | None = None,
    ):
        super().__init__()
        assert len(channels) == len(attention_levels), \
            "channels and attention_levels must match in length"
        self.num_levels = len(channels)
        self.label_channels = label_channels

        time_emb_dim = time_emb_dim or channels[0] * 4
        self.time_embedding = TimeEmbedding(in_dim=channels[0], emb_dim=time_emb_dim)

        # Stem (image → first feature width)
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # --- Encoder (no SPADE — standard GroupNorm) --- #
        self.down_resblocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = channels[0]
        for lvl in range(self.num_levels):
            target_ch = channels[lvl]
            blocks = nn.ModuleList()
            for r in range(num_res_blocks):
                blocks.append(ResBlock(
                    in_ch=ch if r == 0 else target_ch,
                    out_ch=target_ch,
                    time_emb_dim=time_emb_dim,
                    label_channels=None,
                ))
            self.down_resblocks.append(blocks)
            ch = target_ch
            # Downsample after every level except the deepest
            self.downsamplers.append(Downsample(ch) if lvl < self.num_levels - 1 else nn.Identity())

        # --- Bottleneck (SPADE) --- #
        mid_ch = channels[-1]
        self.mid_block_1 = ResBlock(mid_ch, mid_ch, time_emb_dim,
                                    label_channels=label_channels, spade_hidden=spade_hidden)
        self.mid_attn = SelfAttention2D(mid_ch, num_heads=8) if attention_levels[-1] else nn.Identity()
        self.mid_block_2 = ResBlock(mid_ch, mid_ch, time_emb_dim,
                                    label_channels=label_channels, spade_hidden=spade_hidden)

        # --- Decoder (SPADE everywhere) --- #
        # Each decoder level concatenates the matching encoder skip onto the
        # incoming features (which carry the previous decoder level's channel
        # count, NOT necessarily this level's width). Track `prev_ch`
        # explicitly so the first ResBlock of each level sizes its norm1
        # correctly — the bug we hit was hard-coding `target_ch + skip_ch`
        # which only happens to be right when consecutive widths match.
        self.up_resblocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        prev_ch = channels[-1]  # bottleneck output channel count
        for lvl in reversed(range(self.num_levels)):
            target_ch = channels[lvl]
            skip_ch = channels[lvl]
            blocks = nn.ModuleList()
            for r in range(num_res_blocks):
                in_ch = (prev_ch + skip_ch) if r == 0 else target_ch
                blocks.append(ResBlock(
                    in_ch=in_ch,
                    out_ch=target_ch,
                    time_emb_dim=time_emb_dim,
                    label_channels=label_channels,
                    spade_hidden=spade_hidden,
                ))
            self.up_resblocks.append(blocks)
            self.up_attns.append(
                SelfAttention2D(target_ch, num_heads=8) if attention_levels[lvl] else nn.Identity()
            )
            # Upsample after every level except the shallowest
            self.upsamplers.append(Upsample(target_ch) if lvl > 0 else nn.Identity())
            prev_ch = target_ch

        # --- Output head --- #
        self.out_norm = nn.GroupNorm(32, channels[0])
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        # Zero-init the output conv so the model starts by predicting exactly
        # zero noise — the trivial baseline. Non-zero predictions then come
        # only from learned signal, which speeds up gradient routing through
        # the SPADE-conditioned decoder. Standard trick from DiT (AdaLN-Zero)
        # and adopted by Imagen, SD v2 etc.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ----------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,            # (B, in_channels, H, W) noisy image
        timesteps: torch.Tensor,    # (B,) int64
        label: torch.Tensor,        # (B, label_channels, H, W) one-hot label
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)

        h = self.stem(x)
        skips: list[torch.Tensor] = []

        # Encoder
        for lvl in range(self.num_levels):
            for block in self.down_resblocks[lvl]:
                h = block(h, t_emb, label=None)   # encoder uses standard GroupNorm
            skips.append(h)
            h = self.downsamplers[lvl](h)

        # Bottleneck
        h = self.mid_block_1(h, t_emb, label=label)
        h = self.mid_attn(h)
        h = self.mid_block_2(h, t_emb, label=label)

        # Decoder
        for i, lvl in enumerate(reversed(range(self.num_levels))):
            skip = skips[lvl]
            # Concat the skip onto the first ResBlock's input
            h = torch.cat([h, skip], dim=1)
            for r, block in enumerate(self.up_resblocks[i]):
                h = block(h, t_emb, label=label)
                # After the first block the channel count is back to target;
                # no further concat needed for subsequent ResBlocks.
            h = self.up_attns[i](h)
            h = self.upsamplers[i](h)

        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)
