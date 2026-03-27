# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simplified DiT3D backbone for Diffusion Forcing Transformer (DFoT).

Adapted from: https://github.com/kwsong0113/diffusion-forcing-transformer

This module provides a self-contained DiT3D (Diffusion Transformer 3D) backbone
that can load pretrained weights from the DFoT checkpoints hosted on HuggingFace.
The architecture implements AdaLN-Zero conditioned transformer blocks with 3D
rotary position embeddings for video diffusion.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import Mlp, PatchEmbed


@dataclass
class DiT3DConfig:
    """Configuration for the DiT3D backbone."""

    in_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    max_tokens: int = 16
    spatial_resolution: int = 32
    external_cond_dim: int = 0


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepMLP(nn.Module):
    """Two-layer MLP for timestep embedding, matching diffusers TimestepEmbedding."""

    def __init__(self, in_channels, time_embed_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class StochasticTimeEmbedding(nn.Module):
    """Noise level embedding: sinusoidal encoding followed by MLP projection."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.timesteps = SinusoidalEmbedding(frequency_embedding_size)
        self.embedding = TimestepMLP(frequency_embedding_size, hidden_size)

    def forward(self, t):
        t_flat = t.reshape(-1)
        t_emb = self.timesteps(t_flat)
        return self.embedding(t_emb)


class RotaryEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for spatiotemporal attention."""

    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.empty(0), persistent=True)

    def forward(self, q, k):
        if self.freqs.numel() == 0:
            return q, k

        seq_len = q.shape[-2]
        freqs = self.freqs[:seq_len]

        cos = freqs.cos()
        sin = freqs.sin()

        rot_dim = freqs.shape[-1] * 2

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

        q_rot = self._apply_rotary(q_rot, cos, sin)
        k_rot = self._apply_rotary(k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    @staticmethod
    def _apply_rotary(x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with optional 3D RoPE."""

    def __init__(self, hidden_size, num_heads, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

        if use_rope:
            self.rope = RotaryEmbedding3D()
        else:
            self.rope = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization with Zero initialization (AdaLN-Zero).

    Produces shift, scale, and gate from the conditioning signal.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size),
        )

    def forward(self, x, c):
        mod = self.modulation(c)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=True):
        super().__init__()
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(hidden_size, num_heads, use_rope=use_rope)
        self.norm2 = AdaLayerNormZero(hidden_size)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
        )

    def forward(self, x, c):
        x_norm, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x_norm)
        x_norm, gate_mlp = self.norm2(x, c)
        x = x + gate_mlp * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final layer: AdaLN + linear projection to patch space."""

    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x, c):
        mod = self.modulation(c)
        shift, scale = mod.chunk(2, dim=-1)
        x = self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


class DiTBase(nn.Module):
    """Core DiT transformer with positional embeddings and transformer blocks."""

    def __init__(self, cfg: DiT3DConfig, num_patches: int):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        out_features = cfg.patch_size**2 * cfg.in_channels

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    use_rope=True,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, out_features)

    def forward(self, x, c):
        for block in self.blocks:
            x = block(x, c)

        # Use mean of conditioning for final layer
        c_mean = c.mean(dim=1, keepdim=True).expand_as(c)
        x = self.final_layer(x, c_mean)
        return x


class DiT3D(nn.Module):
    """DiT3D backbone for video diffusion.

    Takes noisy video frames and per-token noise levels as input,
    and predicts the denoised output in the same spatiotemporal shape.

    Args:
        cfg: DiT3DConfig with architecture hyperparameters.
    """

    def __init__(self, cfg: DiT3DConfig):
        super().__init__()
        self.cfg = cfg

        num_patches_spatial = (cfg.spatial_resolution // cfg.patch_size) ** 2
        self.num_patches_spatial = num_patches_spatial

        # Patch embedding: converts (B*T, C, H, W) -> (B*T, num_patches, hidden_size)
        self.patch_embedder = PatchEmbed(
            img_size=cfg.spatial_resolution,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_channels,
            embed_dim=cfg.hidden_size,
            bias=True,
        )

        # Noise level embedding
        self.noise_level_pos_embedding = StochasticTimeEmbedding(cfg.hidden_size)

        # External condition projection (e.g., camera poses)
        if cfg.external_cond_dim > 0:
            self.external_cond_embedding = nn.Sequential(
                nn.Linear(cfg.external_cond_dim, cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
            )
        else:
            self.external_cond_embedding = None

        # Core transformer
        self.dit_base = DiTBase(cfg, num_patches_spatial)

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        external_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the DiT3D backbone.

        Args:
            x: Noisy video frames, shape (B, T, C, H, W).
            noise_levels: Per-token noise levels, shape (B, T).
            external_cond: Optional external conditioning (e.g., camera poses),
                           shape (B, T, D).

        Returns:
            Predicted output, shape (B, T, C, H, W).
        """
        B, T, C, H, W = x.shape

        # Patchify: (B, T, C, H, W) -> (B*T, C, H, W) -> (B*T, num_patches, hidden)
        x_flat = rearrange(x, "b t c h w -> (b t) c h w")
        x_tokens = self.patch_embedder(x_flat)  # (B*T, num_patches, hidden)
        x_tokens = rearrange(
            x_tokens, "(b t) n d -> b (t n) d", b=B, t=T
        )  # (B, T*num_patches, hidden)

        # Noise level embedding: (B, T) -> (B, T, hidden) -> (B, T*num_patches, hidden)
        noise_emb = self.noise_level_pos_embedding(noise_levels)  # (B*T, hidden)
        noise_emb = rearrange(noise_emb, "(b t) d -> b t d", b=B, t=T)

        # Add external conditioning if provided
        if external_cond is not None and self.external_cond_embedding is not None:
            ext_emb = self.external_cond_embedding(external_cond)  # (B, T, hidden)
            noise_emb = noise_emb + ext_emb

        # Repeat noise embedding for each spatial patch
        cond = noise_emb.unsqueeze(2).expand(
            B, T, self.num_patches_spatial, -1
        )  # (B, T, num_patches, hidden)
        cond = rearrange(cond, "b t n d -> b (t n) d")  # (B, T*num_patches, hidden)

        # Transformer forward
        x_tokens = self.dit_base(x_tokens, cond)  # (B, T*num_patches, patch_dim)

        # Unpatchify: (B, T*num_patches, patch_dim) -> (B, T, C, H, W)
        x_tokens = rearrange(
            x_tokens, "b (t n) d -> (b t) n d", t=T, n=self.num_patches_spatial
        )
        output = self._unpatchify(x_tokens, H, W)  # (B*T, C, H, W)
        output = rearrange(output, "(b t) c h w -> b t c h w", b=B, t=T)

        return output

    def _unpatchify(self, x, h, w):
        """Reconstruct spatial dimensions from patch tokens.

        Args:
            x: Patch tokens, shape (BT, num_patches, patch_size^2 * C).
            h: Original height.
            w: Original width.

        Returns:
            Reconstructed tensor, shape (BT, C, H, W).
        """
        p = self.cfg.patch_size
        c = self.cfg.in_channels
        h_patches = h // p
        w_patches = w // p
        x = rearrange(
            x,
            "bt (hp wp) (p1 p2 c) -> bt c (hp p1) (wp p2)",
            hp=h_patches,
            wp=w_patches,
            p1=p,
            p2=p,
            c=c,
        )
        return x


def load_dit3d_from_checkpoint(checkpoint_path: str, cfg: DiT3DConfig) -> DiT3D:
    """Load a DiT3D model from a PyTorch Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        cfg: Model configuration.

    Returns:
        DiT3D model with loaded weights.
    """
    model = DiT3D(cfg)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract backbone weights (strip Lightning module prefix)
    prefix = "diffusion_model.model."
    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            backbone_state[k[len(prefix) :]] = v

    model.load_state_dict(backbone_state, strict=False)
    return model
