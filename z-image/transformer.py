# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image Transformer for TT hardware.

Self-contained transformer that copies the architecture from
diffusers' ZImageTransformer2DModel but eliminates all XLA graph-break
sources:
  - No complex64 (real-valued RoPE)
  - No len(tensor), no Python ifs on tensor values
  - No pad_sequence / split / variable-length list ops
  - No inspect.signature introspection in attention
  - Direct F.scaled_dot_product_attention (no dispatch_attention_fn)

All shapes are precomputed for B=1 at construction time.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        out_size,
        mid_size=None,
        frequency_embedding_size=256,
        t_scale=1.0,
        max_period=10000,
    ):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        # Precompute frequency table (with t_scale baked in) as a buffer
        # to avoid graph breaks from torch.arange/torch.exp in forward.
        half = frequency_embedding_size // 2
        freqs = (
            torch.exp(
                -math.log(max_period)
                * torch.arange(0, half, dtype=torch.float32)
                / half
            )
            * t_scale
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)
        return x * self.weight


class RealRopeEmbedder(nn.Module):
    """Real-valued RoPE. Precomputes cos/sin lookup tables as buffers."""

    def __init__(self, theta: float, axes_dims: List[int], axes_lens: List[int]):
        super().__init__()
        self.axes_dims = axes_dims
        for i, (d, e) in enumerate(zip(axes_dims, axes_lens)):
            freqs = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
            )
            timestep = torch.arange(e, device="cpu", dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            self.register_buffer(f"cos_{i}", freqs.cos())
            self.register_buffer(f"sin_{i}", freqs.sin())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Unrolled for 3 axes (always 3 for Z-Image)
        return torch.cat(
            [
                self.cos_0[ids[:, 0]],
                self.cos_1[ids[:, 1]],
                self.cos_2[ids[:, 2]],
                self.sin_0[ids[:, 0]],
                self.sin_1[ids[:, 1]],
                self.sin_2[ids[:, 2]],
            ],
            dim=-1,
        )


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Attention – no inspect.signature, no dispatch registry
# ---------------------------------------------------------------------------


def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Real-valued rotary embedding (no complex64)."""
    half = freqs_cis.shape[-1] // 2
    cos = freqs_cis[..., :half].unsqueeze(2)
    sin = freqs_cis[..., half:].unsqueeze(2)

    x = x_in.float().reshape(*x_in.shape[:-1], -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
    return out.type_as(x_in)


class Attention(nn.Module):
    """Self-attention with QKNorm and real-valued RoPE.

    Single class combining what diffusers splits across Attention,
    AttnProcessor, and dispatch_attention_fn. No introspection, no
    registry lookups, no graph breaks.
    """

    def __init__(self, dim, n_heads, qk_norm=True, eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(self, x, attn_mask=None, freqs_cis=None):
        q = self.to_q(x).unflatten(-1, (self.n_heads, self.head_dim))
        k = self.to_k(x).unflatten(-1, (self.n_heads, self.head_dim))
        v = self.to_v(x).unflatten(-1, (self.n_heads, self.head_dim))

        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = _apply_rotary_emb(q, freqs_cis)
            k = _apply_rotary_emb(k, freqs_cis)

        # [B, seq, heads, head_dim] -> [B, heads, seq, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )

        # [B, heads, seq, head_dim] -> [B, seq, dim]
        out = out.permute(0, 2, 1, 3).flatten(2, 3)
        out = out.to(x.dtype)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5, qk_norm=True, modulation=True):
        super().__init__()
        self.attention = Attention(dim, n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod
            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attn_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(self.ffn_norm1(x) * scale_mlp)
            )
        else:
            attn_out = self.attention(
                self.attention_norm1(x), attn_mask=attn_mask, freqs_cis=freqs_cis
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


# ---------------------------------------------------------------------------
# Top-level transformer
# ---------------------------------------------------------------------------


class ZImageTransformer(nn.Module):
    """Complete Z-Image transformer for B=1 with fixed shapes.

    Precomputes all padding, position IDs, and attention masks at
    construction time so forward() is pure tensor math.

    Args:
        source: The original ZImageTransformer2DModel to copy weights from.
        cap_len: Number of non-padding caption tokens (e.g. 18).
        image_shape: (C, F, H, W) of the latent input (e.g. (16, 1, 160, 90)).
        patch_size: Spatial patch size (default 2).
        f_patch_size: Temporal patch size (default 1).
    """

    def __init__(self, source, cap_len, image_shape, patch_size=2, f_patch_size=1):
        super().__init__()
        cfg = source.config

        # Core dimensions
        dim = cfg.dim
        n_heads = cfg.n_heads
        norm_eps = cfg.norm_eps
        qk_norm = cfg.qk_norm
        self.out_channels = cfg.in_channels

        # Timestep embedder (t_scale baked into frequency table)
        self.t_embedder = TimestepEmbedder(
            min(dim, ADALN_EMBED_DIM), mid_size=1024, t_scale=cfg.t_scale
        )

        # Embedders
        self.x_embedder = nn.Linear(
            f_patch_size * patch_size * patch_size * cfg.in_channels, dim, bias=True
        )
        self.cap_embedder = nn.Sequential(
            RMSNorm(cfg.cap_feat_dim, eps=norm_eps),
            nn.Linear(cfg.cap_feat_dim, dim, bias=True),
        )

        # Pad tokens
        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        # RoPE
        self.rope_embedder = RealRopeEmbedder(
            theta=cfg.rope_theta,
            axes_dims=cfg.axes_dims,
            axes_lens=cfg.axes_lens,
        )

        # Transformer layers
        self.noise_refiner = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, norm_eps, qk_norm, modulation=True)
                for _ in range(cfg.n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, norm_eps, qk_norm, modulation=False)
                for _ in range(cfg.n_refiner_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, norm_eps, qk_norm, modulation=True)
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_layer = FinalLayer(
            dim, patch_size * patch_size * f_patch_size * self.out_channels
        )

        # ---------------------------------------------------------------
        # Copy weights from the original diffusers model
        # ---------------------------------------------------------------
        self._copy_weights(source, patch_size, f_patch_size)

        # Match the source model's dtype (e.g. bfloat16)
        source_dtype = next(source.parameters()).dtype
        self.to(source_dtype)

        # ---------------------------------------------------------------
        # Precompute fixed-shape constants for B=1
        # ---------------------------------------------------------------
        C, F_, H, W = image_shape
        pH = pW = patch_size
        pF = f_patch_size

        self._C = C
        self._F = F_
        self._H = H
        self._W = W
        self._pH = pH
        self._pW = pW
        self._pF = pF

        F_tokens = F_ // pF
        H_tokens = H // pH
        W_tokens = W // pW
        self._F_tokens = F_tokens
        self._H_tokens = H_tokens
        self._W_tokens = W_tokens

        cap_padding_len = (-cap_len) % SEQ_MULTI_OF
        cap_total = cap_len + cap_padding_len
        self._cap_ori_len = cap_len
        self._cap_padding_len = cap_padding_len

        image_ori_len = F_tokens * H_tokens * W_tokens
        image_padding_len = (-image_ori_len) % SEQ_MULTI_OF
        image_total = image_ori_len + image_padding_len
        self._image_ori_len = image_ori_len
        self._image_padding_len = image_padding_len

        # Position IDs
        image_ori_pos_ids = torch.stack(
            torch.meshgrid(
                torch.arange(
                    cap_total + 1, cap_total + 1 + F_tokens, dtype=torch.int32
                ),
                torch.arange(0, H_tokens, dtype=torch.int32),
                torch.arange(0, W_tokens, dtype=torch.int32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(image_ori_len, 3)
        zero_pad = torch.zeros(image_padding_len, 3, dtype=torch.int32)
        self.register_buffer(
            "image_pos_ids", torch.cat([image_ori_pos_ids, zero_pad], dim=0)
        )

        cap_pos_ids = torch.stack(
            torch.meshgrid(
                torch.arange(1, 1 + cap_total, dtype=torch.int32),
                torch.arange(0, 1, dtype=torch.int32),
                torch.arange(0, 1, dtype=torch.int32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(cap_total, 3)
        self.register_buffer("cap_pos_ids", cap_pos_ids)

        # Pad masks
        self.register_buffer(
            "image_pad_mask",
            torch.cat(
                [
                    torch.zeros(image_ori_len, dtype=torch.bool),
                    torch.ones(image_padding_len, dtype=torch.bool),
                ]
            ),
        )
        self.register_buffer(
            "cap_pad_mask",
            torch.cat(
                [
                    torch.zeros(cap_len, dtype=torch.bool),
                    torch.ones(cap_padding_len, dtype=torch.bool),
                ]
            ),
        )

        # Attention masks
        self.register_buffer(
            "x_attn_mask", torch.ones(1, image_total, dtype=torch.bool)
        )
        self.register_buffer(
            "cap_attn_mask", torch.ones(1, cap_total, dtype=torch.bool)
        )
        self.register_buffer(
            "unified_attn_mask",
            torch.ones(1, image_total + cap_total, dtype=torch.bool),
        )

    def _copy_weights(self, source, patch_size, f_patch_size):
        """Copy weights from a diffusers ZImageTransformer2DModel."""
        key = f"{patch_size}-{f_patch_size}"

        # Simple 1:1 copies (t_embedder: only copy MLP weights, freqs buffer is precomputed)
        self.t_embedder.mlp.load_state_dict(source.t_embedder.mlp.state_dict())
        self.x_embedder.load_state_dict(source.all_x_embedder[key].state_dict())
        self.cap_embedder.load_state_dict(source.cap_embedder.state_dict())
        self.x_pad_token.data.copy_(source.x_pad_token.data)
        self.cap_pad_token.data.copy_(source.cap_pad_token.data)
        self.final_layer.load_state_dict(source.all_final_layer[key].state_dict())

        # Copy transformer blocks — need to map diffusers Attention -> our Attention
        for dst_list, src_list in [
            (self.noise_refiner, source.noise_refiner),
            (self.context_refiner, source.context_refiner),
            (self.layers, source.layers),
        ]:
            for dst_block, src_block in zip(dst_list, src_list):
                # Norms and FFN copy directly
                dst_block.attention_norm1.load_state_dict(
                    src_block.attention_norm1.state_dict()
                )
                dst_block.ffn_norm1.load_state_dict(src_block.ffn_norm1.state_dict())
                dst_block.attention_norm2.load_state_dict(
                    src_block.attention_norm2.state_dict()
                )
                dst_block.ffn_norm2.load_state_dict(src_block.ffn_norm2.state_dict())
                dst_block.feed_forward.load_state_dict(
                    src_block.feed_forward.state_dict()
                )

                if dst_block.modulation:
                    dst_block.adaLN_modulation.load_state_dict(
                        src_block.adaLN_modulation.state_dict()
                    )

                # Attention: diffusers stores Q/K/V/out differently
                src_attn = src_block.attention
                dst_attn = dst_block.attention
                dst_attn.to_q.load_state_dict(src_attn.to_q.state_dict())
                dst_attn.to_k.load_state_dict(src_attn.to_k.state_dict())
                dst_attn.to_v.load_state_dict(src_attn.to_v.state_dict())
                # diffusers to_out is ModuleList [Linear, Dropout?]
                dst_attn.to_out.load_state_dict(src_attn.to_out[0].state_dict())

                if dst_attn.norm_q is not None:
                    dst_attn.norm_q.load_state_dict(src_attn.norm_q.state_dict())
                    dst_attn.norm_k.load_state_dict(src_attn.norm_k.state_dict())

    @property
    def dtype(self):
        return self.x_embedder.weight.dtype

    def forward(self, image, t, cap_feat):
        """Forward pass for B=1 with precomputed shapes.

        Args:
            image: [C, F, H, W] latent tensor.
            t: [1] normalized timestep.
            cap_feat: [cap_len, cap_feat_dim] caption features.

        Returns:
            [C, F, H, W] denoised output.
        """
        C = self._C
        F_tokens = self._F_tokens
        H_tokens = self._H_tokens
        W_tokens = self._W_tokens
        pF = self._pF
        pH = self._pH
        pW = self._pW
        image_ori_len = self._image_ori_len
        image_padding_len = self._image_padding_len
        cap_padding_len = self._cap_padding_len

        # 1. Timestep embedding (t_scale baked into freqs buffer)
        t_emb = self.t_embedder(t)

        # 2. Patchify image: [C, F, H, W] -> [image_ori_len, patch_dim]
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
            image_ori_len, pF * pH * pW * C
        )

        # 3. Pad image features
        image = torch.cat([image, image[-1:].expand(image_padding_len, -1)], dim=0)

        # 4. Pad caption features
        cap_feat = torch.cat(
            [cap_feat, cap_feat[-1:].expand(cap_padding_len, -1)], dim=0
        )

        # 5. Embed
        x = self.x_embedder(image)
        adaln_input = t_emb.type_as(x)
        x = torch.where(self.image_pad_mask.unsqueeze(-1), self.x_pad_token, x)
        cap = self.cap_embedder(cap_feat)
        cap = torch.where(self.cap_pad_mask.unsqueeze(-1), self.cap_pad_token, cap)

        # 6. RoPE
        x_freqs_cis = self.rope_embedder(self.image_pos_ids)
        cap_freqs_cis = self.rope_embedder(self.cap_pos_ids)

        # 7. Add batch dim
        x = x.unsqueeze(0)
        x_freqs_cis = x_freqs_cis.unsqueeze(0)
        cap = cap.unsqueeze(0)
        cap_freqs_cis = cap_freqs_cis.unsqueeze(0)

        # 8. Noise refiner (unrolled)
        x = self.noise_refiner[0](x, self.x_attn_mask, x_freqs_cis, adaln_input)
        x = self.noise_refiner[1](x, self.x_attn_mask, x_freqs_cis, adaln_input)

        # 9. Context refiner (unrolled)
        cap = self.context_refiner[0](cap, self.cap_attn_mask, cap_freqs_cis)
        cap = self.context_refiner[1](cap, self.cap_attn_mask, cap_freqs_cis)

        # 10. Unified layers (unrolled)
        unified = torch.cat([x, cap], dim=1)
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)

        unified = self.layers[0](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[1](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[2](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[3](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[4](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[5](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[6](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[7](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[8](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[9](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[10](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[11](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[12](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[13](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[14](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[15](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[16](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[17](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[18](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[19](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[20](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[21](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[22](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[23](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[24](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[25](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[26](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[27](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[28](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )
        unified = self.layers[29](
            unified, self.unified_attn_mask, unified_freqs_cis, adaln_input
        )

        # 11. Final layer + unpatchify
        unified = self.final_layer(unified, adaln_input)
        out = unified[0, :image_ori_len]
        out = (
            out.view(F_tokens, H_tokens, W_tokens, pF, pH, pW, self.out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(self.out_channels, self._F, self._H, self._W)
        )

        return out
