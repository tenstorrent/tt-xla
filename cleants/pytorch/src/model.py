# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CleanTS model architecture for time series forecasting.

Reconstructed from EINK/CleanTS-65M weights: encoder-only transformer
with SwiGLU FFN, QK normalization, RMSNorm, and quantile output heads.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc_gate = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.silu(self.fc_gate(x)) * self.fc1(x)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dropout_p: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.attn_dropout = nn.Dropout(attn_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, attn_dropout_p)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout_p)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ResidualMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features, hidden_features)
        self.output_layer = nn.Linear(hidden_features, out_features)
        self.residual_layer = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(F.silu(self.hidden_layer(x))) + self.residual_layer(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout_p, attn_dropout_p)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CleanTS(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 9,
        patch_size: int = 32,
        max_seq_len: int = 512,
        quantile_levels: Optional[List[float]] = None,
        scaling: bool = True,
        dropout_p: float = 0.0,
        attn_dropout_p: float = 0.0,
    ):
        super().__init__()

        if quantile_levels is None:
            quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.d_model = d_model
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.quantile_levels = quantile_levels
        self.scaling = scaling
        self.num_quantiles = len(quantile_levels)

        num_heads = d_model // 64
        d_ff = 2048  # Derived from weight shapes

        self.in_proj = ResidualMLP(patch_size, d_model, d_model)
        self.mask_encoding = nn.Embedding(1, d_model)
        self.encoder = Encoder(
            d_model, num_layers, num_heads, d_ff, dropout_p, attn_dropout_p
        )
        self.out_proj = ResidualMLP(d_model, d_model, patch_size * self.num_quantiles)

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = past_values.shape

        # Compute scaling factors
        if self.scaling:
            loc = past_values.mean(dim=-1, keepdim=True)
            scale = past_values.std(dim=-1, keepdim=True).clamp(min=1e-5)
            past_values = (past_values - loc) / scale
        else:
            loc = torch.zeros(B, 1, device=past_values.device, dtype=past_values.dtype)
            scale = torch.ones(B, 1, device=past_values.device, dtype=past_values.dtype)

        # Patch the time series: (B, T) -> (B, num_patches, patch_size)
        num_patches = T // self.patch_size
        x = past_values[:, : num_patches * self.patch_size].reshape(
            B, num_patches, self.patch_size
        )

        # Project patches to d_model
        x = self.in_proj(x)

        # Encode
        x = self.encoder(x)

        # Project to quantile outputs: (B, num_patches, patch_size * num_quantiles)
        x = self.out_proj(x)

        # Reshape: (B, num_patches, num_quantiles, patch_size)
        x = x.reshape(B, num_patches, self.num_quantiles, self.patch_size)

        # Un-scale
        if self.scaling:
            x = x * scale.unsqueeze(-1).unsqueeze(-1) + loc.unsqueeze(-1).unsqueeze(-1)

        return x
