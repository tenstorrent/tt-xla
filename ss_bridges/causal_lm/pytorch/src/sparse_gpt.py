# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SparseGPT model implementation for the bridges procedure from Gao et al. (2025).

This is a minimal implementation of the SparseGPT architecture that matches
the configuration format used by jacobcd52's sparse transformer models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class SparseAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, d_model, n_heads, d_head, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=bias)
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=bias)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SparseMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d_model, d_mlp, bias=True):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_mlp, bias=bias)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=bias)

    def forward(self, x):
        return self.down_proj(F.gelu(self.up_proj(x)))


class SparseTransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, d_model, n_heads, d_head, d_mlp, bias=True):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SparseAttention(d_model, n_heads, d_head, bias)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SparseMLP(d_model, d_mlp, bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SparseGPT(nn.Module):
    """Sparse GPT model for causal language modeling.

    Architecture based on the config format from jacobcd52's sparse transformers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config["d_model"]
        n_heads = d_model // config["d_head"]

        self.embed = nn.Embedding(config["vocab_size"], d_model)
        self.layers = nn.ModuleList(
            [
                SparseTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_head=config["d_head"],
                    d_mlp=config["d_mlp"],
                    bias=config.get("use_bias", True),
                )
                for _ in range(config["n_layer"])
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, config["vocab_size"], bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
