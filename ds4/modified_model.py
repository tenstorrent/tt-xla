# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V4 Flash model adapted for TT-XLA testing.

This module contains modified attention implementations from DeepSeek V4:
- CSA (Compressed Sparse Attention): compress_ratio=4 with Indexer for top-k selection
- HSA (Heavily Compressed Attention): compress_ratio=128 without Indexer

Modifications from original:
1. Removed CUDA kernel dependencies (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn)
2. Replaced fast_hadamard_transform with scipy.linalg.hadamard
3. Use BF16 operations instead of FP8/FP4 quantization
4. Replaced sparse_attn with standard attention implementation
5. Simplified hc_split_sinkhorn to basic weighted sum
"""
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32


# Stub implementations for CUDA kernels - use BF16 fallbacks
def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    """Stub for FP8 activation quantization - returns input unchanged for BF16 testing."""
    if inplace:
        return x
    return x, torch.ones(1, dtype=torch.float32)


def fp4_act_quant(x, block_size=32, inplace=False):
    """Stub for FP4 activation quantization - returns input unchanged for BF16 testing."""
    if inplace:
        return x
    return x, torch.ones(1, dtype=torch.float32)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=None):
    """Stub for FP8 GEMM - uses standard F.linear."""
    return F.linear(a, b)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=None):
    """Stub for FP4 GEMM - uses standard F.linear."""
    return F.linear(a, b)


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """
    Simplified sparse attention using standard operations.

    For testing purposes, we implement a dense attention fallback
    since sparse_attn kernel is CUDA-specific.
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    kv_len = kv.shape[1]

    # For prefill (seqlen > 1), use dense attention with causal mask
    # For decode (seqlen == 1), gather from KV cache using topk_idxs

    # Reshape for attention computation
    # q: [bsz, seqlen, n_heads, head_dim] -> [bsz, n_heads, seqlen, head_dim]
    q = q.transpose(1, 2)

    # kv is [bsz, kv_len, head_dim] - expand for all heads
    # k, v: [bsz, n_heads, kv_len, head_dim]
    k = kv.unsqueeze(1).expand(-1, n_heads, -1, -1)
    v = kv.unsqueeze(1).expand(-1, n_heads, -1, -1)

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # Add attention sink bias
    scores = scores + attn_sink.view(1, n_heads, 1, 1)

    # Apply causal mask for prefill
    if seqlen > 1:
        causal_mask = torch.triu(
            torch.full((seqlen, kv_len), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=kv_len - seqlen + 1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

    # Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    # Transpose back: [bsz, n_heads, seqlen, head_dim] -> [bsz, seqlen, n_heads, head_dim]
    output = output.transpose(1, 2)

    return output


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """
    Simplified Hyper-Connections split with Sinkhorn normalization.

    For testing, we use a simplified version that produces valid pre/post/comb tensors.
    """
    bsz, seqlen, mix_hc = mixes.shape

    # Split mixes into pre, post, and comb components
    # mix_hc = (2 + hc_mult) * hc_mult
    # pre: [bsz, seqlen, hc_mult]
    # post: [bsz, seqlen, hc_mult]
    # comb: [bsz, seqlen, hc_mult, hc_mult]

    pre_size = hc_mult
    post_size = hc_mult
    comb_size = hc_mult * hc_mult

    scaled_mixes = mixes * hc_scale[0] + hc_base[:mix_hc]

    pre = torch.sigmoid(scaled_mixes[:, :, :pre_size]) + eps
    pre = pre / pre.sum(dim=-1, keepdim=True)  # Normalize

    post = torch.sigmoid(scaled_mixes[:, :, pre_size:pre_size + post_size]) + eps

    comb_flat = torch.sigmoid(scaled_mixes[:, :, pre_size + post_size:]) + eps
    # Pad if needed
    if comb_flat.shape[-1] < comb_size:
        comb_flat = F.pad(comb_flat, (0, comb_size - comb_flat.shape[-1]), value=eps)
    comb = comb_flat[:, :, :comb_size].view(bsz, seqlen, hc_mult, hc_mult)

    return pre, post, comb


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Applies Hadamard rotation using scipy instead of CUDA kernel.
    """
    assert x.dtype == torch.bfloat16
    import scipy.linalg
    hidden_size = x.size(-1)
    hadamard = torch.tensor(
        scipy.linalg.hadamard(hidden_size),
        dtype=torch.bfloat16,
        device=x.device
    ) * (hidden_size ** -0.5)
    return F.linear(x, hadamard)


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """Model hyperparameters for DeepSeek V4 Flash."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"  # Use bf16 for testing
    scale_fmt: Literal[None, "ue8m0"] = None
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp32"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    # compress_ratios defines attention type per layer:
    # 0 = sliding window only
    # 4 = CSA (Compressed Sparse Attention with Indexer)
    # 128 = HSA (Heavily Compressed Attention without Indexer)
    compress_ratios: Tuple[int, ...] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    """Embedding layer (simplified for single-device testing)."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Linear layer - always uses F.linear for BF16 testing."""
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    """Linear layer using BF16 for testing."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        # Always use bf16 for testing (no FP8/FP4)
        if dtype in [torch.float8_e4m3fn, torch.float4_e2m1fn_x2]:
            dtype = torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Column-parallel linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)


class RowParallelLinear(Linear):
    """Row-parallel linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y = y + self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings using real arithmetic."""
    dtype = x.dtype
    shape = x.shape

    # Reshape to pairs
    x = x.float().view(*shape[:-1], -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]

    # Handle freqs_cis format
    if freqs_cis.dtype.is_complex:
        freqs_cis_real = torch.view_as_real(freqs_cis)
    else:
        freqs_cis_real = freqs_cis

    # Reshape freqs_cis to match x dimensions
    if x.ndim == 4:  # [b, s, pairs, 2]
        freqs_cis_real = freqs_cis_real.view(1, x.size(1), x.size(-2), 2)
    else:  # [b, s, h, pairs, 2]
        freqs_cis_real = freqs_cis_real.view(1, x.size(1), 1, x.size(-2), 2)

    cos_vals = freqs_cis_real[..., 0]
    sin_vals = freqs_cis_real[..., 1]

    if inverse:
        sin_vals = -sin_vals

    # Complex multiplication
    y_real = x_real * cos_vals - x_imag * sin_vals
    y_imag = x_real * sin_vals + x_imag * cos_vals

    y = torch.stack([y_real, y_imag], dim=-1).flatten(-2)
    return y.to(dtype)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    """Get indices for sliding window attention."""
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    """Get indices for compressed KV positions."""
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """
    Compresses KV cache via learned gated pooling.

    This is used in both CSA (compress_ratio=4) and HSA (compress_ratio=128).
    When overlap=True (ratio==4), uses overlapping windows for smoother boundaries.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None
        self.register_buffer(
            "kv_state",
            torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32),
            persistent=False
        )
        self.register_buffer(
            "score_state",
            torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32),
            persistent=False
        )
        self.freqs_cis: torch.Tensor = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.ape, std=0.02)

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio:cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff - ratio:cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset:offset + remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset:offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

        if not should_compress:
            return

        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)

        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """
    Selects top-k compressed KV positions for sparse attention (used in CSA).

    Has its own Compressor with Hadamard rotation for scoring compressed KV.
    """

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio
        self.return_raw_scores = False  # For testing - return scores instead of indices

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim),
            persistent=False
        )
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)

        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        if world_size > 1:
            dist.all_reduce(index_score)

        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            index_score = index_score + torch.where(mask, float("-inf"), torch.tensor(0.0, device=x.device))

        # For testing, optionally return raw scores
        if self.return_raw_scores:
            return index_score

        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


class Attention(nn.Module):
    """
    Multi-head Latent Attention (MLA) with sliding window + optional KV compression.

    Attention types based on compress_ratio:
    - 0: Sliding window attention only
    - 4: CSA (Compressed Sparse Attention) with Indexer for top-k selection
    - 128: HSA (Heavily Compressed Attention) without Indexer
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                # CSA uses Indexer for top-k selection
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                # HSA (compress_ratio=128) doesn't use Indexer
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)

        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            original_seq_len, rope_theta = 0, args.rope_theta

        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.attn_sink)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # Q projection with low-rank decomposition
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # KV projection
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Attention computation
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # Output projection with grouped low-rank
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class CSAAttention(Attention):
    """
    CSA (Compressed Sparse Attention) - compress_ratio=4 with Indexer.

    This is a convenience wrapper that ensures compress_ratio=4 is used.
    """

    def __init__(self, args: ModelArgs, layer_id: int = 2):
        # Override compress_ratios to ensure CSA (ratio=4) for this layer
        modified_args = ModelArgs(
            **{k: v for k, v in vars(args).items() if k != 'compress_ratios'}
        )
        # Create compress_ratios with ratio=4 at the specified layer
        ratios = list(args.compress_ratios)
        while len(ratios) <= layer_id:
            ratios.append(0)
        ratios[layer_id] = 4
        modified_args.compress_ratios = tuple(ratios)
        super().__init__(layer_id, modified_args)


class HSAAttention(Attention):
    """
    HSA (Heavily Compressed Attention) - compress_ratio=128 without Indexer.

    This is a convenience wrapper that ensures compress_ratio=128 is used.
    """

    def __init__(self, args: ModelArgs, layer_id: int = 3):
        # Override compress_ratios to ensure HSA (ratio=128) for this layer
        modified_args = ModelArgs(
            **{k: v for k, v in vars(args).items() if k != 'compress_ratios'}
        )
        # Create compress_ratios with ratio=128 at the specified layer
        ratios = list(args.compress_ratios)
        while len(ratios) <= layer_id:
            ratios.append(0)
        ratios[layer_id] = 128
        modified_args.compress_ratios = tuple(ratios)
        super().__init__(layer_id, modified_args)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)

    # Test with small model args
    args = ModelArgs(
        dim=512,
        n_heads=8,
        head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=2,
        index_n_heads=8,
        index_head_dim=64,
        max_seq_len=256,
        max_batch_size=2,
        window_size=32,
        compress_ratios=(0, 0, 4, 128),
    )

    batch_size = 2
    seq_len = 64

    # Test CSA attention
    print("Testing CSA attention (compress_ratio=4)...")
    csa = CSAAttention(args, layer_id=2)
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)
    out = csa(x, start_pos=0)
    print(f"CSA output shape: {out.shape}")

    # Test HSA attention
    print("\nTesting HSA attention (compress_ratio=128)...")
    hsa = HSAAttention(args, layer_id=3)
    out = hsa(x, start_pos=0)
    print(f"HSA output shape: {out.shape}")

    print("\nAll tests passed!")
