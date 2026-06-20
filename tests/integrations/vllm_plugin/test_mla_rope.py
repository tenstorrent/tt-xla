# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device numerical check for the MLA decoupled YaRN RoPE.

DeepSeek-V2/V3 MLA applies a decoupled rotary embedding to ``q_pe`` and ``k_pe``
inside ``MultiHeadLatentAttentionWrapper.forward`` (``mla.py:158``) *before* the
attention impl runs:

    q[..., qk_nope:], k_pe = self.rotary_emb(positions, q[..., qk_nope:], k_pe)

``test_mla_attention_impl.py`` feeds the impl already-rotated, *random* ``q_pe`` /
``k_pe``, so it never exercises this rotation — a wrong YaRN rope (or a bad TT
lowering of the gather + GPT-J rotate) would be invisible there but would
corrupt every generated token end-to-end (the positional garbage seen in
``test_tensor_parallel_mla_prefill_decode``).

We reproduce ``DeepseekScalingRotaryEmbedding``'s exact YaRN cos/sin cache and
``forward_native`` rotation (the same pure helpers vLLM uses), then compare the
rotation executed on the TT device against the CPU result, for both a prefill
(arange positions) and a decode (single offset position) call. Building the real
``CustomOp`` module here would require a full ``VllmConfig`` context; the math is
identical, and the point is to validate TT's execution of the rope ops.
"""
import math

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from vllm.model_executor.layers.rotary_embedding.common import (
    rotate_gptj,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)

REQUIRED_PCC = 0.99

# DeepSeek-V2-Lite rope config (from its HF config).
_QK_ROPE_HEAD_DIM = 64  # rotary_dim == head_size for the decoupled pe
_NUM_HEADS = 16
_BASE = 10000.0
_FACTOR = 40.0  # yarn scaling factor
_ORIGINAL_MAX_POSITION = 4096
_BETA_FAST = 32
_BETA_SLOW = 1
_MSCALE = 0.707
_MSCALE_ALL_DIM = 0.707
_EXTRAPOLATION_FACTOR = 1.0
_ATTN_FACTOR = 1.0


def _yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _compute_cos_sin_cache() -> torch.Tensor:
    """Replicates DeepseekScalingRotaryEmbedding._compute_cos_sin_cache."""
    rotary_dim = _QK_ROPE_HEAD_DIM
    pos_freqs = _BASE ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (_FACTOR * pos_freqs)
    low, high = yarn_find_correction_range(
        _BETA_FAST, _BETA_SLOW, rotary_dim, _BASE, _ORIGINAL_MAX_POSITION
    )
    inv_freq_mask = (
        1 - yarn_linear_ramp_mask(low, high, rotary_dim // 2, dtype=torch.float)
    ) * _EXTRAPOLATION_FACTOR
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    mscale = (
        _yarn_get_mscale(_FACTOR, _MSCALE)
        / _yarn_get_mscale(_FACTOR, _MSCALE_ALL_DIM)
        * _ATTN_FACTOR
    )
    t = torch.arange(_ORIGINAL_MAX_POSITION * _FACTOR, dtype=torch.float32)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos() * mscale
    sin = freqs.sin() * mscale
    return torch.cat((cos, sin), dim=-1)


def _apply_rope(positions, q_pe, k_pe, cos_sin_cache):
    """Replicates DeepseekScalingRotaryEmbedding.forward_native (is_neox=False).

    q_pe: [tokens, N, R];  k_pe: [tokens, 1, R];  positions: [tokens].
    rotary_dim == head_size here, so there is no pass-through split.
    """
    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
    q_rot = q_pe * cos + rotate_gptj(q_pe) * sin
    k_rot = k_pe * cos + rotate_gptj(k_pe) * sin
    return q_rot, k_rot


def _pcc(device_out: torch.Tensor, golden: torch.Tensor) -> float:
    x = device_out.flatten().float()
    y = golden.flatten().float()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    return 1.0 if denom == 0 else float((vx @ vy) / denom)


@pytest.mark.nightly
@pytest.mark.parametrize("positions_desc", ["prefill", "decode"])
def test_mla_yarn_rope_matches_cpu(positions_desc):
    """TT-device YaRN rope must match the CPU rope for q_pe and k_pe."""
    xr.set_device_type("TT")

    seq_len = 32
    R = _QK_ROPE_HEAD_DIM
    N = _NUM_HEADS
    dtype = torch.bfloat16

    torch.manual_seed(0)
    # Shapes exactly as passed at mla.py:158.
    q_pe = (torch.randn(seq_len, N, R) / math.sqrt(R)).to(dtype)
    k_pe = (torch.randn(seq_len, 1, R) / math.sqrt(R)).to(dtype)
    if positions_desc == "prefill":
        positions = torch.arange(seq_len, dtype=torch.int64)
    else:
        # Decode: one token per user at a non-trivial absolute position.
        positions = torch.full((seq_len,), 137, dtype=torch.int64)

    cache = _compute_cos_sin_cache().to(dtype)

    # ----- CPU golden -----
    q_gold, k_gold = _apply_rope(positions, q_pe, k_pe, cache)

    # ----- TT device -----
    device = torch_xla.device()
    q_dev, k_dev = _apply_rope(
        positions.to(device),
        q_pe.to(device),
        k_pe.to(device),
        cache.to(device),
    )
    torch_xla.sync()
    q_dev, k_dev = q_dev.cpu(), k_dev.cpu()

    assert q_dev.shape == q_gold.shape == (seq_len, N, R)
    assert k_dev.shape == k_gold.shape == (seq_len, 1, R)

    q_pcc = _pcc(q_dev, q_gold)
    k_pcc = _pcc(k_dev, k_gold)
    assert q_pcc >= REQUIRED_PCC, f"q_pe rope PCC {q_pcc:.5f} < {REQUIRED_PCC}"
    assert k_pcc >= REQUIRED_PCC, f"k_pe rope PCC {k_pcc:.5f} < {REQUIRED_PCC}"
