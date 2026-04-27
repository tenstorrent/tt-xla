# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stubs for the tilelang-backed kernels imported by inference/model.py from the
# DeepSeek-V4-Flash HF repo. The real kernels require CUDA + tilelang and are
# not usable on TT/XLA. We only need:
#
#   - act_quant / fp8_gemm / fp4_gemm: tests run with bf16 weights, so the
#     model's `linear()` never dispatches into these. Kept as raise-stubs.
#   - sparse_attn / hc_split_sinkhorn: required for any test that runs full
#     `Block.forward` on CPU (e.g. realistic_inputs.py prefix pass). Both have
#     pure-torch implementations below that match the tilelang kernel
#     semantics — slow on large shapes but fine for the small CPU passes the
#     unit tests use.

import math

import torch


def _unsupported(name: str):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not implemented in the tt-xla test stub. "
            "These tests run the model in bf16 and must not dispatch to "
            "quantized GEMM kernels."
        )

    _fn.__name__ = name
    return _fn


def act_quant(x, block_size, scale_fmt=None, scale_dtype=None, inplace=False):
    """Two distinct call shapes in the upstream model:

    1. `x, s = act_quant(x, block_size, scale_fmt, scale_dtype)` from `linear`
       to feed FP8 GEMM. Only reachable when the weight is FP8/FP4 — we run
       in bf16, so this path is never taken. Raise to surface accidental
       dispatch.
    2. `act_quant(kv, 64, ..., True)` inside `Attention.forward` as a QAT
       simulation that quantizes-then-dequantizes in place. The return value
       is discarded; the side effect is small numerical noise we can safely
       skip on CPU."""
    if inplace:
        return None
    raise NotImplementedError(
        "act_quant non-inplace path requires FP8/FP4 weights; the tt-xla "
        "test stub runs the model in bf16, so this should be unreachable."
    )


def fp4_act_quant(x, block_size, inplace=False):
    """QAT-simulation in-place quantize used by Attention/Compressor for FP4
    activations. Same rationale as `act_quant`: skip the simulation on CPU."""
    if inplace:
        return None
    raise NotImplementedError(
        "fp4_act_quant non-inplace path is unused in bf16; the tt-xla test "
        "stub does not implement it."
    )


fp8_gemm = _unsupported("fp8_gemm")
fp4_gemm = _unsupported("fp4_gemm")


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Pure-torch port of the tilelang `hc_split_sinkhorn_kernel`.

    Mirrors the kernel in inference/kernel.py exactly: splits `mixes` into
    three sections (pre / post / comb), applies a row-softmax + column
    normalize on `comb`, then runs `sinkhorn_iters - 1` alternating
    row/column normalization steps.

    Shapes:
      mixes:     [b, s, mix_hc] where mix_hc = (2 + hc_mult) * hc_mult
      hc_scale:  [3]
      hc_base:   [mix_hc]
      pre, post: [b, s, hc_mult]
      comb:      [b, s, hc_mult, hc_mult]
    """
    b, s, mix_hc = mixes.size()
    expected_mix_hc = (2 + hc_mult) * hc_mult
    assert (
        mix_hc == expected_mix_hc
    ), f"mixes last dim {mix_hc} != (2+hc_mult)*hc_mult = {expected_mix_hc}"

    pre_mix = mixes[..., :hc_mult]
    post_mix = mixes[..., hc_mult : 2 * hc_mult]
    comb_mix = mixes[..., 2 * hc_mult :].view(b, s, hc_mult, hc_mult)

    pre = torch.sigmoid(pre_mix * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_mix * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])

    base_2d = hc_base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = comb_mix * hc_scale[2] + base_2d  # [b, s, hc_mult, hc_mult]

    # Initial row softmax + eps, then column normalize with eps in denom.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    # Sinkhorn iterations
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure-torch port of `sparse_attn_kernel`.

    Per (batch, seq) position: gather kv at the topk indices, compute
    attention scores, apply softmax with `attn_sink` as an extra "sink"
    logit (whose value contribution is zero), and return the weighted sum
    over kv. Out-of-range indices (-1) are masked to -inf in the score
    space and zeroed in the gather space.

    Shapes:
      q:          [b, m, h, d]      bf16
      kv:         [b, n, d]         bf16
      attn_sink:  [h]               fp32
      topk_idxs:  [b, m, topk]      int32
      output:     [b, m, h, d]      bf16
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]

    # Mask invalid index positions (model uses -1 as sentinel).
    valid = topk_idxs >= 0  # [b, m, topk]
    safe_idx = topk_idxs.clamp_min(0).long()

    # Gather kv at safe indices.
    # kv:        [b, n, d]
    # safe_idx:  [b, m, topk]  -> expand to [b, m, topk, d] for gather along n
    safe_idx_exp = safe_idx.unsqueeze(-1).expand(b, m, topk, d)
    kv_exp = kv.unsqueeze(1).expand(b, m, kv.size(1), d)
    kv_topk = kv_exp.gather(2, safe_idx_exp)  # [b, m, topk, d]
    # Zero out invalid slots so they contribute nothing to the einsum.
    kv_topk = kv_topk * valid.unsqueeze(-1).to(kv_topk.dtype)

    # Scores = q @ kv_topk^T per (b, m), in fp32 for stability.
    scores = (
        torch.einsum("bmhd,bmkd->bmhk", q.float(), kv_topk.float()) * softmax_scale
    )  # [b, m, h, topk]
    # Mask invalid positions to -inf so softmax ignores them.
    scores = scores.masked_fill(
        ~valid.unsqueeze(2),  # [b, m, 1, topk]
        float("-inf"),
    )

    # Append `attn_sink` as an extra logit per head with implicit zero value.
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
    scores_with_sink = torch.cat([scores, sink], dim=-1)  # [b, m, h, topk+1]
    weights = torch.softmax(scores_with_sink, dim=-1)
    weights_kv = weights[..., :-1]  # drop the sink slot

    # Weighted sum of kv values.
    o = torch.einsum("bmhk,bmkd->bmhd", weights_kv, kv_topk.float())
    return o.to(q.dtype)
