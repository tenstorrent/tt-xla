# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Validate the TT torch Hyper-Connections ops (``vllm_tt.layers.mhc``) against
the reference DeepSeek-V4 ``modified_model`` HC math.

vLLM's ``mhc`` is tilelang/CUDA-only and crashes on import on TT (so a DSV4 model
can't even be constructed); ``vllm_tt.layers.mhc`` reimplements ``mhc_pre`` /
``mhc_post`` / ``hc_head_fused_kernel`` in torch. These tests pin those torch ops
to the reference ``hc_split_sinkhorn`` + ``Block.hc_pre/hc_post`` /
``ParallelHead.hc_head`` semantics (the Sinkhorn iteration uses the *real*
reference kernel, not a reimplementation).
"""
import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("third_party.tt_forge_models.deepseek_v4.modified_model")
import vllm_tt.layers.mhc as tt_mhc  # noqa: E402

from third_party.tt_forge_models.deepseek_v4.modified_model.kernel import (  # noqa: E402
    hc_split_sinkhorn,
)

_HC, _D, _B, _S = 4, 16, 1, 4  # hc_mult, hidden, batch, seq
_MIX = (2 + _HC) * _HC
_NORM_EPS, _HC_EPS, _SINK_ITERS, _POST_MULT = 1e-6, 1e-6, 20, 2.0


# --- reference (verbatim from modified_model Block.hc_pre/hc_post + head) ---
def _ref_hc_pre(x, fn, scale, base):
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + _NORM_EPS)
    mixes = F.linear(xf, fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, _HC, _SINK_ITERS, _HC_EPS)
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
    return y.to(dtype), post, comb


def _ref_hc_post(x, residual, post, comb):
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    return y.type_as(x)


def _ref_hc_head(x, fn, scale, base):
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + _NORM_EPS)
    mixes = F.linear(xf, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + _HC_EPS
    y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
    return y.to(dtype)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    va, vb = a - a.mean(), b - b.mean()
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


@pytest.fixture
def hc_params():
    torch.manual_seed(0)
    x = (torch.randn(_B, _S, _HC, _D) * 0.5).to(torch.bfloat16)
    fn = torch.randn(_MIX, _HC * _D)
    scale = torch.randn(3)
    base = torch.randn(_MIX)
    return x, fn, scale, base


@pytest.mark.push
def test_mhc_pre_matches_reference(hc_params):
    x, fn, scale, base = hc_params
    y_ref, post_ref, comb_ref = _ref_hc_pre(x, fn, scale, base)
    post_mix, comb_mix, layer_input = tt_mhc.mhc_pre(
        x, fn, scale, base, _NORM_EPS, _HC_EPS, _HC_EPS, _POST_MULT, _SINK_ITERS
    )
    assert torch.allclose(comb_mix, comb_ref, atol=1e-5), "Sinkhorn comb differs"
    assert torch.allclose(post_mix, post_ref.unsqueeze(-1), atol=1e-5), "post differs"
    # layer_input is bf16; compare with a PCC bar.
    assert _pcc(layer_input, y_ref) > 0.999, "mixed layer input differs"


@pytest.mark.push
def test_mhc_post_matches_reference(hc_params):
    x, fn, scale, base = hc_params
    _, post_ref, comb_ref = _ref_hc_pre(x, fn, scale, base)
    sub_out = (torch.randn(_B, _S, _D) * 0.5).to(torch.bfloat16)  # sub-layer output
    ref = _ref_hc_post(sub_out, x, post_ref, comb_ref)
    got = tt_mhc.mhc_post(sub_out, x, post_ref.unsqueeze(-1), comb_ref)
    assert got.shape == ref.shape == (_B, _S, _HC, _D)
    assert _pcc(got, ref) > 0.999


@pytest.mark.push
def test_hc_head_matches_reference(hc_params):
    x, _, _, _ = hc_params
    fn = torch.randn(_HC, _HC * _D)  # head fn produces hc_mult mix values
    scale = torch.randn(1)
    base = torch.randn(_HC)
    ref = _ref_hc_head(x, fn, scale, base)  # [B, S, D]

    hs_flat = x.reshape(-1, _HC, _D)
    out = torch.empty(hs_flat.shape[0], _D, dtype=torch.bfloat16)
    tt_mhc._hc_head_fused_kernel(
        hs_flat, fn, scale, base, out, _D, _NORM_EPS, _HC_EPS, _HC
    )
    assert _pcc(out.view(_B, _S, _D), ref) > 0.999


@pytest.mark.push
def test_ops_registered_on_torch_ops_vllm():
    # The three ops must be registered (so vLLM's DecoderLayer torch.ops calls
    # resolve) after importing the shim.
    assert tt_mhc._already_registered("mhc_pre")
    assert tt_mhc._already_registered("mhc_post")
    assert tt_mhc._already_registered("hc_head_fused_kernel")
