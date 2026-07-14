# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-agnostic (torch) Hyper-Connections for DeepSeek-V4 on TT.

vLLM's ``vllm.model_executor.layers.mhc`` implements the DSV4 Hyper-Connections
ops (``mhc_pre`` / ``mhc_post`` / ``hc_head_fused_kernel``) as **tilelang** JIT
kernels. tilelang is CUDA-only, and the module has *unguarded* module-level
``@tilelang.jit`` decorators, so merely ``import``-ing it crashes on TT
(``'NoneType' object has no attribute 'jit'``) — which means a DSV4 model cannot
even be *constructed* on TT (``DeepseekV4DecoderLayer.__init__`` imports it).

This module reimplements the three ops in plain torch (mirroring the reference
``modified_model`` Hyper-Connections: ``hc_split_sinkhorn`` + ``Block.hc_pre`` /
``hc_post`` + ``ParallelHead.hc_head``) and registers them on
``torch.ops.vllm.*`` under the TT platform dispatch key (XLA), so the ops trace
onto the device. ``install()`` aliases this module into
``sys.modules["vllm.model_executor.layers.mhc"]`` so vLLM's lazy
``import ...mhc`` returns this (already-registered) module instead of the
crashing tilelang one.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from vllm.utils.torch_utils import direct_register_custom_op

from ..logger import tt_init_logger

logger = tt_init_logger(__name__)

_VLLM_MHC_NAME = "vllm.model_executor.layers.mhc"


# --------------------------------------------------------------------------- #
# Torch implementations (mirror modified_model Hyper-Connections)
# --------------------------------------------------------------------------- #
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """residual [*, hc, H] -> (post_mix [*, hc, 1], comb_mix [*, hc, hc],
    layer_input [*, H] bf16). Mixes the hc residual streams into the single
    sub-layer input (``pre``), and returns the ``post`` / Sinkhorn-normalized
    ``comb`` mixers consumed by ``mhc_post``."""
    outer = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]

    xf = residual.reshape(*outer, hc * hidden).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(xf, fn.float()) * rsqrt  # [*, (2 + hc) * hc]

    pre_raw = mixes[..., :hc]
    post_raw = mixes[..., hc : 2 * hc]
    comb_raw = mixes[..., 2 * hc :]

    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(
        post_raw * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (comb_raw * hc_scale[2] + hc_base[2 * hc :]).reshape(*outer, hc, hc)

    # Sinkhorn: row softmax + eps, then column-normalize; iterate.
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre.unsqueeze(-1) * xf.reshape(*outer, hc, hidden), dim=-2
    )  # [*, H]
    return post.unsqueeze(-1), comb, layer_input.to(torch.bfloat16)


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand the sub-layer output ``x`` [*, H] back to hc streams and mix in
    the (Sinkhorn-combined) residual streams: out [*, hc, H]."""
    term1 = post_layer_mix.float() * x.unsqueeze(-2).float()  # [*, hc, H]
    term2 = (comb_res_mix.unsqueeze(-1).float() * residual.unsqueeze(-2).float()).sum(
        dim=-3
    )  # [*, hc, H]
    return (term1 + term2).to(x.dtype)


def _hc_head_fused_kernel(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    out: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int,
) -> None:
    """Reduce the hc streams to one at the head: fills ``out`` [T, H] in place.
    hs_flat is [T, hc_mult, hidden_size]."""
    tokens = hs_flat.shape[0]
    xf = hs_flat.reshape(tokens, hc_mult * hidden_size).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(xf, fn.float()) * rsqrt  # [T, hc_mult]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps  # [T, hc_mult]
    y = torch.sum(pre.unsqueeze(-1) * hs_flat.float(), dim=1)  # [T, H]
    out.copy_(y.to(out.dtype))


# --------------------------------------------------------------------------- #
# Fakes (shape inference)
# --------------------------------------------------------------------------- #
def _mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    post_mix = torch.empty(
        *outer, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        *outer, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        *outer, hidden, dtype=torch.bfloat16, device=residual.device
    )
    return post_mix, comb_mix, layer_input


def _mhc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


# --------------------------------------------------------------------------- #
# Registration + module install
# --------------------------------------------------------------------------- #
def _already_registered(op_name: str) -> bool:
    try:
        getattr(torch.ops.vllm, op_name)
        return True
    except Exception:
        return False


def _register() -> None:
    if not _already_registered("mhc_pre"):
        direct_register_custom_op(
            op_name="mhc_pre", op_func=mhc_pre, mutates_args=[], fake_impl=_mhc_pre_fake
        )
    if not _already_registered("mhc_post"):
        direct_register_custom_op(
            op_name="mhc_post",
            op_func=mhc_post,
            mutates_args=[],
            fake_impl=_mhc_post_fake,
        )
    if not _already_registered("hc_head_fused_kernel"):
        direct_register_custom_op(
            op_name="hc_head_fused_kernel",
            op_func=_hc_head_fused_kernel,
            mutates_args=["out"],
        )


_register()


def install() -> None:
    """Alias this module as ``vllm.model_executor.layers.mhc`` so vLLM's lazy
    import returns this (torch) implementation instead of the tilelang one that
    crashes on TT. No-op if the real module was already imported."""
    existing = sys.modules.get(_VLLM_MHC_NAME)
    if existing is sys.modules[__name__]:
        return
    if existing is not None:
        logger.warning(
            "vllm.model_executor.layers.mhc already imported (%s); TT torch "
            "Hyper-Connections may not take effect.",
            existing,
        )
    sys.modules[_VLLM_MHC_NAME] = sys.modules[__name__]
    logger.info("[TT] Installed torch Hyper-Connections (mhc_pre/mhc_post/hc_head).")
