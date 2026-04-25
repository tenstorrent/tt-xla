# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Stubs for the tilelang-backed kernels imported by inference/model.py from the
# DeepSeek-V4-Flash HF repo. The real kernels require CUDA + tilelang and are
# not usable on TT/XLA. These stubs only need to satisfy the top-level import;
# for MoE/MLP tests we run the model with bf16 weights, so `linear()` never
# dispatches into act_quant / fp8_gemm / fp4_gemm. sparse_attn and
# hc_split_sinkhorn are only used by Attention and Block, neither of which is
# exercised by the MoE/MLP unit tests.


def _unsupported(name: str):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not implemented in the tt-xla test stub. "
            "These tests run the model in bf16 and must not dispatch to "
            "quantized GEMM or the tilelang sparse-attn / Sinkhorn kernels."
        )

    _fn.__name__ = name
    return _fn


act_quant = _unsupported("act_quant")
fp4_act_quant = _unsupported("fp4_act_quant")
fp8_gemm = _unsupported("fp8_gemm")
fp4_gemm = _unsupported("fp4_gemm")
sparse_attn = _unsupported("sparse_attn")
hc_split_sinkhorn = _unsupported("hc_split_sinkhorn")
