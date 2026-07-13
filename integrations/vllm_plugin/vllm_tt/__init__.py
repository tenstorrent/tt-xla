# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Register TT attention backends at module import time
register_backend(
    backend=AttentionBackendEnum.CUSTOM,
    class_path="vllm_tt.attention_impls.attention.TTAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASH_ATTN_MLA,
    class_path="vllm_tt.attention_impls.attention_mla.TTMLAAttentionBackend",
)
# DeepSeek-V4 sliding-window (SWA-only) attention. DSV4 is a sparse-MLA model,
# so we override vLLM's sparse-MLA enum slot (FLASHMLA_SPARSE) — mirroring how
# the dense MLA backend overrides FLASH_ATTN_MLA above. This is a lazy override
# (register_backend only stores the class path; the module is imported on first
# get_class()), so it is dormant until a DSV4 model routes to it. NOTE: routing
# DSV4 vs DeepSeek-V3.2 (both sparse-MLA) through get_attn_backend_cls needs the
# model-architecture gating described in DSV4_TT_Next_Steps.md §3.1.
register_backend(
    backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    class_path="vllm_tt.attention_impls.attention_dsv4.TTDeepseekV4AttentionBackend",
)


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_tt.platform.TTPlatform"


def register_oot_layers():
    # Patch vLLM Attention symbol from the general-plugins path.
    import vllm.model_executor.layers.attention as _attn_pkg
    import vllm.model_executor.layers.attention.attention as _attn_mod
    import vllm.model_executor.layers.attention.encoder_only_attention as _enc_attn_mod
    from vllm_tt.attention_impls.attention import TTAttention, TTEncoderOnlyAttention

    _attn_mod.Attention = TTAttention
    _attn_pkg.Attention = TTAttention
    _enc_attn_mod.Attention = TTAttention

    _enc_attn_mod.EncoderOnlyAttention = TTEncoderOnlyAttention
    _attn_pkg.EncoderOnlyAttention = TTEncoderOnlyAttention

    # Registers all OOT backends
    from .attention_impls import attention_mla  # noqa: F401
    from .layers.fused_moe import TTFusedMoE  # noqa: F401
