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

    # DRAFT (pending review): patch the FusedMoE factory for TT MoE.
    from .layers.fused_moe import install_tt_fused_moe

    install_tt_fused_moe()

    # Gemma4's multimodal encoder queries torch.accelerator.get_memory_info(),
    # which asserts on the TT device; route it to a host-memory estimate.
    from .platform import install_tt_accelerator_memory_info

    install_tt_accelerator_memory_info()
