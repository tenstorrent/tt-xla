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


def _install_cuda_noop_stubs():
    """Replace torch.cuda.Stream/Event/stream with no-ops on non-CUDA (TT)."""
    import contextlib

    import torch
    from vllm.platforms import current_platform

    if current_platform.is_cuda_alike():
        return

    class _NoOpStream:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def wait_stream(self, *a, **k):
            pass

        def synchronize(self, *a, **k):
            pass

        def record_event(self, *a, **k):
            return None

    class _NoOpEvent:
        def __init__(self, *a, **k):
            pass

        def record(self, *a, **k):
            pass

        def wait(self, *a, **k):
            pass

        def synchronize(self, *a, **k):
            pass

        def query(self, *a, **k):
            return True

        def elapsed_time(self, *a, **k):
            return 0.0

    torch.cuda.Stream = _NoOpStream
    torch.cuda.Event = _NoOpEvent
    torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()


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

    # Some GPU models (e.g. DeepSeek-V4) unconditionally allocate
    # torch.cuda.Stream/Event for compute-overlap during construction, which
    # crashes on TT (no CUDA). Replace them with no-ops: the overlap is
    # irrelevant on the single XLA stream and the code paths that use them are
    # replaced by TT OOT forwards.
    _install_cuda_noop_stubs()

    # DeepSeek-V4 Hyper-Connections: vLLM's mhc ops are tilelang/CUDA-only and
    # crash on import on TT, which blocks DSV4 model construction entirely.
    # Install the torch reimplementation as vllm.model_executor.layers.mhc
    # before any DSV4 layer is built.
    from .layers import mhc as _tt_mhc

    _tt_mhc.install()

    # DeepSeek-V4 RoPE: the YaRN cos/sin cache is arange(original_max * factor)
    # = arange(1_048_576) (~268MB fp32), impractical to tilize onto the mesh.
    # Cap it to max_model_len while preserving exact frequencies.
    from .layers import rope_cache_cap as _tt_rope_cap

    _tt_rope_cap.install()

    # Registers all OOT backends
    from .attention_impls import attention_dsv4  # noqa: F401  (DSV4 OOT wrapper)
    from .attention_impls import attention_mla  # noqa: F401
    from .layers.fused_moe import TTFusedMoE  # noqa: F401
