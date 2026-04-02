# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Register TT attention backend at module import time
register_backend(
    backend=AttentionBackendEnum.CUSTOM,
    class_path="vllm_tt.attention.TTAttentionBackend",
)

# Register TT MLA attention backend for MLA-based models (e.g. DeepSeek V3/R1).
# Uses ttnn.transformer.flash_mla_prefill and
# ttnn.transformer.paged_flash_multi_latent_attention_decode under the hood.
try:
    register_backend(
        backend=AttentionBackendEnum.CUSTOM_MLA,
        class_path="vllm_tt.attention.TTMLAAttentionBackend",
    )
except AttributeError:
    # CUSTOM_MLA enum value not present in this vLLM version; TTMLAAttentionBackend
    # is still importable and usable directly by MLA model implementations.
    pass


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_tt.platform.TTPlatform"
