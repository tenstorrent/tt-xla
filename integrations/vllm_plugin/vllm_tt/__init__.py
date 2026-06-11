# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Register TT attention backends at module import time
register_backend(
    backend=AttentionBackendEnum.CUSTOM,
    class_path="vllm_tt.attention.TTAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASH_ATTN_MLA,
    class_path="vllm_tt.attention_mla.TTMLAAttentionBackend",
)
# DeepSeek Sparse Attention (sparse MLA, e.g. DeepSeek-V3.2). Registered under the
# sparse-MLA enum slot; TTPlatform.get_attn_backend_cls routes use_mla+use_sparse
# here.
register_backend(
    backend=AttentionBackendEnum.FLASHMLA_SPARSE,
    class_path="vllm_tt.attention_dsa.TTDSAAttentionBackend",
)


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_tt.platform.TTPlatform"


def register_mla_oot_layer():
    from . import attention_mla  # noqa: F401
