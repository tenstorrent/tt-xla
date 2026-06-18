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


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # NOTE: the fp8->bf16 dequant hook is installed worker-side (in
    # TTModelRunner.load_model() and TTPlatform.check_and_update_config()),
    # NOT here — importing vllm's fp8 module during early platform discovery
    # breaks platform resolution (empty device_type).
    return "vllm_tt.platform.TTPlatform"
