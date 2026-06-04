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
    # Raise torch_xla's parameter-wrapping threshold (note the upstream typo
    # "THREADSHOLD") so large graphs are NOT emitted as a thin
    # @main -> call @SyncTensorsGraph.N + tuple wrapper. That wrapper gets
    # inlined by the compiler, dropping the per-arg sdy.sharding +
    # ttcore.argument_type annotations, which causes the qwen3-32b bs32 QB2
    # sharding-collapse crash (#5032) and the trace-ON ttnn.empty OOM.
    # Default 1900 trips on the bs32 TP prefill (~4997 params); decode (~902)
    # stays under. Set in the parent so spawned workers inherit it; use
    # setdefault so it can be overridden from the shell to compare behavior.
    os.environ.setdefault("XLA_PARAMETER_WRAPPING_THREADSHOLD", "100000")
    return "vllm_tt.platform.TTPlatform"
