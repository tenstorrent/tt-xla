# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Raise torch-xla's parameter-tupling threshold (typo name is upstream).
# Large graphs such as Gemma-4 b32 emit >3200 parameters, which triggers a
# ManualComputationOp lowering path the PJRT plugin cannot currently handle.
# Keep parameters individual by bumping the threshold; setdefault lets callers
# override via the environment when they need the original behaviour.
os.environ.setdefault("XLA_PARAMETER_WRAPPING_THREADSHOLD", "10000")

register_backend(
    backend=AttentionBackendEnum.CUSTOM,
    class_path="vllm_tt.attention.TTAttentionBackend",
)


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_tt.platform.TTPlatform"
