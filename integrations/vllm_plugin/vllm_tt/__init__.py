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
    return "vllm_tt.platform.TTPlatform"


def register_moe_oot_layer():
    # OOT-registers TTFusedMoE (CustomOp.register_oot) so Gemma-4's FusedMoE
    # uses our dense / expert-parallel routing path under XLA SPMD. Mirrors the
    # MLA backend's register_*_oot_layer + vllm.general_plugins pattern.
    from .layers.fused_moe import TTFusedMoE  # noqa: F401
