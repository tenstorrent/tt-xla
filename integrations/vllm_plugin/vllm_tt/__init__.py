# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# --- TT-XLA branch probe (TEMPORARY — remove before merge) ----------------------------
# Prints once per process at plugin import — the earliest possible point, BEFORE model
# load / warmup / trace-capture — so it is hit even if the model later DRAM-OOMs during
# warmup. Used to confirm a specific tt-xla branch/wheel is the one actually loaded in a
# tt-shield CI run. Grep CI logs for the marker string. Edit `branch-tag` per branch.
print(
    "[TT-XLA-BRANCH-PROBE] vllm_tt plugin imported "
    f"(pid={os.getpid()}) from {__file__} "
    "| branch-tag=kmabee/chunked_prefill_isue_4986_explore.rebase.followups",
    file=sys.stderr,
    flush=True,
)
# -------------------------------------------------------------------------------------

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
