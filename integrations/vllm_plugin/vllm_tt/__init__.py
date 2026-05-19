# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

# Register TT attention backends at module import time. These are
# class-path *strings* — no actual import of attention.py or
# attention_mla.py happens here. Workers and the main process both run
# this module-level code, so the resolved backend path is visible
# everywhere. The classes themselves load lazily when vLLM does
# `get_class()` / `resolve_obj_by_qualname()`, by which point
# vllm.config is fully initialized.
register_backend(
    backend=AttentionBackendEnum.CUSTOM,
    class_path="vllm_tt.attention.TTAttentionBackend",
)
register_backend(
    backend=AttentionBackendEnum.FLASH_ATTN_MLA,
    class_path="vllm_tt.attention_mla.TTMLAAttentionBackend",
)


def register():
    # Setting worker multiprocessing method to spawn to avoid hangs in consecutive vllm pytest runs
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # NOTE: attention_mla is NOT imported here. vLLM resolves
    # current_platform lazily on first attribute access, which can fire
    # while vllm.config itself is still being imported (its model.py
    # imports current_platform at module top-level). attention_mla's
    # import chain reaches back into vllm.config and would hit a
    # partially-initialized-module ImportError. The MLA registration
    # (FLASH_ATTN_MLA backend + OOT TTMultiHeadLatentAttentionWrapper)
    # is triggered from TTPlatform.check_and_update_config, which vLLM
    # calls only after the config modules are fully loaded.
    return "vllm_tt.platform.TTPlatform"
