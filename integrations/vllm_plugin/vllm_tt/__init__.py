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
    # partially-initialized-module ImportError. The OOT MLA layer is
    # registered separately from `register_mla_oot_layer` below, which
    # vLLM invokes via the `vllm.general_plugins` entry point.
    return "vllm_tt.platform.TTPlatform"


def register_mla_oot_layer():
    """`vllm.general_plugins` callback — install the TT OOT MLA layer.

    Importing ``attention_mla`` runs its ``@register_oot`` decorator, which
    installs ``TTMultiHeadLatentAttentionWrapper`` into vLLM's
    ``op_registry_oot``. Unlike the FLASH_ATTN_MLA *backend* (registered
    lazily above by class-path string), the OOT registry holds live class
    objects, so the module must actually be imported for the layer to
    register.

    vLLM's ``load_general_plugins()`` invokes this once per process (main,
    engine-core, and worker) at a point where ``vllm.config`` is fully
    initialized and before any model is built — so importing ``attention_mla``
    here is safe even though its import chain re-enters ``vllm.config`` (the
    partial-init hazard that keeps it out of ``register()``).
    """
    from . import attention_mla  # noqa: F401
