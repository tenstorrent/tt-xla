# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
"""

from __future__ import annotations

import copy
import gc
import logging
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.sparse_mlp import enable_sparse_mlp
from ttxla_tools.logging import logger

from tests.benchmark.utils import compute_pcc
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)

# Want to initialize a KV cache for a model but the tensor is too large to fit on DRAM of a single chip. 
# To work around sharding rules, we can create a tensor like kv_cache = torch.zeros(1,1,1,1), move it to 
# device, shard it, call sync to force it to shard and the repeat it to the desired shape on each local 
# device.


def _mock_kv_update(cache: torch.Tensor, new_k: torch.Tensor, page: torch.Tensor):
    """Minimal device-resident KV write: add, store into one page, return a
    scalar witness.

    Kept deliberately cheap in DRAM. `new_k` is a single page (1 MiB at these
    dims), the update is in-place so no second full-size cache is materialized,
    and only the witness travels back to host -- returning `cache` itself would
    gather 1 GiB.
    """
    updated = new_k + 1.0
    cache.index_copy_(0, page, updated)
    # Forces the write to be part of the executed graph; page 0 only, so the
    # readback is 1 MiB of reduction, not the whole cache.
    return cache[0].sum(dtype=torch.float32).reshape(1)


def _run_mock(k_cache, mesh, device, num_kv_heads, block_size, head_size):
    """Ship one page, run the mock update, and block until the device is idle."""
    
    new_k = torch.ones(
        1, num_kv_heads, block_size, head_size, dtype=torch.bfloat16
    ).to(device)
    # Match the cache's head-axis split so the write needs no resharding.
    xs.mark_sharding(new_k, mesh, (None, "_axis_1", None, None))
    page = torch.zeros(1, dtype=torch.int64).to(device)

    _compiled_mock = torch.compile(
        _mock_kv_update, backend="tt"
    )

    print(f"{time.time()}: Running mock KV update (page 0)")
    witness = _compiled_mock(k_cache, new_k, page)
    # wait_device_ops so the DRAM peak is still live while the logger samples.
    torch_xla.sync(wait=True)
    
    print(f"{time.time()}: Mock update done")
    return True


def test_kv_cache_init_workaround() -> None:
    # vLLM K-cache geometry: (num_blocks, num_kv_heads, block_size, head_size)
    NUM_BLOCKS  = 1024        # vLLM: kv_cache_tensor.size // spec.page_size_bytes
    BLOCK_SIZE  = 32          # tokens per page
    NUM_KV_HEADS = 128        # must be divisible by the TP axis size
    HEAD_SIZE   = 128

    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    # Keep const-eval off: a zeros/repeat/add chain is fully const-foldable, and
    # on the default settings it would be evaluated on the host, so nothing
    # would ever land in device DRAM.
    torch_xla.set_custom_compile_options(
        {
            "enable_const_eval_inputs_to_system_memory": False,
            "enable_const_eval_on_cpu": False,
        }
    )

    print(f"{time.time()}: Sleep 20 seconds attach PID to memory logger tool now")
    #time.sleep(20) # give 20 seconds to attach PID to memory logger tool

    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()


    # 1) tiny seed with the TP-axis extent on the head dim -> 1 elem/device after sharding
    k_seed = torch.zeros(1, 8, 32, 32, dtype=torch.bfloat16).to(device)
    xs.mark_sharding(k_seed, mesh, (None, "_axis_1", None, None))
    torch_xla.sync(wait=True)           # force the shard before it grows

    target = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    # 2) grow locally; repeat factors are per-shard, so dim1 factor is heads/tp
    k_cache = k_seed.repeat(NUM_BLOCKS, NUM_KV_HEADS // 8, BLOCK_SIZE // 32, HEAD_SIZE // 32)
    # global shape (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE),
    # local shape  (NUM_BLOCKS, NUM_KV_HEADS // 8, BLOCK_SIZE, HEAD_SIZE)
    xs.mark_sharding(k_cache, mesh, (None, "_axis_1", None, None))  # re-pin after repeat

    # The repeat above is only lazy IR until something executes; this is what
    # actually lands the cache in device DRAM.
    _run_mock(k_cache, mesh, device, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    print(f"{time.time()}: Sleep 10 seconds so memory logger jumps are clear")
    time.sleep(10)    

    print(f"{time.time()}: Final tensor shape: {k_cache.shape}")
    assert tuple(k_cache.shape) == target



def test_kv_cache_init_without_workaround() -> None:
    # vLLM K-cache geometry: (num_blocks, num_kv_heads, block_size, head_size)
    NUM_BLOCKS  = 1024        # vLLM: kv_cache_tensor.size // spec.page_size_bytes
    BLOCK_SIZE  = 32          # tokens per page
    NUM_KV_HEADS = 128        # must be divisible by the TP axis size
    HEAD_SIZE   = 128

    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    print(f"{time.time()}: Sleep 10 seconds attach PID to memory logger tool now")
    time.sleep(10) # give 10 seconds to attach PID to memory logger tool

    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()


    target = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    # No seed/repeat trick: the whole cache is built host-side (1 GiB at these
    # dims) and handed to the device in one piece.
    k_cache = torch.zeros(target, dtype=torch.bfloat16).to(device)

    xs.mark_sharding(k_cache, mesh, (None, "_axis_1", None, None))


    _run_mock(k_cache, mesh, device, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    print(f"{time.time()}: Sleep 10 seconds so memory logger jumps are clear")
    time.sleep(10)

    print(f"{time.time()}: Final tensor shape: {k_cache.shape}")
    assert tuple(k_cache.shape) == target