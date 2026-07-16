# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the MRv2 runner KV cache allocation logic.

``TTModelRunnerV2._allocate_kv_caches`` (see vllm_tt/model_runner_v2.py) is the
device-coupled core of ``initialize_kv_cache``, split out so its allocation
logic runs portably on cpu (``device="cpu"``). On-device allocation on real TT
hardware is exercised separately (not in the cpu-only suite).

They pin the allocation contract TT owns: separate [k, v] tensors per standard
layer with the right shape/dtype, the num_blocks = size // page_size math, and
the hybrid / empty-group / block-size guards.
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_tt.model_runner_v2 import TTModelRunnerV2

BLOCK_SIZE = 32
NUM_KV_HEADS = 8
HEAD_SIZE = 64


def make_runner(device="cpu", kv_cache_dtype=torch.bfloat16, block_size=BLOCK_SIZE):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device(device)
    r.kv_cache_dtype = kv_cache_dtype
    r.enable_tensor_parallel = False
    r.block_size = block_size
    return r


def make_config(num_blocks=16, layers=("layer.0",), block_size=BLOCK_SIZE):
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    tensors = [
        KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
        for name in layers
    ]
    group = KVCacheGroupSpec(
        layer_names=list(layers), kv_cache_spec=spec, is_eagle_group=False
    )
    return KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=[group]
    )


@pytest.mark.push
@pytest.mark.cpu
def test_allocate_standard_kv_caches_shapes_and_dtype():
    r = make_runner()
    num_blocks = 16
    kv = r._allocate_kv_caches(make_config(num_blocks=num_blocks, layers=("l0", "l1")))

    assert set(kv) == {"l0", "l1"}
    for name in ("l0", "l1"):
        k, v = kv[name]  # separate K/V tensors
        assert tuple(k.shape) == (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        assert tuple(v.shape) == (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
        assert k.dtype == torch.bfloat16
        assert k.device.type == "cpu"


@pytest.mark.push
@pytest.mark.cpu
def test_allocate_num_blocks_from_tensor_size():
    r = make_runner()
    # Two pages' worth of bytes -> two blocks.
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    cfg = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=2 * spec.page_size_bytes, shared_by=["l0"])
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["l0"], kv_cache_spec=spec, is_eagle_group=False
            )
        ],
    )
    k, _ = r._allocate_kv_caches(cfg)["l0"]
    assert k.shape[0] == 2


@pytest.mark.push
@pytest.mark.cpu
def test_allocate_kv_cache_dtype_override():
    # spec.dtype may be a 1-byte accounting dtype; buffers use kv_cache_dtype.
    r = make_runner(kv_cache_dtype=torch.float32)
    k, _ = r._allocate_kv_caches(make_config())["layer.0"]
    assert k.dtype == torch.float32


@pytest.mark.push
@pytest.mark.cpu
def test_empty_group_config_skips():
    r = make_runner()
    cfg = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    assert r._allocate_kv_caches(cfg) == {}


@pytest.mark.push
@pytest.mark.cpu
def test_hybrid_config_rejected():
    r = make_runner()
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    group = KVCacheGroupSpec(
        layer_names=["l0"], kv_cache_spec=spec, is_eagle_group=False
    )
    cfg = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes, shared_by=["l0"])],
        kv_cache_groups=[group, group],
    )
    with pytest.raises(NotImplementedError):
        r._allocate_kv_caches(cfg)


@pytest.mark.push
@pytest.mark.cpu
def test_block_size_mismatch_rejected():
    r = make_runner(block_size=16)  # runner block_size != config's 32
    with pytest.raises(AssertionError):
        r._allocate_kv_caches(make_config(block_size=BLOCK_SIZE))
