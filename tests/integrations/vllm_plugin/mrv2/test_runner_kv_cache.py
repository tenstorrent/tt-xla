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

import inspect

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

# is_eagle_group was added to KVCacheGroupSpec after vllm 0.19; pass it only
# when the installed version accepts it so the suite runs on both.
_GROUP_HAS_EAGLE = "is_eagle_group" in inspect.signature(KVCacheGroupSpec).parameters


def make_group(layer_names, kv_cache_spec):
    kwargs = {"layer_names": list(layer_names), "kv_cache_spec": kv_cache_spec}
    if _GROUP_HAS_EAGLE:
        kwargs["is_eagle_group"] = False
    return KVCacheGroupSpec(**kwargs)


def make_runner(device="cpu", kv_cache_dtype=torch.bfloat16, block_size=BLOCK_SIZE):
    r = object.__new__(TTModelRunnerV2)
    r.device = torch.device(device)
    r.kv_cache_dtype = kv_cache_dtype
    r.enable_tensor_parallel = False
    r.block_size = block_size
    # Single full-attention group, as initialize_kv_cache would leave it.
    r._num_kv_cache_groups = 1
    r._group_block_sizes = [block_size]
    r._group_is_sliding = [False]
    r._group_window_blocks = [0]
    r.max_num_reqs = 4
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
    group = make_group(layers, spec)
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
        kv_cache_groups=[make_group(["l0"], spec)],
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
def test_allocate_uniform_type_heterogeneous_head_sizes():
    # Gemma-4-style: same-type attention layers with different head sizes are
    # merged into one group whose spec is a UniformTypeKVCacheSpecs. Allocation
    # must unwrap the per-layer spec (its own page_size/head_size), not use the
    # group wrapper's, or the tensor_size % page_size assert fails for a layer.
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    r = make_runner()
    spec_a = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=64,
        dtype=torch.bfloat16,
    )
    spec_b = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=128,
        dtype=torch.bfloat16,
    )
    group_spec = UniformTypeKVCacheSpecs(
        block_size=BLOCK_SIZE, kv_cache_specs={"l0": spec_a, "l1": spec_b}
    )
    nblocks = 4
    cfg = KVCacheConfig(
        num_blocks=nblocks,
        kv_cache_tensors=[
            KVCacheTensor(size=nblocks * spec_a.page_size_bytes, shared_by=["l0"]),
            KVCacheTensor(size=nblocks * spec_b.page_size_bytes, shared_by=["l1"]),
        ],
        kv_cache_groups=[make_group(["l0", "l1"], group_spec)],
    )
    caches = r._allocate_kv_caches(cfg)
    # Each layer allocated with its own head size (last dim), not the wrapper's.
    assert caches["l0"][0].shape[-1] == 64
    assert caches["l1"][0].shape[-1] == 128


@pytest.mark.push
@pytest.mark.cpu
def test_hybrid_allocates_a_ring_for_the_sliding_group():
    # A sliding group is a per-user ring (window_blocks per slot + a null block),
    # not a slice of the shared pool, so its cache must be sized from the ring
    # geometry and not from the pool tensor size.
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    full = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )
    sliding = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        sliding_window=BLOCK_SIZE * 2,
    )
    r = make_runner()
    window_blocks = 8  # stick-aligned round-up of cdiv(64, 32) + 1
    r._num_kv_cache_groups = 2
    r._group_block_sizes = [BLOCK_SIZE, BLOCK_SIZE]
    r._group_is_sliding = [False, True]
    r._group_window_blocks = [0, window_blocks]

    cfg = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[
            KVCacheTensor(size=full.page_size_bytes * 4, shared_by=["full0"]),
            KVCacheTensor(size=sliding.page_size_bytes * 4, shared_by=["swa0"]),
        ],
        kv_cache_groups=[
            make_group(["full0"], full),
            make_group(["swa0"], sliding),
        ],
    )
    caches = r._allocate_kv_caches(cfg)

    assert set(caches) == {"full0", "swa0"}
    # Full group: blocks come from its pool tensor size.
    assert caches["full0"][0].shape[0] == 4
    # Sliding group: one sub-ring per request slot, plus the shared null block.
    assert caches["swa0"][0].shape[0] == window_blocks * r.max_num_reqs + 1


@pytest.mark.push
@pytest.mark.cpu
def test_block_size_mismatch_rejected():
    r = make_runner(block_size=16)  # runner block_size != config's 32
    with pytest.raises(AssertionError):
        r._allocate_kv_caches(make_config(block_size=BLOCK_SIZE))


@pytest.mark.push
@pytest.mark.cpu
def test_cross_layer_kv_sharing_points_child_at_target():
    # Gemma-4: a child layer reuses an earlier layer's KV cache. Wiring must
    # alias the child to the target's tensors and add it to the target's group,
    # or decode indexes the child's empty placeholder (kv_cache[0]) and crashes.
    r = make_runner()
    r.shared_kv_cache_layers = {"l1": "l0"}
    cfg = make_config(layers=("l0",))  # only the target allocates a cache
    kv = r._allocate_kv_caches(cfg)
    assert set(kv) == {"l0"}

    r._maybe_setup_cross_layer_kv_sharing(kv, cfg)

    assert kv["l1"] is kv["l0"]  # child aliases the target's [k, v]
    assert "l1" in cfg.kv_cache_groups[0].layer_names


@pytest.mark.push
@pytest.mark.cpu
def test_cross_layer_kv_sharing_noop_without_shared_layers():
    r = make_runner()
    r.shared_kv_cache_layers = {}
    cfg = make_config(layers=("l0",))
    kv = r._allocate_kv_caches(cfg)
    r._maybe_setup_cross_layer_kv_sharing(kv, cfg)
    assert set(kv) == {"l0"}
    assert cfg.kv_cache_groups[0].layer_names == ["l0"]
