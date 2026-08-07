# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_cache_shard_factor (no hardware required).

The factor tells the worker how much of vLLM's TP-unaware KV budget each chip
really needs, so it must equal the number of ways initialize_kv_cache actually
shards the cache. Under-reporting starves the block pool (the #5796 bug: DP+TP
returned 1 while heads were replicated on all 32 chips); over-reporting hands
out blocks the chip cannot hold and OOMs at runtime.
"""
import types

import pytest
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm_tt.vllm_distributed_utils import ParallelismMode, kv_cache_shard_factor


def _runner(mode, shape, mesh_style="shape"):
    """Runner stub exposing just what kv_cache_shard_factor reads.

    mesh_style covers both mesh surfaces the function accepts: xs.Mesh
    (.shape() dict) and the plain axis_names/mesh_shape pair.
    """
    axis_names = ("batch", "model")
    if mesh_style == "shape":
        mesh = types.SimpleNamespace(shape=lambda: dict(zip(axis_names, shape)))
    else:
        mesh = types.SimpleNamespace(axis_names=axis_names, mesh_shape=shape)
    return types.SimpleNamespace(
        parallel_mode=mode,
        enable_tensor_parallel=mode
        in (
            ParallelismMode.TENSOR_PARALLEL_ONLY_1D,
            ParallelismMode.TENSOR_PARALLEL_ONLY_2D,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ),
        mesh=mesh,
    )


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize("mesh_style", ["shape", "axis_names"])
@pytest.mark.parametrize(
    "mode,shape,expected",
    [
        # No TP: cache is replicated, budget is already right.
        (ParallelismMode.DISABLED, (1, 1), 1),
        (ParallelismMode.DATA_PARALLEL_ONLY, (2, 1), 1),
        # TP-only: heads shard tp_size ways.
        (ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 2), 2),
        (ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 4), 4),
        (ParallelismMode.TENSOR_PARALLEL_ONLY_2D, (2, 4), 4),
        # DP+TP: heads shard on "model" too — the #5796 fix. Blocks stay
        # replicated on "batch", so dp_size must NOT enter the factor.
        (ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4), 4),
        (ParallelismMode.DATA_TENSOR_PARALLEL, (8, 4), 4),
    ],
)
def test_shard_factor(mode, shape, expected, mesh_style):
    assert kv_cache_shard_factor(_runner(mode, shape, mesh_style)) == expected


@pytest.mark.push
@pytest.mark.cpu
@pytest.mark.parametrize("mla_spec_cls", [MLAAttentionSpec, SlidingWindowMLASpec])
def test_mla_is_replicated_not_head_sharded(mla_spec_cls):
    """MLA holds one replicated latent cache, so TP must not inflate the budget.

    num_kv_heads is 1 for MLA, which is not divisible by tp_size; returning
    tp_size here would hand out tp_size times more blocks than a chip holds.
    """
    runner = _runner(ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4))
    spec = object.__new__(mla_spec_cls)
    assert kv_cache_shard_factor(runner, {"layer.0": spec}) == 1
    # Without the specs the function cannot know, and still reports the axis.
    assert kv_cache_shard_factor(runner) == 4


@pytest.mark.push
@pytest.mark.cpu
def test_non_mla_spec_still_shards():
    runner = _runner(ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 4))
    spec = object.__new__(FullAttentionSpec)
    assert kv_cache_shard_factor(runner, {"layer.0": spec}) == 4


@pytest.mark.push
@pytest.mark.cpu
def test_dp_tp_matches_tp_only_for_same_model_axis():
    """Adding DP replicas must not change the per-chip head sharding."""
    tp_only = kv_cache_shard_factor(
        _runner(ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 4))
    )
    dp_tp = kv_cache_shard_factor(_runner(ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4)))
    assert dp_tp == tp_only == 4
