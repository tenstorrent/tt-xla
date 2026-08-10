# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_budget_scale_factor (no hardware required).

The factor tells the worker how much of vLLM's TP-unaware KV budget each chip
really needs, so it must equal the number of ways the cache is actually split.
Under-reporting starves the block pool; over-reporting hands out blocks the
chip cannot hold and OOMs at allocation.

With a per-replica block pool the cache is split twice: heads on "model" and
blocks on "batch". MLA is the exception — its single latent cache is
replicated, not head-sharded, so only the block split counts.
"""
import types

import pytest
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm_tt.vllm_distributed_utils import ParallelismMode, kv_budget_scale_factor

pytestmark = [pytest.mark.push, pytest.mark.cpu]


def _runner(mode, shape, shard_kv_blocks):
    """Runner stub exposing just what kv_budget_scale_factor reads."""
    axis_names = ("batch", "model")
    dp_size, _ = shape
    return types.SimpleNamespace(
        parallel_mode=mode,
        enable_tensor_parallel=mode
        in (
            ParallelismMode.TENSOR_PARALLEL_ONLY_1D,
            ParallelismMode.TENSOR_PARALLEL_ONLY_2D,
            ParallelismMode.DATA_TENSOR_PARALLEL,
        ),
        mesh=types.SimpleNamespace(shape=lambda: dict(zip(axis_names, shape))),
        dp_size=dp_size,
        shard_kv_blocks=shard_kv_blocks,
    )


def _spec(cls):
    return {"layer.0": object.__new__(cls)}


@pytest.mark.parametrize(
    "mode,shape,shard_blocks,expected",
    [
        # No TP: cache replicated, budget already right.
        (ParallelismMode.DISABLED, (1, 1), False, 1),
        (ParallelismMode.DATA_PARALLEL_ONLY, (2, 1), False, 1),
        # TP-only: heads shard tp_size ways, blocks replicated.
        (ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 4), False, 4),
        # DP+TP with a per-replica pool: heads AND blocks split.
        (ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4), True, 8),
        (ParallelismMode.DATA_TENSOR_PARALLEL, (4, 2), True, 8),
    ],
)
def test_scale_factor(mode, shape, shard_blocks, expected):
    assert kv_budget_scale_factor(_runner(mode, shape, shard_blocks)) == expected


@pytest.mark.parametrize("mla_cls", [MLAAttentionSpec, SlidingWindowMLASpec])
def test_mla_counts_blocks_only(mla_cls):
    """MLA replicates the latent cache, so the "model" factor must drop out."""
    runner = _runner(ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4), True)
    # dp_size alone, NOT tp_size * dp_size — the cache is not head-sharded.
    assert kv_budget_scale_factor(runner, _spec(mla_cls)) == 2
    # Without the specs the function cannot know, and still counts both.
    assert kv_budget_scale_factor(runner) == 8


@pytest.mark.parametrize("mla_cls", [MLAAttentionSpec, SlidingWindowMLASpec])
def test_mla_without_block_sharding_is_unscaled(mla_cls):
    """TP-only MLA is replicated on every chip, so the budget stands as-is."""
    runner = _runner(ParallelismMode.TENSOR_PARALLEL_ONLY_1D, (1, 4), False)
    assert kv_budget_scale_factor(runner, _spec(mla_cls)) == 1
    assert kv_budget_scale_factor(runner) == 4


def test_non_mla_spec_still_counts_both():
    runner = _runner(ParallelismMode.DATA_TENSOR_PARALLEL, (2, 4), True)
    assert kv_budget_scale_factor(runner, _spec(FullAttentionSpec)) == 8
