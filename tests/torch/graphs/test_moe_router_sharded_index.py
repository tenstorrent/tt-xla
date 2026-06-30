# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Repro for the MoE-router sharded-index bug (tt-xla #5409).

The `*_buggy` graphs are the two ops from
`DeepseekV3MoEToA2AAdapter.route_tokens_to_experts`
(`python_package/tt_torch/sparse_mlp.py`) that produce wrong results for users on non-zero
mesh-rows when the token/batch axis is sharded:

Both lower to a 2-D index built from `[row, col]`, where `row` is an iota over the sharded
(token/batch) axis. When tt-mlir splits the program into per-device local shapes
(UpdateGlobalToLocalShapes), that iota is shrunk to its local size WITHOUT adding the per-shard
offset, so every shard emits `0..local-1` instead of `shard_id*local + 0..local-1`:

1. `scatter_(1, idx, 1)` — the local row iota writes only into local rows `0..local-1`, so only
   mesh-row 0's rows are correct; rows 1-3 come back zero.

2. `table.gather(1, idx)` — the table is all-gathered to all 64 rows, but the local row iota
   reads rows `0..local-1` of it, so shard `s` reads the wrong rows (the verified IR shows
   `stablehlo.iota dim=0 : tensor<16xui32>` feeding the gather over an all-gathered `64x256`
   table). The earlier "fp16 flat-index rounding" hypothesis is wrong — the index is an integer
   iota; the bug is the missing offset, same as scatter.

The `*_onehot` graphs are the workaround shipped in #5409: `scatter_` -> `one_hot + any` and
`gather` -> `one_hot + einsum`, so the `arange` is over the (unsharded) expert/group dim and the
index stays small + exact.

The selection indices are passed as explicit, deterministic inputs (NOT derived from a `topk`),
so the only variable between buggy/onehot is the scatter/gather lowering itself — `topk` on
random bf16 can pick different experts on TT vs CPU and would otherwise mask the real bug.

Inputs are sharded on the batch axis 4 ways (16 users/row, matching GLM 4.7's 64 users on 4
mesh-rows). `*_buggy` tests are xfail (wrong rows 1-3) until the iota/gather lowering is fixed in
tt-mlir; `*_onehot` tests must pass.
"""

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh

from tests.utils import incorrect_result, parametrize_arch

# GLM 4.7-like router shapes: 64 users sharded 4 ways across mesh-rows (16/row).
GLOBAL_BATCH = 64
NUM_EXPERTS = 256
NUM_GROUPS = 8
TOPK_GROUP = 4
TOP_K = 8
BATCH_ROWS = 4


class ScatterMaskBuggy(torch.nn.Module):
    """Buggy mask construction via `scatter_` (as in route_tokens_to_experts)."""

    def __init__(self, num_cols: int):
        super().__init__()
        self.num_cols = num_cols

    def forward(self, idx):  # idx: [B, T] int -> mask [B, num_cols]
        out = torch.zeros(
            idx.shape[0], self.num_cols, dtype=torch.bfloat16, device=idx.device
        )
        out.scatter_(1, idx, 1.0)
        return out


class ScatterMaskOneHot(torch.nn.Module):
    """Fixed mask via one_hot + any (the #5409 workaround)."""

    def __init__(self, num_cols: int):
        super().__init__()
        self.num_cols = num_cols

    def forward(self, idx):  # idx: [B, T] int -> mask [B, num_cols]
        return (
            (idx.unsqueeze(-1) == torch.arange(self.num_cols, device=idx.device))
            .any(dim=1)
            .to(torch.bfloat16)
        )


class GatherBuggy(torch.nn.Module):
    """Buggy weight selection via `gather` (as in route_tokens_to_experts)."""

    def forward(self, table, idx):  # table [B, E], idx [B, K] -> [B, K]
        return table.gather(1, idx)


class GatherOneHot(torch.nn.Module):
    """Fixed weight selection via one_hot + einsum (the #5409 workaround)."""

    def forward(self, table, idx):  # table [B, E], idx [B, K] -> [B, K]
        n_cols = table.shape[-1]
        ohw = (idx.unsqueeze(-1) == torch.arange(n_cols, device=idx.device)).to(
            table.dtype
        )
        return torch.einsum("be,bke->bk", table, ohw)


def _batch_sharded_mesh():
    """4-row mesh that shards the batch axis (mimics GLM cluster_axis=0, 16 users/row)."""
    num_devices = xr.global_runtime_device_count()
    assert num_devices % BATCH_ROWS == 0, f"need a multiple of {BATCH_ROWS} devices"
    mesh_shape = (BATCH_ROWS, num_devices // BATCH_ROWS)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


def _shard_inputs_on_batch(args, kwargs):
    """Data-parallel shard spec: shard every input's dim 0 on the batch axis.

    The (args, kwargs) signature selects the runner's data-parallel path that
    shards activations rather than model weights.
    """
    return {arg: ("batch", None) for arg in args}


def _group_indices():
    """Deterministic top-`TOPK_GROUP` group ids per row: same groups for every row."""
    row = torch.arange(TOPK_GROUP, dtype=torch.int64)  # groups 0..TOPK_GROUP-1
    return row.unsqueeze(0).expand(GLOBAL_BATCH, TOPK_GROUP).contiguous()


def _expert_indices():
    """Deterministic top-`TOP_K` expert ids per row, spread across the expert dim."""
    row = torch.arange(TOP_K, dtype=torch.int64) * (
        NUM_EXPERTS // TOP_K
    )  # 0,32,...,224
    return row.unsqueeze(0).expand(GLOBAL_BATCH, TOP_K).contiguous()


# ------------------------------------------------------------------------------
# scatter (group mask)
# ------------------------------------------------------------------------------


@parametrize_arch(["galaxy"])
@pytest.mark.xfail(
    reason=incorrect_result(
        "scatter row-iota over sharded batch axis misses per-shard offset; "
        "mesh-rows 1-3 routing zeroed (#5409)"
    )
)
def test_scatter_mask_buggy(arch):
    run_graph_test(
        ScatterMaskBuggy(NUM_GROUPS),
        [_group_indices()],
        framework=Framework.TORCH,
        mesh=_batch_sharded_mesh(),
        shard_spec_fn=_shard_inputs_on_batch,
    )


@parametrize_arch(["galaxy"])
def test_scatter_mask_onehot_fixed(arch):
    run_graph_test(
        ScatterMaskOneHot(NUM_GROUPS),
        [_group_indices()],
        framework=Framework.TORCH,
        mesh=_batch_sharded_mesh(),
        shard_spec_fn=_shard_inputs_on_batch,
    )


# ------------------------------------------------------------------------------
# gather (top-k weights)
# ------------------------------------------------------------------------------


@parametrize_arch(["galaxy"])
@pytest.mark.xfail(
    reason=incorrect_result(
        "gather's row iota over the sharded batch axis is localized without the "
        "per-shard offset; shards 1-3 read the wrong rows of the all-gathered table (#5409)"
    )
)
def test_gather_weight_buggy(arch):
    run_graph_test(
        GatherBuggy(),
        [
            torch.randn(GLOBAL_BATCH, NUM_EXPERTS, dtype=torch.bfloat16),
            _expert_indices(),
        ],
        framework=Framework.TORCH,
        mesh=_batch_sharded_mesh(),
        shard_spec_fn=_shard_inputs_on_batch,
    )


@parametrize_arch(["galaxy"])
def test_gather_weight_onehot_fixed(arch):
    run_graph_test(
        GatherOneHot(),
        [
            torch.randn(GLOBAL_BATCH, NUM_EXPERTS, dtype=torch.bfloat16),
            _expert_indices(),
        ],
        framework=Framework.TORCH,
        mesh=_batch_sharded_mesh(),
        shard_spec_fn=_shard_inputs_on_batch,
    )
