# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test that reshape ops involving dimension splits/merges on sharded tensors
are handled correctly by the sharding (propagation) related passes.

Example from a real model: a Linear with column-parallel sharding produces a
tensor sharded on its last dim, then a reshape splits that dim.
"""

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh

from tests.utils import failed_ttmlir_compilation, parametrize_arch

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------


class LinearFollowedByReshape(torch.nn.Module):
    """Linear followed by a dimension-splitting reshape."""

    def __init__(self, in_dim, out_dim, num_splits):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, dtype=torch.bfloat16)
        self.num_splits = num_splits
        self.split_dim = out_dim // num_splits

    def forward(self, x):
        out = self.linear(x)
        return out.view(x.shape[0], self.num_splits, self.split_dim)


class ReshapeSplit(torch.nn.Module):
    """Reshape that splits a dimension (e.g. input arrives pre-sharded from a prior graph)."""

    def __init__(self, num_splits, split_dim):
        super().__init__()
        self.num_splits = num_splits
        self.split_dim = split_dim

    def forward(self, x):
        return x.view(x.shape[0], self.num_splits, self.split_dim)


class ReshapeMerge(torch.nn.Module):
    """Reshape that merges two dimensions into one."""

    def forward(self, x):
        # Merge dim 1 and dim 2: (batch, d1, d2) -> (batch, d1*d2)
        return x.view(x.shape[0], -1)


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------


@pytest.mark.push
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "in_dim,out_dim,num_splits",
    [(5120, 30720, 6)],
)
@pytest.mark.xfail(reason=failed_ttmlir_compilation("Compilation failure"))
def test_reshape_after_sharded_linear(in_dim, out_dim, num_splits, arch):
    """Column-parallel Linear followed by a dimension-splitting reshape.

    The weight is sharded on its out_features dimension (column-parallel).
    The reshape then splits the sharded output into (num_splits, split_dim).
    """
    # Example: in_dim=5120, out_dim=30720, num_splits=6
    #
    # Global view:
    # model.linear.weight - 30720x5120
    # input - 1x5120
    # output - 1x30720 (input * weight^T)
    # reshape output - 1x6x5120 (30720 elements)
    #
    # Local (device) view for 1x4 mesh where weight is sharded on dim 0:
    #
    # model.linear.weight - 7680x5120
    # input - 1x5120
    # output - 1x7680
    # reshape output - 1x6x1280 (7680 elements)
    #
    model = LinearFollowedByReshape(in_dim, out_dim, num_splits)
    hidden_states = torch.randn(1, in_dim, dtype=torch.bfloat16)

    # Mesh config
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        return {
            model.linear.weight: ("model", None),
        }

    run_graph_test(
        model,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.push
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "total_dim,num_splits",
    [(30720, 6)],
)
@pytest.mark.xfail(reason=failed_ttmlir_compilation("Compilation failure"))
def test_split_presharded_input(total_dim, num_splits, arch):
    """Reshape with dimension split on a pre-sharded input tensor.

    Simulates receiving the output of a column-parallel Linear that is
    already sharded on its last dimension, then splitting that dimension.
    """
    # Example: total_dim=30720, num_splits=6
    #
    # Global view:
    # input  - 1x30720
    # output - 1x6x5120 (30720 elements)
    #
    # Local (device) view for 1x4 mesh where input is sharded on dim 1:
    #
    # input  - 1x7680
    # output - 1x6x1280 (7680 elements)
    #
    model = ReshapeSplit(num_splits, split_dim=total_dim // num_splits)
    x = torch.randn(1, total_dim, dtype=torch.bfloat16)

    # Mesh config
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        return {args[0]: (None, "model")}

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.push
@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "num_splits,split_dim",
    [(6, 5120)],
)
def test_merge_presharded_input(num_splits, split_dim, arch):
    """Reshape that merges two dimensions on a sharded tensor.

    The inverse of dimension splitting: a 3D tensor with a sharded last
    dimension is reshaped to merge the last two dims into one.
    """
    # Example: num_splits=6, split_dim=5120
    #
    # Global view:
    # input  - 1x6x5120
    # output - 1x30720 (6*5120 elements)
    #
    # Local (device) view for 1x4 mesh where input is sharded on dim 2:
    #
    # input  - 1x6x1280
    # output - 1x7680 (6*1280 elements)
    #
    model = ReshapeMerge()
    x = torch.randn(1, num_splits, split_dim, dtype=torch.bfloat16)

    # Mesh config
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        return {args[0]: (None, None, "model")}

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
