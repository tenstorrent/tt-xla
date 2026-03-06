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

from tests.utils import parametrize_arch


class LinearReshape(torch.nn.Module):
    """Linear followed by a dimension-splitting reshape."""

    def __init__(self, in_dim, out_dim, num_splits):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, dtype=torch.bfloat16)
        self.num_splits = num_splits
        self.split_dim = out_dim // num_splits

    def forward(self, x):
        out = self.linear(x)
        return out.view(x.shape[0], self.num_splits, self.split_dim)


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "in_dim,out_dim,num_splits",
    [
        (5120, 30720, 6),
        (256, 1024, 4),
    ],
    ids=["regular", "small"],
)
def test_reshape_after_sharded_linear(in_dim, out_dim, num_splits, arch):
    """Column-parallel Linear + dimension-splitting reshape."""
    xr.set_device_type("TT")

    model = LinearReshape(in_dim, out_dim, num_splits)
    hidden_states = torch.randn(1, in_dim, dtype=torch.bfloat16)  # 1xin_dim

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        return {
            model.linear.weight: ("batch", None),
            model.linear.bias: ("batch",),
        }

    run_graph_test(
        model,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


class ReshapeOnly(torch.nn.Module):
    """Just a reshape — input arrives pre-sharded from a prior graph."""

    def __init__(self, num_splits, split_dim):
        super().__init__()
        self.num_splits = num_splits
        self.split_dim = split_dim

    def forward(self, x):
        return x.view(x.shape[0], self.num_splits, self.split_dim)


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "total_dim,num_splits",
    [
        (30720, 6),
        (1024, 4),
    ],
    ids=["dit_30720_6", "small_1024_4"],
)
def test_reshape_split_presharded_input(total_dim, num_splits, arch):
    """Reshape with dimension split on a pre-sharded input tensor."""
    xr.set_device_type("TT")

    split_dim = total_dim // num_splits
    model = ReshapeOnly(num_splits, split_dim)
    x = torch.randn(1, total_dim, dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        # Shard the input activation on dim=1
        return {args[0]: (None, "batch")}

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
