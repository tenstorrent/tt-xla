# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators.evaluation_config import AtolConfig, ComparisonConfig, PccConfig
from infra.evaluators import TorchComparisonEvaluator
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh


# these examples should be moved to the test infra once this issue is closed: https://github.com/tenstorrent/tt-xla/issues/1822
def create_device_mesh(mesh_shape, mesh_names):
    assert len(mesh_shape) == len(
        mesh_names
    ), "Mesh shape and names must match in length"
    num_devices = xr.global_runtime_device_count()
    assert (
        np.prod(mesh_shape) == num_devices
    ), "Mesh shape must match the number of devices"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    return mesh


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_all_reduce(shard_dim):
    """Test all_reduce operation with sharding on different dimensions.
    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    # XLA won't insert all-reduce (only for xm.all_reduce) op unless this is set.
    # when shardy is generating all-reduce ops, this is not needed.
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    xr.set_device_type("TT")
    enable_spmd()
    # Create tensor with values that make reduction easy to verify
    t = torch.ones(256, 512)
    t = t.to(torch_xla.device())

    if shard_dim == 0:
        # Shard on batch dimension (dim 0)
        mesh = create_device_mesh((2, 1), ("batch", "model"))
        xs.mark_sharding(t, mesh, ("batch", None))
        groups = [[0, 1]]
    else:
        # Shard on model dimension (dim 1)
        mesh = create_device_mesh((1, 2), ("batch", "model"))
        xs.mark_sharding(t, mesh, (None, "model"))
        groups = [[0, 1]]

    # Perform all_reduce sum operation
    y = xm.all_reduce(xm.REDUCE_SUM, t, groups=groups)

    torch_xla.sync()
    y = y.to("cpu")
    print(f"All-reduce shard dim: {shard_dim}, Y Shape: {y.shape}")

    expected = torch.ones(256, 512) * 2.0

    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.001),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(y, expected)


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_all_gather(shard_dim):
    """Test all_gather operation with sharding on different dimensions.
    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    xr.set_device_type("TT")
    enable_spmd()
    # Random inputs
    t = torch.rand(8192, 784)
    golden = copy.deepcopy(t)
    print("Golden shape: ", golden.shape)

    t = t.to(torch_xla.device())

    if shard_dim == 0:
        mesh = create_device_mesh((2, 1), ("batch", "model"))
        # 2 devices along the “batch” axis, 1 along “model”.
        xs.mark_sharding(t, mesh, ("batch", None))
        groups = [[0, 1]]
        gather_dim = 0
    else:
        mesh = create_device_mesh((1, 2), ("batch", "model"))
        xs.mark_sharding(t, mesh, (None, "model"))
        groups = [[0, 1]]
        gather_dim = 1

    y = xm.all_gather(t, gather_dim, groups=groups, pin_layout=False)

    y = y.to("cpu")
    print(f"All-gather shard dim: {shard_dim}, Y Shape: {y.shape}")
    chunks = torch.chunk(y, len(groups[0]), dim=gather_dim)

    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.001),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    for i in range(1, len(chunks)):
        comparator.compare(chunks[i], chunks[0])
    comparator.compare(chunks[0], golden)
