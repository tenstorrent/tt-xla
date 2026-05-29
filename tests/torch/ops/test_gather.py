# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.compiler_config import CompilerConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh


class SimpleGather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input, index):
        return torch.gather(input, self.dim, index)


# Note, 1D and >4D tensors are not supported by ttnn.gather. A workaround will be added
# to tt-mlir in the future to support them.
@pytest.mark.parametrize(
    "input_shape,index_shape,dim",
    [
        # ---- 2D: all dims (positive and negative), varying index shapes ----
        pytest.param((32, 64), (16, 64), 0, id="2d_dim0"),
        pytest.param((8, 32), (8, 16), 1, id="2d_dim1"),
        pytest.param((16, 32), (16, 32), 1, id="2d_dim1_same_shape"),
        pytest.param((16, 32), (16, 64), 1, id="2d_dim1_larger"),
        pytest.param((16, 32), (8, 16), 1, id="2d_dim1_both_smaller"),
        pytest.param((32, 64), (16, 32), 0, id="2d_dim0_both_smaller"),
        pytest.param((32, 64), (16, 64), -2, id="2d_negdim0"),
        pytest.param((8, 32), (8, 16), -1, id="2d_negdim1"),
        # ---- 3D: all dims, varying index shapes ----
        pytest.param((4, 32, 64), (2, 32, 64), 0, id="3d_dim0"),
        pytest.param((4, 32, 64), (4, 8, 64), 1, id="3d_dim1"),
        pytest.param((4, 16, 64), (4, 16, 32), 2, id="3d_dim2"),
        pytest.param((4, 16, 64), (4, 16, 128), 2, id="3d_dim2_larger"),
        pytest.param((4, 16, 64), (2, 8, 32), 2, id="3d_dim2_all_smaller"),
        pytest.param((4, 32, 64), (4, 8, 64), -2, id="3d_negdim1"),
        pytest.param((4, 16, 64), (4, 16, 32), -1, id="3d_negdim2"),
        # ---- 4D: all dims ----
        pytest.param((2, 4, 32, 64), (1, 4, 32, 64), 0, id="4d_dim0"),
        pytest.param((2, 4, 32, 64), (2, 2, 32, 64), 1, id="4d_dim1"),
        pytest.param((2, 4, 32, 64), (2, 4, 16, 64), 2, id="4d_dim2"),
        pytest.param((2, 4, 32, 64), (2, 4, 32, 16), 3, id="4d_dim3"),
        pytest.param((2, 4, 32, 64), (2, 4, 32, 16), -1, id="4d_negdim3"),
        pytest.param((2, 4, 32, 64), (1, 2, 16, 32), 2, id="4d_dim2_all_smaller"),
    ],
)
def test_gather_replicated(input_shape, index_shape, dim):
    xr.set_device_type("TT")

    input = torch.randn(*input_shape, dtype=torch.bfloat16)
    index_max = input_shape[dim]
    index = torch.randint(0, index_max, index_shape, dtype=torch.int64)

    run_graph_test(
        SimpleGather(dim),
        [input, index],
        framework=Framework.TORCH,
    )


def _build_shard_spec(rank, strategy):
    if strategy == "replicated":
        return tuple([None] * rank)
    if strategy == "shard_batch":
        return ("batch",) + tuple([None] * (rank - 1))
    if strategy == "shard_model":
        return tuple([None] * (rank - 1)) + ("model",)
    raise ValueError(f"unknown sharding strategy: {strategy}")


@pytest.mark.parametrize(
    "input_shape,index_shape,dim",
    [
        ((32,), (16,), 0),
        ((8, 32), (8, 16), 1),
        ((32, 64), (16, 64), 0),
        ((4, 32, 64), (4, 8, 64), 1),
        ((4, 16, 64), (4, 16, 32), 2),
        ((2, 4, 32, 64), (2, 4, 16, 64), 2),
        ((2, 2, 4, 16, 32), (2, 2, 4, 8, 32), 3),
    ],
    ids=["1d_dim0", "2d_dim1", "2d_dim0", "3d_dim1", "3d_dim2", "4d_dim2", "5d_dim3"],
)
@pytest.mark.parametrize("input_shard", ["replicated", "shard_batch", "shard_model"])
@pytest.mark.parametrize("index_shard", ["replicated", "shard_batch", "shard_model"])
def test_gather_simple(input_shape, index_shape, dim, input_shard, index_shard):
    xr.set_device_type("TT")

    class SimpleGather(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x, index):
            return torch.gather(x, self.dim, index)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, num_devices // 2) if num_devices >= 2 else (1, 1)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(module, args, kwargs):
        x, index = args
        return {
            x: _build_shard_spec(x.ndim, input_shard),
            index: _build_shard_spec(index.ndim, index_shard),
        }

    x = torch.randn(*input_shape, dtype=torch.bfloat16)
    index_max = input_shape[dim]
    index = torch.randint(0, index_max, index_shape, dtype=torch.int64)

    run_graph_test(
        SimpleGather(dim),
        [x, index],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
