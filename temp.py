# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.comparators.torch_comparator import TorchComparator
from torch.utils.data import DataLoader
from torch_xla.distributed.spmd import Mesh
from torchvision import datasets, transforms

from tests.infra.comparators.comparison_config import ComparisonConfig


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    # os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    # os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "1,8"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
    # os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    # os.system('tt-smi -r')

    xr.set_device_type("TT")
    xr.use_spmd()


class ScatterOp(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, operand, scatter_indices, updates):
        """
        operand:         [71, 32]
        scatter_indices: [71, 4, 2]    (int64)
        updates:         [71, 4]       (bf16)

        stablehlo.scatter with:
        inserted_window_dims       = [0, 1]
        scatter_dims_to_operand    = [0, 1]
        index_vector_dim           = 2
        computation(old, new) = new
        """
        result = operand.clone()

        idx0 = scatter_indices[..., 0].long()  # [71, 4]
        idx1 = scatter_indices[..., 1].long()  # [71, 4]

        result[idx0, idx1] = updates

        return result


def test_scatter_sharding():
    setup_tt_environment()
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    axis_names = ("batch", "model")
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)

    # Create inputs on CPU first
    operand = torch.randint(0, 100, (71, 32), dtype=torch.bfloat16)
    # scatter_indices[:, :, 0] should be in [0, 71) and scatter_indices[:, :, 1] should be in [0, 32)
    scatter_indices = torch.zeros((71, 4, 2), dtype=torch.int64)
    scatter_indices[:, :, 0] = torch.randint(
        0, 71, (71, 4), dtype=torch.int64
    )  # First dimension
    scatter_indices[:, :, 1] = torch.randint(
        0, 32, (71, 4), dtype=torch.int64
    )  # Second dimension
    updates = torch.randint(100, 200, (71, 4), dtype=torch.bfloat16)

    # Compute golden output on CPU (no mesh or sharding needed)
    model_cpu = ScatterOp()
    golden = model_cpu(operand, scatter_indices, updates)

    # Now run on TT device
    device = torch_xla.device()
    model = ScatterOp().to(device=device, dtype=torch.bfloat16)

    operand = operand.to(device=device, dtype=torch.bfloat16)
    scatter_indices = scatter_indices.to(device=device, dtype=torch.int64)
    updates = updates.to(device=device, dtype=torch.bfloat16)
    xs.mark_sharding(operand, mesh, (None, "model"))

    output = model(operand, scatter_indices, updates)
    output = output.to("cpu")

    breakpoint()

    # Compare using the comparator
    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


if __name__ == "__main__":
    test_scatter_sharding()
