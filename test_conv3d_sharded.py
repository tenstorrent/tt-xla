# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig, TorchComparisonEvaluator
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
import os

class SimpleConv3dModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

    def forward(self, x):
        return self.conv(x)

def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh

def run_conv3d_on_tt(in_channels, out_channels, input_shape, kernel_size=3, stride=1, padding=0):
    torch.manual_seed(42)
    model: nn.Module = SimpleConv3dModel(
        in_channels, out_channels, kernel_size, stride, padding
    ).to(dtype=torch.bfloat16)
    model = model.eval()
    model.compile(backend="tt")

    torch.manual_seed(123)
    input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)

    device = xm.xla_device()
    input_tensor = input_tensor.to(device)
    
    
    # shard input tensor
    mesh = create_device_mesh()
    # input tensor is BCDHW
    xs.mark_sharding(input_tensor, mesh, ("batch", None, "model", None, None))
    
    model = model.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    return output


def run_conv3d_on_cpu(in_channels, out_channels, input_shape, kernel_size=3, stride=1, padding=0):
    torch.manual_seed(42)
    model: nn.Module = SimpleConv3dModel(
        in_channels, out_channels, kernel_size, stride, padding
    ).to(dtype=torch.bfloat16)
    model = model.eval()
    model = torch.compile(model, backend="inductor")

    torch.manual_seed(123)
    input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)

    with torch.no_grad():
        output = model(input_tensor)

    return output


@pytest.mark.parametrize(
    "in_channels,out_channels,input_shape,kernel_size,stride,padding",
    [
        (1, 1, (1, 1, 3, 8, 8), 3, 1, 0),
        (2, 2, (1, 2, 3, 8, 8), 3, 1, 0),
        (3, 3, (1, 3, 3, 8, 8), 3, 1, 0),
        (4, 8, (1, 4, 3, 8, 8), 3, 1, 0),
        (8, 16, (1, 8, 3, 8, 8), 3, 1, 0),
        (16, 32, (1, 16, 3, 16, 16), 3, 1, 0),
        (32, 64, (1, 32, 3, 16, 16), 3, 1, 0),
        (8, 16, (1, 8, 3, 8, 16), 3, 1, 0),
        (8, 16, (1, 8, 5, 8, 8), 3, 1, 0),
        (8, 16, (1, 8, 3, 8, 8), 1, 1, 0),
        (8, 16, (1, 8, 5, 12, 12), 5, 1, 0),
        (32, 96, (1, 32, 3, 258, 258), 3, 1, 0), # most relevant example
    ],
    ids=[
        "minimal_1to1",
        "small_2to2",
        "rgb_3to3",
        "expand_4to8",
        "expand_8to16",
        "medium_16to32",
        "medium_32to64",
        "asymmetric_hw",
        "temporal_depth5",
        "kernel1x1x1",
        "kernel5x5x5",
        "large_32to96",
    ],
)
def test_conv3d_sharded(in_channels, out_channels, input_shape, kernel_size, stride, padding):
    xr.set_device_type("TT")
    
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    xr.use_spmd()
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    output_tt = run_conv3d_on_tt(in_channels, out_channels, input_shape, kernel_size, stride, padding)
    output_cpu = run_conv3d_on_cpu(in_channels, out_channels, input_shape, kernel_size, stride, padding)

    evaluator = TorchComparisonEvaluator(ComparisonConfig())
    evaluator.evaluate(output_tt.cpu(), output_cpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
