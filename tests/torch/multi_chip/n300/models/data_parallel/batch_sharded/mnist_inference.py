# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental import plugins
import pytest

from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["MESH_SHAPE"] = "8,1"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            # Find tt-xla repo root by traversing up from current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            while current_dir != "/" and not os.path.exists(os.path.join(current_dir, "build")):
                current_dir = os.path.dirname(current_dir)
            return os.path.join(current_dir, "build/src/tt/pjrt_plugin_tt.so")

    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.set_device_type("TT")
    xr.use_spmd()


class MNISTLinear(nn.Module):
    """Simple linear MNIST model for inference testing."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 512, num_classes: int = 10, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)


def inference_on_multiple_devices(batch_size: int = 64, input_size: int = 784):
    """Run MNIST linear inference with data parallel across multiple devices."""
    
    setup_tt_environment()
    torch.manual_seed(42)
    
    # Get number of devices and create mesh
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids=device_ids, mesh_shape=(num_devices,), axis_names=("data",))
    
    # Create model and move to device
    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)
    device = torch_xla.device()
    model = model.to(device)
    
    # Initialize model parameters with some values for testing
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.01)
    
    # Create input data
    total_batch_size = batch_size * num_devices
    input_data = torch.randn(total_batch_size, input_size, dtype=torch.bfloat16)
    
    # Move to device and mark sharding
    sharded_input = input_data.to(device, dtype=torch.bfloat16)
    xs.mark_sharding(sharded_input, mesh, ("data", None))
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(sharded_input)
    
    # Synchronize and return results
    torch_xla.sync(wait=True)
    return outputs, input_data


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        (64, 784),
        (128, 784),
    ],
)
def test_mnist_linear_inference_multichip(batch_size: int, input_size: int):
    """Test MNIST linear inference with data parallel across multiple chips."""
    
    # Run CPU reference
    torch.manual_seed(42)
    cpu_model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)
    
    # Initialize with same values as device model
    with torch.no_grad():
        for param in cpu_model.parameters():
            param.fill_(0.01)
    
    # Run multichip inference first to get device count
    tt_output, input_data = inference_on_multiple_devices(batch_size, input_size)
    
    # Create CPU input with same size as was used on devices  
    input_data = input_data.cpu()
    
    # CPU inference
    cpu_model.eval()
    with torch.no_grad():
        cpu_output = cpu_model(input_data)
    
    # Compare results
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05),
            pcc=PccConfig(required_pcc=0.95),
        )
    )
    comparator.compare(tt_output.cpu(), cpu_output)