# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace with data parallel

import os
import torch
import torch.nn as nn
import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from infra.utilities.torch_multichip_utils import enable_spmd

class MNISTLinear(nn.Module):
    """Simple linear MNIST model for inference testing."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_classes: int = 10,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

def replicate_bias(bias, mesh):
    """
    Replicate a bias tensor across all devices in the mesh.

    Use this when the corresponding weight is sharded along the inner dimension (columns),
    meaning the bias needs to be replicated on all devices since each device computes
    partial results that need the full bias vector.

    Args:
        bias: Bias tensor to replicate (can be None)
        mesh: XLA SPMD mesh defining the device topology
    """
    if bias is None:
        return
    xs.clear_sharding(bias)
    xs.mark_sharding(bias, mesh, (None,))  # replicate (no axis mapping)


def shard_bias(bias, mesh):
    """
    Shard a bias tensor along the 'model' axis.

    Use this when the corresponding weight is sharded along the outer dimension (rows),
    meaning each device computes a subset of the output features and needs only
    the corresponding slice of the bias vector. Raises ValueError if bias size
    is not evenly divisible by the number of devices.

    Args:
        bias: Bias tensor to shard (can be None)
        mesh: XLA SPMD mesh defining the device topology

    Raises:
        ValueError: If bias length is not divisible by the number of devices
    """
    num_devices = xr.global_runtime_device_count()
    xs.clear_sharding(bias)

    if bias is None:
        return
    if bias.numel() % num_devices == 0:
        # bias can be sharded
        xs.mark_sharding(bias, mesh, ("model",))
    else:
        msg = (
            f"Bias length {bias.numel()} not divisible by #devices={num_devices}; "
            "replicating bias instead of sharding."
        )
        raise ValueError(msg)


# Set up TT backend
xr.set_device_type("TT")
enable_spmd()
        
# Set up codegen options
options = {
     "backend": "codegen_py",
     "export_path": "mnist_codegen_tp",
}
torch_xla.set_custom_compile_options(options)

torch.manual_seed(42)

batch_size = 32
input_size = 784

model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

# Create random input data.
input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

# Move inputs and model to device.
device = xm.xla_device()
model = model.eval()
model.compile(backend="tt")
model = model.to(device)
input_data = input_data.to(device)

# Create a mesh.
num_devices = xr.global_runtime_device_count()
mesh_shape = (1, num_devices)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

# fc1: weight [hidden, input] -> shard rows (output features)
xs.mark_sharding(model.fc1.weight, mesh, ("model", None))
shard_bias(model.fc1.bias, mesh)

# fc2: weight [hidden, hidden] -> shard cols (input features)
xs.mark_sharding(model.fc2.weight, mesh, (None, "model"))
replicate_bias(model.fc2.bias, mesh)

# fc3: weight [num_classes, hidden] -> shard rows (out_features)
xs.mark_sharding(model.fc3.weight, mesh, ("model", None))
shard_bias(model.fc3.bias, mesh)

# 2D input (N, D) -> (None, None) means no sharding (replicated) on both dims
xs.mark_sharding(input_data, mesh, (None, None))

# Run model
with torch.no_grad():
    output = model(input_data)
    
print(f"Output shape: {output.shape}")
print("Success!")
