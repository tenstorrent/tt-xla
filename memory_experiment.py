# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.core.xla_builder import create_placeholder_tensor
import torch_xla

class SimpleAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single parameter tensor for elementwise addition
        self.add_tensor = nn.Parameter(torch.ones(16, 16, dtype=torch.bfloat16))

    def forward(self, x):
        return x + self.add_tensor


def test_simple_add():
    """Test simple elementwise addition on TT device."""
    # Set device type to TT
    xr.set_device_type("TT")

    # Instantiate model
    torch.manual_seed(42)
    model = SimpleAddModel()

    # Put it in inference mode
    model = model.eval()

    # Get TT device
    device = xm.xla_device()

    print("Moving model to device...", flush=True)
    # Move model to device
    model = model.to(device)

    print(f"Model successfully moved to device: {device}")

    # Create 16x16 input
    
    # inputs = torch.randn(16, 16, dtype=torch.float32)
    # inputs.numpy().tofile('16x16xf32.pt')
    
    # need to specify size / dtype as this gives you back a flat tensor
    inputs = torch.from_file('16x16xf32.pt', dtype=torch.float32, size=256).reshape(16, 16).to(torch.bfloat16)
    storage = inputs.untyped_storage()
    print(f"Storage size: {storage.size()}")
    print(f"Storage data_ptr: {storage.data_ptr()} | filepath: {storage.filename}")
    print(f"Is storage memory-mapped: {storage.is_shared()}")  # Should be True for shared=True
    # inputs.numpy().tofile('16x16xbf16.pt')
    print(inputs.shape)
    print("Moving inputs to device...", flush=True)
    inputs = inputs.to(device)

    # Perform elementwise addition
    output = model(inputs)

    # print(f"Output device: {output.device}")
    # print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


class SimpleAddModelPlaceholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single parameter tensor for elementwise addition
        self.add_tensor = create_placeholder_tensor(shape=(16, 16), dtype=torch.bfloat16)

    def forward(self, x):
        return x + self.add_tensor

def test_create_placeholder_tensor():
    """Test creating a placeholder tensor on TT device."""
    # Set device type to TT
    xr.set_device_type("TT")

    # Create a placeholder tensor
    placeholder = create_placeholder_tensor(shape=(16, 16), dtype=torch.bfloat16)
    print(f"Placeholder tensor: {placeholder.shape}")
    device = xm.xla_device()
    placeholder = placeholder.to(device)
    
    print("placeholder device: ", placeholder.device)
    
    model = SimpleAddModelPlaceholder()
    model.compile(backend="tt")
    
    model = model.to(device)

    output = model(placeholder)
    torch_xla.sync()



def test_simple_spmd():
    """Test simple elementwise addition on TT device."""
    # Set device type to TT
    import os
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()
    # Set up SPMD and create a 1x2 device mesh.
    import torch_xla.distributed.spmd as xs
    import numpy as np
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import Mesh
    num_devices = xr.global_runtime_device_count()

    # Create a 1x2 mesh using 2 TT logical devices
    devices = np.array(range(num_devices))
    mesh = Mesh(devices, (1, num_devices), ("batch", "model"))

    print(f"SPMD mesh created: {mesh}")


    # Instantiate model
    torch.manual_seed(42)
    model = SimpleAddModel()

    # Put it in inference mode
    model = model.eval()

    # Get TT device
    device = xm.xla_device()

    print("Moving model to device...", flush=True)
    # Move model to device
    model = model.to(device)

    print(f"Model successfully moved to device: {device}")

    # Create 16x16 input
    
    inputs = torch.randn(16, 16, dtype=torch.float32)
    # inputs.numpy().tofile('16x16xf32.pt')
    
    # need to specify size / dtype as this gives you back a flat tensor
    print(inputs.shape)
    print("Moving inputs to device...", flush=True)
    inputs = inputs.to(device)
    
    print("Marking inputs for sharding...", flush=True)
    inputs = xs.mark_sharding(inputs, mesh, (None, "model"))
    # Perform elementwise addition
    print("Running model...", flush=True)
    output = model(inputs)

    # print(f"Output device: {output.device}")
    # print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")