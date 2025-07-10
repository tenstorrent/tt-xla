# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import os
import pytest

# needs to be set at module level to unsure it gets picked up before torch-xla C++ code is initialized
os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
    os.environ["LOGGER_LEVEL"] = "DEBUG"

    from torch_xla.experimental import plugins

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so"
            )

    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.use_spmd()
    torch_xla.sync(True, True)


def create_mesh():
    """Create device mesh for testing."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))

def test_matmul():
    """Test distributed matmul operation with sharded inputs and CCL collectives.
    
    This test demonstrates a distributed matrix multiplication where:
    1. Input matrices A and B are sharded across devices
    2. Local matmuls are performed on each device
    3. Results are combined using all-reduce to get the final output
    """
    setup_tt_environment()
    mesh = create_mesh()
    
    # Matrix dimensions
    M, N, K = 2048, 1024, 512
    
    # Create random input matrices
    a = (torch.rand(M, K) - 1.0) * 0.1
    b = (torch.rand(K, N) - 1.0) * 0.1
    
    # Compute reference output on CPU
    expected = torch.matmul(a, b)
    
    # Move tensors to XLA device
    a = a.to(torch_xla.device())
    b = b.to(torch_xla.device())
    
    # Test with 1D sharding first - only shard A along batch dimension
    print(f"Sharding A with shape {a.shape}")
    print(f"Sharding B with shape {b.shape}")
    
    xs.mark_sharding(a, mesh, ("batch", None))
    xs.mark_sharding(b, mesh, (None, "model"))
    
    # Perform local matmul on each device
    print("Performing local matmul...")
    c_local = torch.matmul(a, b)
    print(f"Local matmul result shape: {c_local.shape}")

    c_batch_gathered = xm.all_gather(c_local, 0, groups=[[0, 4], [1, 5], [2, 6], [3, 7]], pin_layout=False)
    c = xm.all_gather(c_batch_gathered, 1, groups=[[0, 1, 2, 3], [4, 5, 6, 7]], pin_layout=False)
    c = c.cpu()

    # c = xs.enable_manual_sharding(c_local, (None, None), mesh=mesh)
    # c = c_local.cpu()
    
    print(f"Input A shape: {a.shape}")
    print(f"Input B shape: {b.shape}")
    print(f"Output C shape: {c.shape}")
    print(f"Expected shape: {expected.shape}")
    
    # Verify the result matches the reference
    # Note: We need to handle potential padding in the distributed case
    assert c.shape == expected.shape, f"Shape mismatch: {c.shape} vs {expected.shape}"
    assert torch.allclose(c, expected, atol=0.01), \
        "Distributed matmul result does not match reference"
    
    print("Distributed matmul test passed!")


if __name__ == "__main__":
    test_matmul()
    print("All tests passed!")
