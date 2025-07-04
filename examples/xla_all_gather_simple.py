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
import copy
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding
import pytest


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
    os.environ["LOGGER_LEVEL"] = "DEBUG"

    from torch_xla.experimental import plugins

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
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


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_all_gather(shard_dim):
    """Test all_gather operation with sharding on different dimensions.

    Args:
        shard_dim: Dimension to shard on (0 for batch, 1 for model)
    """
    setup_tt_environment()
    mesh = create_mesh()

    # Random inputs between 0 and 0.1
    t = (torch.rand(8192, 784) - 0.0) * 0.1
    golden = copy.deepcopy(t)

    t = t.to(torch_xla.device())

    if shard_dim == 0:
        # Shard on batch dimension (dim 0)
        xs.mark_sharding(t, mesh, ("batch", None))
        # Correct replica groups for batch sharding: pair devices across batch rows
        groups = [[0, 4], [1, 5], [2, 6], [3, 7]]
        gather_dim = 0
    else:
        # Shard on model dimension (dim 1)
        xs.mark_sharding(t, mesh, (None, "model"))
        # For model sharding: two groups of 4 devices each (one group per batch row)
        groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        gather_dim = 1

    y = xm.all_gather(t, gather_dim, groups=groups, pin_layout=False)

    y = y.to("cpu")
    print(f"Shard dim: {shard_dim}, Y Shape: {y.shape}")

    assert torch.allclose(y, golden, atol=0.001)


if __name__ == "__main__":
    test_all_gather(0)
    test_all_gather(1)
    print("All tests passed!")
