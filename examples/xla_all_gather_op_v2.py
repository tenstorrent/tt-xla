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

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

from torch_xla.experimental import plugins


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        return os.path.join(
            os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
        )


plugins.register_plugin("TT", TTPjrtPlugin())

# Enable XLA SPMD execution mode.
xr.use_spmd()


# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()

mesh_shape = (2, 4)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("x", "y"))

# Random inputs between 0 and 0.1
t = (torch.rand(1024, 8192) - 0.0) * 0.1
w = (torch.rand(8192, 4096) - 0.0) * 0.1

golden = t @ w

t = t.to(xm.xla_device())
w = w.to(xm.xla_device())

xs.mark_sharding(t, mesh, ("x", None))
xs.mark_sharding(w, mesh, (None, None))

from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding

print("Sharding of t:")
visualize_tensor_sharding(t, use_color=False)
print("Sharding of w:")
visualize_tensor_sharding(w, use_color=False)

# In this version of the all gather, we perform the all-gather on t first to
# reconstruct the full tensor on all devices, and then perform the matmul.
t = torch_xla._XLAC._xla_all_gather(t, 0, 2, [[0, 4], [1, 5], [2, 6], [3, 7]], True)
# t = xm.all_gather(t, 0, [[0, 4], [1, 5], [2, 6], [3, 7]], True)

y = t @ w

y = y.to("cpu")
print(f"Y Shape: {y.shape}")
print(f"Golden Shape: {golden.shape}")
print(f"Y: {y}")
print(f"Golden: {golden}")
t = t.to("cpu")
print(f"T Shape: {t.shape}")
print(f"T: {t}")
