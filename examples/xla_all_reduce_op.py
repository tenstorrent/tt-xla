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
num_devices1 = len(xm.get_xla_supported_devices())
print(f"Number of devices (orig): {num_devices}")
print(f"Number of devices: {num_devices1}")

mesh_shape = (4, 2)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("x", "y"))


t = (torch.rand(1024, 8192) - 0.0) * 0.1
w = (torch.rand(8192, 4096) - 0.0) * 0.1

golden = t @ w

t = t.to(xm.xla_device())
w = w.to(xm.xla_device())

xs.mark_sharding(t, mesh, (None, "x"))
xs.mark_sharding(w, mesh, ("x", None))

from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding

print("Sharding of t:")
visualize_tensor_sharding(t, use_color=False)
print("Sharding of w:")
visualize_tensor_sharding(w, use_color=False)

# spmd all-reduce doesnt have an easily accessible API, so we use the internal API (for now)
y = t @ w

y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, y, 1.0, [[0]])
y = y.to("cpu")
print(f"Is close: {torch.allclose(y, golden, atol=0.5)}")
print("y: ")
print(y)
print("golden: ")
print(golden)
# assert torch.allclose(y, golden, atol=0.5)
