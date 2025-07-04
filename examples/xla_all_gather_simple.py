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

# Enable XLA SPMD execution mode.
xr.use_spmd()


# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()

mesh_shape = (2, 4)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

# Random inputs between 0 and 0.1
t = (torch.rand(8192, 784) - 0.0) * 0.1

t = t.to(xm.xla_device())


# Original
xs.mark_sharding(t, mesh, ("batch", "model"))
# New
# xs.enable_manual_sharding(t, ("batch", "model"), mesh=mesh)

# Internal all-gather API
# For PyTorch v2.8 and above, use the following:
y = torch_xla._XLAC._xla_all_gather(
    t, 0, 4, [[0,1,2,3], [4,5,6,7]], True, 1, True
)


# For PyTorch v2.7, use the following:
# y = torch_xla._XLAC._xla_all_gather(
#     t, 0, 4, [[0,1,2,3], [4,5,6,7]], True, 
# )

# y = xm.all_gather(t, 0, [[0,1,2,3], [4,5,6,7]], pin_layout=True)
# xs.disable_manual_sharding(y, ("batch", "model"), full_shape=, mesh=mesh)
y = y.to("cpu")
print(f"Y Shape: {y.shape}")
print(f"Y: {y}")