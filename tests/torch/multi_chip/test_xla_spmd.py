# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.experimental import plugins

os.environ["PJRT_DEVICE"] = "tt"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        return os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")


plugins.register_plugin("tt", TTPjrtPlugin())


# Enable XLA SPMD execution mode.
xr.use_spmd()

# Device mesh, partition spec and input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()
device_ids = np.array(range(num_devices))
mesh_shape = (1, 2)
partition_spec = ("x", "y")
mesh = xs.Mesh(device_ids, mesh_shape, partition_spec)

# Random inputs between 0 and 0.1
t = torch.rand(8192, 784) * 0.1
w = torch.rand(784, 8192) * 0.1

golden = t @ w

t = t.to(xm.xla_device())
w = w.to(xm.xla_device())

xs.mark_sharding(t, mesh, partition_spec)
xs.mark_sharding(w, mesh, ("y", None))

# spmd all-reduce doesnt have an easily accessible API, so we use the internal API (for now)
y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, t @ w, 1.0, [[0, 1]])
assert torch.allclose(y.to("cpu"), golden, atol=0.02)
