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

from jax import config

# config.update("jax_default_device", "tpu")
# config.update("jax_enable_x64", True)
config.update("jax_use_shardy_partitioner", True)

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

"""
First 512 rows of t are in 0,1,2,3 ; t_local = (512, 8192)
Second 512 rows of t are in 4,5,6,7 ; t_local = (512, 8192)

w = (8192, 4096)

y_local = t_local @ w = (512, 4096)


0 1
2 3
4 5
6 7


Concat between [0,4] should be done with mesh_shard op.

[
  1 2
  3 4
]

0 1 2 3
4 5 6 7
"""

mesh_shape = (2, 4)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("x", "y"))

# Random inputs between 0 and 0.1
a = (torch.rand(1024, 8192) - 0.0) * 0.1
b = (torch.rand(8192, 4096) - 0.0) * 0.1
c = (torch.rand(1024, 4096) - 0.0) * 0.1
d = (torch.rand(1024, 4096) - 0.0) * 0.1
e = (torch.rand(1024, 4096) - 0.0) * 0.1
f = (torch.rand(1024, 4096) - 0.0) * 0.1
g = (torch.rand(1024, 4096) - 0.0) * 0.1

# golden = t @ w

# For PyTorch v2.7
a = a.to(xm.xla_device())
b = b.to(xm.xla_device())
c = c.to(xm.xla_device())
d = d.to(xm.xla_device())
e = e.to(xm.xla_device())
f = f.to(xm.xla_device())
g = g.to(xm.xla_device())

# For PyTorch v2.8 and above, use the following:
# t = t.to(torch_xla.device())
# w = w.to(torch_xla.device())
# z = z.to(torch_xla.device())

# xs.mark_sharding(a, mesh, (("y", "x"), None))

xs.mark_sharding(a, mesh, ("x", None))
xs.mark_sharding(b, mesh, (None, "y"))
xs.mark_sharding(c, mesh, (None, None))
xs.mark_sharding(d, mesh, ("x", "y"))
xs.mark_sharding(e, mesh, ("y", "x"))
xs.mark_sharding(f, mesh, (None, "x"))
xs.mark_sharding(g, mesh, ("y", None))
# xs.mark_sharding(g, mesh, (("y", "x"), None))

# t = t + t
# xm.mark_step()

from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding

# print("Sharding of a:")
# visualize_tensor_sharding(a, use_color=False)
# print("Sharding of b:")
# visualize_tensor_sharding(b, use_color=False)
# print("Sharding of c: ")
# visualize_tensor_sharding(c, use_color=False)
# print("Sharding of d: ")
# visualize_tensor_sharding(d, use_color=False)
# print("Sharding of e: ")
# visualize_tensor_sharding(e, use_color=False)
# print("Sharding of f: ")
# visualize_tensor_sharding(f, use_color=False)
# print("Sharding of g: ")
# visualize_tensor_sharding(g, use_color=False)


a = a.to("cpu")
b = b.to("cpu")
c = c.to("cpu")
d = d.to("cpu")
e = e.to("cpu")
f = f.to("cpu")
g = g.to("cpu")

# # In this version of the all gather, we perform the matmul first and then
# # all-gather the matmul result.
# y_local = t @ w

# # Internal all-gather API
# y = torch_xla._XLAC._xla_all_gather(
#     y_local, 0, 2, [[0, 4], [1, 5], [2, 6], [3, 7]], True, None, None
# )
# # y = xm.all_gather(y_local, 0, [[0, 4], [1, 5], [2, 6], [3, 7]], pin_layout=True)

# y = y.to("cpu")
# print(f"Y Shape: {y.shape}")
# print(f"Golden Shape: {golden.shape}")
# print(f"Y: {y}")
# print(f"Golden: {golden}")
