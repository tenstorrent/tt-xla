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
    return os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")

plugins.register_plugin("TT", TTPjrtPlugin())

# Enable XLA SPMD execution mode.
xr.use_spmd()


# Device mesh, this and partition spec as well as the input tensor shape define the individual shard shape.
num_devices = xr.global_runtime_device_count()

mesh_shape = (1, 2)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ('height', 'features'))

# Random inputs between 0 and 0.1
t = torch.rand(8192, 784)*0.1
w = torch.rand(784, 8192)*0.1

golden = t @ w

t = t.to(xm.xla_device())
w = w.to(xm.xla_device())

partition_spec = ('height', 'features')
xs.mark_sharding(t, mesh, partition_spec)
xs.mark_sharding(w, mesh, ('features', None))

# spmd all-reduce doesnt have an easily accessible API, so we use the internal API (for now)
y = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, t @ w, 1.0, [[0, 1]])
assert torch.allclose(y.to('cpu'), golden, atol=0.02)