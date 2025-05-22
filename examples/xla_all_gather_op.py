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

# These cause the number of devices error
# t = t + t
# t = 1 * t
# xm.mark_step()

# Adding a new tensor, doesn't cause the error. But the all gather shape is still wrong.
# z = t + t
# xm.mark_step()

from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding

print("Sharding of t:")
visualize_tensor_sharding(t, use_color=False)
print("Sharding of w:")
visualize_tensor_sharding(w, use_color=False)

y_local = t @ w

# print("Before all-gather, t shape: ", t.shape)
# spmd all-gather doesnt have an easily accessible API, so we use the internal API (for now)
y = torch_xla._XLAC._xla_all_gather(
    y_local, 0, 2, [[0, 4], [1, 5], [2, 6], [3, 7]], True
)


# print("After all-gather, t shape: ", t.shape)
# print("After all-gather, w device: ", w.shape)

y = y.to("cpu")
print(f"Y Shape: {y.shape}")
print(f"Golden Shape: {golden.shape}")
print(f"Y: {y}")
print(f"Golden: {golden}")
# print(f"Is close: {torch.allclose(y, golden, atol=0.5)}")
t = t.to("cpu")
print(f"t Shape: {t.shape}")
# unique_rows, counts = torch.unique(t, dim=0, return_counts=True)
# duplicate_mask = counts > 1
# duplicates = unique_rows[duplicate_mask]
# zero_row_mask = (t == 0).all(dim=0)
# zero_rows = t[zero_row_mask]
# print(f"Zero rows shape: {zero_rows.shape}")
# print(f"Zero rows:\n{zero_rows}")
# print(f"Duplicate rows shape: {duplicates.shape}")
# print(f"Duplicate rows from t:\n{duplicates}")
# print(f"Unique rows shape from t:\n{unique_rows.shape}")
# print(f"Unique rows from t:\n{unique_rows}")
# assert torch.allclose(y, golden, atol=0.5)
