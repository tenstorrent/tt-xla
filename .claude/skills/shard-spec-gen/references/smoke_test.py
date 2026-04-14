import numpy as np
import torch
import torch_xla
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh

xr.set_device_type("TT")
enable_spmd()

num_devices = xr.global_runtime_device_count()
assert num_devices > 0, "No TT devices found"

mesh = Mesh(np.arange(num_devices), (1, num_devices), ("batch", "model"))

w = torch.randn(64, 64, dtype=torch.bfloat16)
x = torch.randn(1, 64, dtype=torch.bfloat16).to(torch_xla.device())
out = x @ w.to(torch_xla.device())
torch_xla.sync()

print(f"OK — {num_devices} devices, mesh shape {mesh.mesh_shape}")
