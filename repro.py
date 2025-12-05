import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import os
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
import torch_xla

# use_tt = os.getenv("USE_TT") == 1

use_tt = True


def create_device_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
    os.environ["PT_XLA_DEBUG"] = "1"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    xr.use_spmd()

    if use_tt:
        xr.set_device_type("TT")
    else:
        os.environ["CPU_NUM_DEVICES"] = "8"


    mesh = create_device_mesh()
    device = torch_xla.device()
    # x = torch.ones((32,16)).to(device)
    x = torch.arange((1024), dtype=torch.int).reshape((32,32)).to(device)
    xs.mark_sharding(x, mesh, (None, 'model'))

    print(torch_xla._XLAC._xla_get_all_device_attributes())
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x), "device", x.device)
    x += 1
    # x@=x
    print(torch_xla._XLAC._get_xla_tensors_hlo([x]))
    torch_xla.sync()

    print("input sharding",torch_xla._XLAC._get_xla_sharding_spec(x), "device", x.device)
    print(x)

main()