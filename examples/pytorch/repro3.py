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

def setup_xla_environment():
    xr.use_spmd()

def create_device_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh

def main():
    xr.set_device_type("TT")

    setup_xla_environment()
    # os.environ["XLA_FORCE_HOST_PLATFORM_DEVICE_COUNT"] = "2"
    # os.environ["CPU_NUM_DEVICES"] = "2" # this seems sufficient when run from inside the test

    mesh = create_device_mesh()
    device = torch_xla.device()
    x = torch.ones((32,16)).to(device)
    xs.mark_sharding(x, mesh, (None, 'model'))
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))
    x += 1
    print(torch_xla._XLAC._get_xla_tensors_hlo([x]))
    torch_xla.sync()
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))

    # print(x)

main()


# related - XLA_ENABLE_PARAM_ALIASING=0 => this does disable parameter aliasing causing metric InputOutputAliasCount to go from 1 -> 0
# however, it seems unrelated to the result, which is still replicated. ? maybe the result is a different tensor either way, but the 
# original input gets lost...

#  XLA_ENABLE_PARAM_ALIASING=0 LOGGER_LEVEL=DEBUG python examples/pytorch/repro2.py

# Seems like this can repro the fact that x+=1 retains sharding, so it's not torchxla by default