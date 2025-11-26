import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm

from transformers.models.llama.modeling_llama import LlamaModel
import os
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
import torch_xla



def setup_xla_environment():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1" # Converts the StableHLO emitted by torch-xla to the Shardy dialect
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
    mesh = create_device_mesh()
    device = torch_xla.device()

    x = torch.ones((32,16)).to(device)
    y = torch.ones((32,16)).to(device)


    xs.mark_sharding(x, mesh, (None, 'model'))


    print("PRE-EXECUTION")
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))
    print("Tensor id x: ", torch_xla._XLAC._xla_get_tensor_id(x))
    print("Tensor id y: ", torch_xla._XLAC._xla_get_tensor_id(y))
    print("Tensor buffer donation x: ", torch_xla._XLAC._get_buffer_donation(x))
    print("Tensor buffer donation y: ", torch_xla._XLAC._get_buffer_donation(y))

    x+=y
    torch_xla.sync()

    print("POST-EXECUTION")
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))
    print("Tensor id x: ", torch_xla._XLAC._xla_get_tensor_id(x))
    print("Tensor id y: ", torch_xla._XLAC._xla_get_tensor_id(y))

    print("Tensor buffer donation x: ", torch_xla._XLAC._get_buffer_donation(x))
    print("Tensor buffer donation y: ", torch_xla._XLAC._get_buffer_donation(y))
    print(torch_xla._XLAC._get_xla_tensor_debug_info(x))




    # print(torch_xla._XLAC._xla_metrics_report()) #io alias metric here

    # results = [el.to('cpu') for el in result]
    # print(results)

main()


# related - XLA_ENABLE_PARAM_ALIASING=0 => this does disable parameter aliasing causing metric InputOutputAliasCount to go from 1 -> 0
# however, it seems unrelated to the result, which is still replicated. ? maybe the result is a different tensor either way, but the 
# original input gets lost...

#  XLA_ENABLE_PARAM_ALIASING=0 LOGGER_LEVEL=DEBUG python examples/pytorch/repro2.py

