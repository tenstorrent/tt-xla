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

"""
This test demonstrates that a sharded input tensor
will lose its sharding if is also output from the graph

Reference verbose log: https://github.com/jameszianxuTT/logdump/blob/main/lost_sharding_simple_2025-11-26_19-58-36.log

On 1x8 llmbox:
x1 - model parameter is 32x32 sharded to local 4x32
x2 - user input is 32x16 sharded to local 32x2

"""

class FooModule(nn.Module):
    def __init__(self):
        super(FooModule, self).__init__()
        # Define x1 as a model param
        # x2 is the input
        self.x1 = nn.Parameter(torch.ones((32, 32)))

    def forward(self, x2):
        x2 *= 2
        y1 = self.x1 @ x2
        return y1, x2

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
    xs.mark_sharding(x, mesh, (None, 'model'))

    foo_model = FooModule()
    tt_model = torch.compile(foo_model, backend='tt')
    tt_model = tt_model.to(device)

    # must move model to device before sharding its param
    xs.mark_sharding(foo_model.x1, mesh, ('model', None))

    print("PRE-EXECUTION")
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))
    print("param sharding " + torch_xla._XLAC._get_xla_sharding_spec(foo_model.x1))

    result = tt_model(x)

    print("POST-EXECUTION")
    print("input sharding " + torch_xla._XLAC._get_xla_sharding_spec(x))
    print("param sharding " + torch_xla._XLAC._get_xla_sharding_spec(foo_model.x1))

    # results = [el.to('cpu') for el in result]
    # print(results)

main()