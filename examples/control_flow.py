import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import sys

from torch_xla.experimental import plugins
class TTPjrtPlugin(plugins.DevicePlugin):

  def library_path(self):
    return os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")

plugins.register_plugin("TT", TTPjrtPlugin())

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed.distributed_c10d as c10d
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

def init_process():
    dist.init_process_group('xla', init_method='xla://')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor):
        tensor = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, tensor, 1.0, [[0], [1]])
        return tensor

def control_flow():
    class ControlFlow(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x - 1
            y = x.item()
            print(y)
            if y < 4:
                return x + 1
            return x * 3

    model = ControlFlow()
    device = xm.xla_device()
    model = model.to(device)
    input = torch.ones(1, device="cpu") * 6
    input = input.to(device)
    out = model(input)
    print(out)
    imput = torch.ones(1, device="cpu")
    imput = imput.to(device)
    out = model(imput)
    print(out)

    input = torch.ones(1, device="cpu") * 6
    input = input.to(device)
    out = model(input)
    print(out)

    input = torch.ones(1, device="cpu")
    input = input.to(device)
    out = model(input)
    print(out)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["TT_XLA_NUM_DEVICES"] = "1"


def run_twice():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x + 7
            x = x * 7
            x = x + 1
            x = torch.nn.functional.gelu(x)
            return x

    model = Basic()
    device = xm.xla_device()
    model = model.to(device)
    input = torch.ones(1, device="cpu")
    input = input.to(device)
    out = model(input)
    print(out)
    out = model(input)
    print(out)

if __name__ == "__main__":
    run_twice()
    # control_flow()
