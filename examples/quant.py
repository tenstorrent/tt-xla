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
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(32, 32)

    def forward(self, tensor):
        tensor = self.linear(tensor)
        return tensor


def quant():
    device = xm.xla_device()
    model = Model().eval()
    breakpoint()
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    model = torch.ao.quantization.prepare(model)
    model_int8 = torch.ao.quantization.convert(model)
    model_int8 = model_int8.to(device)
    input = torch.randn(32, 32, device='cpu')
    input = input.to(device)
    out = model_int8(input)
    print(out)



os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["TT_XLA_NUM_DEVICES"] = "2"

if __name__ == "__main__":
    quant()
