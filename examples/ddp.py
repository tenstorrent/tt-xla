import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os


from torch_xla.experimental import plugins
class TTPjrtPlugin(plugins.DevicePlugin):

  def library_path(self):
    return os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")

plugins.register_plugin("TT", TTPjrtPlugin())

from torch_xla.stablehlo import exported_program_to_stablehlo
from torch.export import export
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from torch.nn.parallel import DistributedDataParallel as DDP

def init_process():
    dist.init_process_group('xla', init_method='xla://')


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        return self.fc1(x)


os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

def ddp():
    init_process()
    device = xm.xla_device()
    model = Linear()
    model = model.to(device)
    ddp_model = DDP(model)
    input = torch.randn(2, 32, 32, dtype=torch.float32, device='cpu')
    input = input.to(device)
    out = ddp_model(input)
    dist.destroy_process_group()
    print(out)


if __name__ == "__main__":
    ddp()
