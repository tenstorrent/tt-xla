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

import torch_xla.core.xla_model as xm



class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        return self.fc1(x)


os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

def sanity():
    device = xm.xla_device()
    model = Linear()
    model = model.to(device)
    input = torch.randn(32, 32, dtype=torch.float32, device='cpu')
    input = input.to(device)
    out = model(input)
    print(out)


if __name__ == "__main__":
    sanity()
