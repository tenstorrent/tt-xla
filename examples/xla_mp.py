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


import jax
import jax._src.xla_bridge as xb


def initialize_pjrt():
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
    plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
    jax.config.update("jax_platforms", "tt,cpu")

def init_process(rank, world_size):
    dist.init_process_group('xla', init_method='xla://')
    # dist.init_process_group(backend="jax", init_method="jax://")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor):
        tensor = tensor + 1
        dist.all_reduce(tensor)

        return tensor

def dar(rank, world_size):
    init_process(rank, world_size)
    tensor = torch.ones(2, device='cpu') * (rank + 1)
    xla_rank = xr.global_ordinal()
    world_size = xr.world_size()
    print(f"Rank: {rank}\nWorld Size: {world_size}")
    assert(xla_rank == rank)
    device = xm.xla_device()
    print(f"Device: {device}")
    tensor = tensor.to(device)
    print(f"Tensor: {tensor}")
    model = Model()
    #prog = export(model, (tensor,))
    #prog.graph_module.graph.print_tabular()
    #shlo = exported_program_to_stablehlo(prog)
    #print(shlo.get_stablehlo_text())
    out = model(tensor)
    # out = out.to('cpu')
    # print(out)
    dist.destroy_process_group()
    print(f"Out: {out}")

def mp():
    world_size = 4
    torch_xla.launch(dar, args=(world_size,))

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        return self.fc1(x)

def sanity():
    device = xm.xla_device()
    model = Linear()
    model = model.to(device)
    input = torch.randn(32, 32, dtype=torch.float32, device='cpu')
    input = input.to(device)
    out = model(input)
    breakpoint()
    print(out)


if __name__ == "__main__":
    sanity()
