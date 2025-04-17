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
default_group = None
def _tt_get_default_group():
    if default_group is None:
        return c10d.GroupMember.WORLD
    return default_group

c10d._get_default_group = _tt_get_default_group

def init_process():
    dist.init_process_group('xla', init_method='xla://')
    global default_group
    # pg_options = {'xla_pg_options': {'mesh': [[1, 2]]}}
    num_devices = int(sys.argv[1])
    default_group = dist.new_group([i for i in range(num_devices)])

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor):
        tensor = torch_xla._XLAC._xla_spmd_all_reduce(xm.REDUCE_SUM, tensor, 1.0, [[0,1]])
        return tensor

def dar(rank):
    init_process()
    print("process initialized")
    tensor = torch.ones(6, device='cpu') * (rank + 1)
    model = Model()
    # prog = export(model, (tensor, ))
    # stablehlo = exported_program_to_stablehlo(prog)
    # print(stablehlo.get_stablehlo_text())
    xla_rank = xr.global_ordinal()
    world_size = xr.world_size()
    print(f"Rank: {rank}\nWorld Size: {world_size}")
    assert(xla_rank == rank)
    device = xm.xla_device()
    print(f"Device: {device}")
    tensor = tensor.to(device)
    print(f"Tensor: {tensor}")
    out = model(tensor)
    # out = out.to('cpu')
    # print(out)
    dist.destroy_process_group()
    print(f"Out: {out}")

def mp():
    num_devices = int(sys.argv[1])
    if num_devices == 1:
        dar(0)
    else:
        torch_xla.launch(dar, args=())


os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

if __name__ == "__main__":
    mp()