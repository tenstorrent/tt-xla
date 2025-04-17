import os
import torch
from torch import nn
import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo, StableHLOExportOptions
import torch_xla

import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from typing import List, Optional, TypedDict


import jax
import jax._src.xla_bridge as xb

def initialize_pjrt():
  path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "tt,cpu")

def test():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # return torch.distributed.all_gather([x], y)
            # return torch.distributed.all_reduce(x)
            # y = torch.distributed.all_reduce(x, torch.distributed.ReduceOp.SUM, 0)
            z = torch.distributed.all_reduce(x, torch.distributed.ReduceOp.AVG, 0)
            return z
        
    initialize_pjrt()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.distributed.init_process_group(world_size=1, rank=0)
    model = Basic()
    prog = export(model, (torch.rand(20, 10), ))
    options = StableHLOExportOptions()
    options.custom_ops_allowed_in_graph.add("tt_custom_shlo_ops")
    shlo = exported_program_to_stablehlo(prog, options)
    print(shlo.get_stablehlo_text())



if __name__ == "__main__":
    test()
