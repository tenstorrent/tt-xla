import os
import torch
from torch import nn
import torch
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo


def test():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

    model = Basic()
    prog = export(model, (torch.rand(20, 10), ))
    shlo = exported_program_to_stablehlo(prog)
    print(shlo.get_stablehlo_text())

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
torch.distributed.init_process_group(world_size=1, rank=0)


if __name__ == "__main__":
    test()
