
import tt_xla.utils as ttxla
from tt_xla.tt_torchax import TorchAXDDPInference

import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)
    def forward(self, x):
        return self.linear(x)

def ddp_spmd(m, input):
  
    m_scripted = torch.jit.script(m)
    with torch.no_grad():
        output_cpu = m_scripted(input)

    ddp_model = TorchAXDDPInference(m_scripted)
    sharded_input = ddp_model.shard_input(input)
    output_ddp = ddp_model(sharded_input)

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output tt-xla ----") 
    print(output_ddp)

if __name__ == "__main__":
    ttxla.initialize()
    ddp_spmd(ToyModel(), torch.randn(1024, 1024))
