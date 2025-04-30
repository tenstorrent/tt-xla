
import torchax
from torchax.tensor import j2t, t2j
import tt_xla.utils as ttxla
from tt_xla.dist import DistributedDataParallel as DDP

import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.linear(x)

def main_ddp_spmd(m, input):
    # Initialize torchax and tt_xla
    torchax.enable_globally()
    ttxla.initialize(use_shardy=True, backend="cpu,tt")
  
    # torch-scripted DDP model and compute using CPU
    m_scripted = torch.jit.script(m)
    with torch.no_grad():
        output_cpu = m_scripted(input)
    
    # torchax DDP model using JAX
    ddp_model = DDP(m_scripted) # pytorch objects (replicated jax weights)
    sharded_input = ddp_model.shard_input(input) # pytorch objects (sharded jax inputs)
  
    # Use jit_step for jax.jit, otherwise it will run with eager mode
    @ddp_model.jit_step
    def ddp_jit(input):
       output_ddp = ddp_model(input)
       return output_ddp
  
    output_ddp = j2t(ddp_jit(sharded_input)) # produce jax outputs

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output tt-xla ----") 
    print(output_ddp)

if __name__ == "__main__":
    main_ddp_spmd(ToyModel(), torch.randn(1024, 1024))
