
import torchax
from torchax.tensor import j2t, t2j
import tt_xla.utils as ttxla
from tt_xla.dist import DistributedDataParallel as DDP

import torch
import alexnet_utils

def alexnet_ddp_spmd(): 
    m = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    m.eval()
    input = alexnet_utils.load_input(8)

    ##############################################
    # Initialize torchax and tt-xla
    torchax.enable_globally()
    ttxla.initialize(use_shardy=True, backend="cpu,tt")
    
    # torch-scripted DDP model and compute using CPU
    m_scripted = torch.jit.script(m)
    #output_cpu = m_scripted(input)
    #utils.print_output(output_cpu)
    #return

    # torchax DDP model using JAX
    ddp_model = DDP(m_scripted)
    sharded_input = ddp_model.shard_input(input)

    # Use jit_step for jax.jit, otherwise it will run with eager mode
    @ddp_model.jit_step
    def step(input):
       output_ddp = ddp_model(input)
       return output_ddp

    output_ddp = j2t(step(sharded_input)) # produce jax outputs
    ##############################################

    alexnet_utils.print_output(output_ddp)

if __name__ == "__main__":
    alexnet_ddp_spmd()
