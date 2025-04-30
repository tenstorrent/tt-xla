
import torchax
from torchax.tensor import j2t, t2j
import tt_xla.utils as ttxla
from tt_xla.dist import DistributedDataParallel as DDP

import torch
from mnist_utils import MNISTModel, TestLoader

def mnist_ddp_spmd(): 
    test_loader = TestLoader(8)

    inputs = []
    with torch.no_grad():
        for data, target in test_loader:
            inputs.append(data)
    input = inputs[0]
    
    m = MNISTModel()
    m.eval()

    with torch.no_grad():
        output_cpu = m(input)
    
    m_scripted = torch.jit.script(m)

    ##############################################
    # Initialize torchax and tt-xla
    torchax.enable_globally()
    ttxla.initialize(use_shardy=True, backend="cpu,tt")

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

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output ddp tt-xla ----") 
    print(output_ddp)

if __name__ == "__main__":
    mnist_ddp_spmd()
