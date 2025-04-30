
import torchax
from torchax.tensor import j2t, t2j
import tt_xla.utils as ttxla

import torch
import alexnet_utils

def alexnet():
    m = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
    m.eval()
    input = alexnet_utils.load_input()
  
    ##############################################
    # Initialize torchax and tt_xla
    torchax.enable_globally()
    ttxla.initialize()
  
    # Move to JAX
    input = input.to('jax')
    m = m.to('jax')
  
    # Compiled mode execution
    m_compiled = torchax.compile(m)
    output = j2t(m_compiled(input))
    ##############################################
  
    alexnet_utils.print_output(output)

if __name__ == "__main__":
    alexnet()
