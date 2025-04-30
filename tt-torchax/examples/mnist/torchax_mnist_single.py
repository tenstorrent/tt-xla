
import torchax
from torchax.tensor import j2t, t2j
import tt_xla.utils as ttxla

import torch
from mnist_utils import MNISTModel, TestLoader

def mnist_single_main():
    test_loader = TestLoader()

    inputs = []
    with torch.no_grad():
        for data, target in test_loader:
            inputs.append(data)
    input = inputs[0]

    m = MNISTModel()
    m.eval()

    with torch.no_grad():
        output_cpu = m(input)

    ##############################################
    # Initialize torchax and tt_xla
    torchax.enable_globally()
    ttxla.initialize()
  
    # Move to JAX
    input = input.to('jax')
    m = m.to('jax')
  
    # Compiled mode execution
    m_compiled = torchax.compile(m)
    output_xla = j2t(m_compiled(input))

    print("---- output cpu ----") 
    print(output_cpu)
    print("---- output tt-xla ----") 
    print(output_xla)
    ##############################################

if __name__ == "__main__":
    mnist_single_main()
