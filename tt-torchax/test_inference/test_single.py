import torchax
from torchax.tensor import j2t, t2j
from torchax.interop import JittableModule
import tt_xla.utils as ttxla

import torch
from torch import nn
from torch.nn import functional as F

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.linear(x)


def main(m, input):
  torchax.enable_globally()
  # Move to JAX
  input = input.to('jax')
  m = m.to('jax')

  # Eager mode execution
  print("\n\n\n---- eager mode ----")
  output_eager = m(input)

  # Jit mode execution
  # m_jitted = JittableModule(m)
  # output_jit = m_jitted(input)

  # Compiled mode execution
  print("\n\n\n---- compiled mode ----")
  m_compiled = torchax.compile(m)
  output_compiled = m_compiled(input)

  # Restore to torch Tensor
  output_eager = j2t(output_eager)
  # output_jit = j2t(output_jit)
  output_compiled = j2t(output_compiled)
  torchax.disable_globally()

  print('<Eager Mode Result>')
  print(output_eager)
  print('\n')
  
  # print('<JIT Mode Result>')
  # print(output_jit)
  # print('\n')

  print('<Compiled Mode Result>')
  print(output_compiled)
  print('\n')

if __name__ == "__main__":
  # Initialize torchax and tt_xla
  ttxla.initialize()
  main(ToyModel(), torch.randn(1024, 1024))
