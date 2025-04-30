
import torchax
from torchax.tensor import j2t, t2j
from torchax.interop import JittableModule
import tt_xla.utils as ttxla_utils

import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = x.view(-1, 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.linear(x)


def main(m, input):
  # Initialize torchax and tt_xla
  torchax.enable_globally()
  ttxla_utils.initialize()

  # Move to JAX
  input = input.to('jax')
  m = m.to('jax')

  # Eager mode execution
  output_eager = m(input)

  # Jit mode execution
  # m_jitted = JittableModule(m)
  # output_jit = m_jitted(input)

  # Compiled mode execution
  m_compiled = torchax.compile(m)
  output_compiled = m_compiled(input)

  # Restore to torch Tensor
  output_eager = j2t(output_eager)
  # output_jit = j2t(output_jit)
  output_compiled = j2t(output_compiled)

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
  main(ToyModel(), torch.randn(1024, 1024))
  #main(MyModel(), torch.randn(3, 3, 28, 28))
