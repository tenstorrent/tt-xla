import torchax
import torch
import torch.nn as nn
import torch.distributed as dist

import os
import sys
import jax
import jax._src.xla_bridge as xb
from jax.experimental import mesh_utils


def initializePJRT():
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")
  # jax.config.update("jax_use_shardy_partitioner", True)

# xla_env = torchax.enable_globally()

def test():
  initializePJRT()
  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(32, 64)

      def forward(self, x):
        return self.fc1(x)

  breakpoint()
  device_tt = jax.devices('tt')
  print("device:: ", device_tt)

  # Execute this model using torch
  m = MyModel()
  m = m.to('jax')

  input = torch.randn(32, 32, dtype=torch.float32, device='cpu')
  input = input.to('jax')
  res = m(input)

  print(res)
  print(type(res))


def init_process_group():
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  dist.init_process_group(backend="jax", init_method="jax://")

  group_ranks = [0]
  return group_ranks

def deinit_process_group():
  dist.destroy_process_group()

def test_mp():
  initializePJRT()
  breakpoint()
  group_ranks = init_process_group()

  def f(index):
    torch.distributed.all_reduce(index)
    return index

  res = torchax.distributed.spawn(f)

  print(res)
  print(type(res))
  deinit_process_group()

if __name__ == "__main__":
    test()
