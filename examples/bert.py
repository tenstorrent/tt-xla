import torchax
import torch
import torch.nn as nn
import torch.distributed as dist

import os
import jax
import jax._src.xla_bridge as xb
from jax.experimental import mesh_utils
from torchax.interop import JittableModule

from transformers import BertModel
from torch.utils import _pytree as pytree
from torchax.interop import torch_view, jax_view


def initializePJRT():
  path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
  if not os.path.exists(path):
    raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}, have you compiled the project?")
  plugin = xb.register_plugin('tt', priority=10, library_path=path, options=None)
  jax.config.update("jax_platforms", "tt,cpu")
  # jax.config.update("jax_use_shardy_partitioner", True)

xla_env = torchax.enable_globally()

class CompiledModule:

    def __init__(self, model):
        weights = model.state_dict()
        weights.update(model.named_parameters())
        self._weights = pytree.tree_map_only(torch.Tensor, torchax.tensor.t2j, weights)
        self._model = model

        self._func_jitted_torch = None #torch_view(func_mod_jitted)


    def _maybe_move_tensor(self, tensor):
        if isinstance(tensor, torch.Tensor) and not isinstance(tensor, torchax.tensor.Tensor):
            return torchax.tensor.t2j(tensor)
        return tensor

    def _make_jitted(self, args, kwargs):
        static = []
        for i, a in enumerate(args):
            if not isinstance(a, torch.Tensor):
                static.append(i + 1)  # weight is 0
        static_argnames = []
        for k, v in kwargs.items():
            if not isinstance(v, torch.Tensor):
                static_argnames.append(k)

        def f(weights, *args, **kwargs):
            weights, args, kwargs = torchax.tensor.wrap((weights, args, kwargs))
            with torchax.functions.XLAFunctionMode(), torchax.tensor.XLADispatchMode():
                res = torch.func.functional_call(self._model, weights, args, kwargs)
                if isinstance(res, tuple) and len(res) == 1:
                    res = res[0]
            return torchax.tensor.unwrap(res)

        fjit = jax.jit(f, static_argnames=tuple(static_argnames))
        return torch_view(fjit)


    def forward(self, *args, **kwargs):
        (args, kwargs) = pytree.tree_map(self._maybe_move_tensor, (args, kwargs))
        if self._func_jitted_torch is None:
            self._func_jitted_torch = self._make_jitted(args, kwargs)
        return self._func_jitted_torch(
            self._weights, 
            *args,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, key):
        return getattr(self._model, key)


def test():
  initializePJRT()
  m = BertModel.from_pretrained("prajjwal1/bert-tiny")
  inputs = torch.randint(20000, (1, 128), device='cpu')

  device_tt = jax.devices('tt')
  print("device:: ", device_tt)

  # Execute this model using torch
  breakpoint()
  weights, jax_func = torchax.extract_jax(m)
  inputs_jax = pytree.tree_map_only(
      torch.Tensor, torchax.tensor.t2j, inputs)

  res = jax.jit(jax_func)(weights, inputs_jax)

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
