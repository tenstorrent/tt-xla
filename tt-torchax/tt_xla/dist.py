import os
import logging
import copy
from typing import List, Optional, Union, Tuple

import jax
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial, wraps

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives
import torch.utils._pytree as torch_pytree
from torch.nn.utils import stateless as torch_stateless

import torchax
from torchax import interop
from torchax.tensor import j2t, t2j

import tt_xla.utils as ttxla

def init_process_group():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="jax", init_method="jax://")

def destroy_process_group():
    dist.destroy_process_group()

def spawn(f, args=(), env: Optional[torchax.tensor.Environment] = None):
    """Wrap `f` in a JAX `shard_map` with the axis name `torch_dist` defined.
    `f` is expected to take the replica index as a positional argument, similar
    to `torch.multiprocessing.spawn`.
    Note: `spawn` does not actually create parallel processes.
    """
    env = env or torchax.default_env()
    #env.config.debug_print_each_op = True
    #env.config.debug_mixed_tensor = True
  
    # Create mesh device with 1xN and sharding across N devices
    mesh, device_count, device = ttxla.open_device(axis=["x", "torch_dist"])
    
    in_spec = P('torch_dist', )
    out_spec = P('torch_dist', )
  
    index = torch.arange(device_count)
    in_sharding = NamedSharding(mesh, in_spec)
    jax_args = [jax.device_put(t2j(index), in_sharding)]
    in_specs = [in_spec]
    in_shardings = [in_sharding]
  
    for arg in args:
        jax_args.append(jax.device_put(t2j(arg), in_sharding))
        in_specs.append(in_spec)
        in_shardings.append(in_sharding)
  
    out_shardings = NamedSharding(mesh, out_spec)
  
    @partial(shard_map, mesh=mesh, in_specs=tuple(in_specs), out_specs=out_spec)
    def jax_wrapper(*jax_args):
        args = env.j2t_iso(jax_args)
        torch_outputs = f(*args)
        return env.t2j_iso(torch_outputs)
  
    jax_outputs = jax.jit(jax_wrapper, in_shardings=tuple(in_shardings), out_shardings=out_shardings)(*jax_args)
  
    return env.j2t_iso(jax_outputs)

class DistributedDataParallel(torch.nn.Module):
    """Re-implementation of DistributedDataParallel using JAX SPMD.
  
    Splits inputs along batch dimension (assumed to be 0) across all devices in
    JAX runtime, including remote devices. Each process should load a distinct
    shard of the input data using e.g. DistributedSampler. Each process' shard
    is then further split among the addressable devices (e.g. local TPU chips)
    by `shard_input`.
  
    Note: since parameters are replicated across addressable devices, inputs
    must also be SPMD sharded using `shard_input` or `replicate_input`.
  
    Example usage:
  
    ```
    jax_model = torchax.distributed.DistributedDataParallel(create_model())
    for data, dataloader:
      jax_data = jax_model.shard_input(data)
      jax_output = jax_model(jax_data)
    ```
    """
    def __init__(
        self,
        module: torch.nn.Module,
        env: Optional[torchax.tensor.Environment] = None,
        **kwargs,
    ):
        if kwargs:
          logging.warning(f"Unsupported kwargs {kwargs}")
  
        super().__init__()
        self._env = env or torchax.default_env()
        #self._env.config.debug_print_each_op = True
        #self._env.config.debug_mixed_tensor = True
  
        # Create mesh device with 1xN and sharding across N devices
        self._mesh, self._device_count, self._device = ttxla.open_device(axis=["x", "batch"])
        self._local_device_count = self._device_count
  
        replicated_state = torch_pytree.tree_map_only(
          torch.Tensor,
          lambda t: self._env.j2t_iso(
            jax.device_put(
              self._env.to_xla(t)._elem, NamedSharding(self._mesh, P())
            )
          ),
          module.state_dict(),
        )
  
        self._out_shardings = []
        self._out_shardings.append(NamedSharding(self._mesh, P()))
        self._out_shardings.append(NamedSharding(self._mesh, P('batch', )))
      
        # TODO: broadcast
        module.load_state_dict(replicated_state, assign=True)
        self._module = module
    
    def shard_input(self, inp):
        per_process_batch_size = inp.shape[0]  # assumes batch dim is 0
        per_replica_batch_size = per_process_batch_size // self._local_device_count
        per_replica_batches = torch.chunk(inp, self._local_device_count)
        global_batch_size = per_replica_batch_size * self._device_count
        global_batch_shape = (global_batch_size,) + inp.shape[1:]
  
        sharding = NamedSharding(self._mesh, P("batch",))
        return self._env.j2t_iso(jax.make_array_from_single_device_arrays(
          global_batch_shape,
          NamedSharding(self._mesh, P("batch")),
          arrays=[
            jax.device_put(self._env.to_xla(batch)._elem, device)
            for batch, device in zip(
              per_replica_batches, sharding.addressable_devices
            )
          ],
        ))
  
    def replicate_input(self, inp):
        return self._env.j2t_iso(
          jax.device_put(inp._elem, NamedSharding(self._mesh, P()))
        )
  
    def jit_step(self, func):
        @partial(interop.jax_jit,
                 kwargs_for_jax_jit={'donate_argnums': 0, 
                                     'out_shardings': tuple(self._out_shardings)})
        def _jit_fn(states, args):
            self.load_state_dict(states)
            outputs = func(*args)
            return self.state_dict(), outputs
  
        @wraps(func)
        def inner(*args):
            jax_states = self.state_dict()
            new_states, outputs = _jit_fn(jax_states, args)
            self.load_state_dict(new_states)
            return outputs
  
        return inner
    
    def forward(self, *args):
        with self._env:
            return self._module(*args)

class FSDPv2(torch.nn.Module):
    """
    FSDPv2 is an implementation of Fully-sharded Data Parallel for inference.
    To implement this, we need 2 things:
      1. Shard inputs on batch dimension (i.e. like DDP).
      2. Shard all the weights in the first dimension.
    """
    def __init__(self, mod):
        super().__init__()

        _axis: Tuple[str, str] = ('fsdp', )
        # Create mesh device with 1xN and sharding across N devices
        self.mesh, self.device_count, self.device = ttxla.open_device(axis=["x", _axis[0]])
        self.local_device_count = self.device_count
        self.in_spec = P(*_axis)
        self.out_spec = self.in_spec
        #self.x_sharding = jax.sharding.NamedSharding(self.mesh, P(_axis))
        self.y_sharding = jax.sharding.NamedSharding(self.mesh, self.in_spec)
        self.replicated = jax.sharding.NamedSharding(self.mesh, P())
        self.sharding = self.y_sharding
        self.mod = mod

    def forward(self, *args):
        args = list(args)
        args[0] = self.shard(args[0])
        res = self.mod(*args)
        return self.shard(res)
  
    def shard(self, x):
        return torchax.interop.call_jax(
            jax.lax.with_sharding_constraint,
            x,
            self.sharding,
        )

    def shard_fsdp_style(self, state_dict, sharding=None):
        if sharding is None:
            sharding = self.y_sharding
        def move_one_tensor(x):
            jval = torchax.tensor.t2j(x)
            return self.sharded_device_put(jval, sharding)

        if isinstance(state_dict, torch.Tensor):
            return move_one_tensor(state_dict)
        res = {}
        for k, v in sorted(state_dict.items()):
            res[k] = move_one_tensor(v)
        return res

    def sharded_device_put(self, tensor, sharding):
        if isinstance(tensor, tuple):
            return tuple(self._sharded_device_put(t, sharding) for t in tensor)

        if self.device_count == self.local_device_count:
            return jax.device_put(tensor, sharding)

        shape = tensor.shape
        x_split = [jax.device_put(tensor[i], device) for device, i in sharding.addressable_devices_indices_map(shape).items()]
        return jax.make_array_from_single_device_arrays(shape, sharding, x_split)

class JittableModule(torch.nn.Module):

    def __init__(self, m: torch.nn.Module, extra_jit_args={}):
        super().__init__()
        self.params, self.buffers = interop.extract_all_buffers(m)
        self._model = m
        self._jitted = {}
        self._extra_jit_args = extra_jit_args

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def functional_call(
            self, method_name, params, buffers, *args, **kwargs):
        kwargs = kwargs or {}
        params_copy = copy.copy(params)
        params_copy.update(buffers)
        with torch_stateless._reparametrize_module(self._model, params_copy):
            res = getattr(self._model, method_name)(*args, **kwargs)
        return res

    def _functional_call(
            self, method_name, params, buffers, *args, **kwargs):
        kwargs = kwargs or {}
        params_copy = copy.copy(params)
        params_copy.update(buffers)
        with torch_stateless._reparametrize_module(self._model, params_copy):
            res = getattr(self._model, method_name)(*args, **kwargs)
        return res

    def forward(self, *args, **kwargs):
        if 'forward' not in self._jitted:
            jitted = jax_jit(
                partial(self.functional_call, 'forward'),
                kwargs_for_jax_jit=self._extra_jit_args,
            )
            def jitted_forward(*args, **kwargs):
                return jitted(self.params, self.buffers, *args, **kwargs)
            self._jitted['forward'] = jitted_forward
        return self._jitted['forward'](*args, **kwargs)

    def __getattr__(self, key):
        if key == '_model':
            return super().__getattr__(key)
        if key in self._jitted:
            return self._jitted[key]
        return getattr(self._model, key)

    def make_jitted(self, key):
        jitted = jax_jit(
            partial(self.functional_call, key), 
            kwargs_for_jax_jit=self._extra_jit_args)
        def call(*args, **kwargs):
            return jitted(self.params, self.buffers, *args, **kwargs)
        self._jitted[key] = call
