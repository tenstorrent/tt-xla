import logging
from typing import Optional

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from functools import partial, wraps

import torch
import torch.distributed._functional_collectives
import torch.utils._pytree as torch_pytree

import torchax
from torchax import interop
from torchax.tensor import j2t, t2j

import tt_xla.utils as ttxla

class TorchAXDDPInference(torch.nn.Module):
    """Re-implementation of TorchAXSingleInference using JAX SPMD.
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
            # Use jit_step for jax.jit
            @self.jit_step
            def step(*args):
               output_ddp = self._module(*args)
               return output_ddp

            torchax.enable_globally()
            output = j2t(step(*args)) # produce jax outputs
            torchax.disable_globally()
            return output

class TorchAXSingleInference(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        **kwargs,
    ):
        super().__init__()
        self._module = module

    def forward(self, *args):
        if isinstance(self._module, TorchAXDDPInference):
            raise ValueError("model must not be an instance of TorchAXDDPInference")

        torchax.enable_globally()
        jax_args = []
        for arg in args:
            jax_args.append(arg.to('jax'))
        model = self._module.to('jax')
 
        m_compiled = torchax.compile(model)
        output = j2t(m_compiled(*jax_args))
        torchax.disable_globally()
        return output
