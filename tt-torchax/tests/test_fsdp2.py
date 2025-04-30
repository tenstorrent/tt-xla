import torchax
from torchax.tensor import j2t, t2j
from torchax import interop
from torchax.interop import jax_view, torch_view
import tt_xla.utils as ttxla
from tt_xla.dist import FSDPv2, JittableModule
from functools import partial, wraps
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map


import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(1024, 1024)
  
    def forward(self, x):
        return self.linear(x)

def main_fsdp2_spmd(m, input):
    # Initialize torchax and tt_xla
    torchax.enable_globally()
    xla_env = torchax.default_env()
    ttxla.initialize(use_shardy=True, backend="cpu,tt")

    # torch-scripted DDP model and compute using CPU
    m_scripted = torch.jit.script(m)
    with torch.no_grad():
        output_cpu = m_scripted(input)

    fsdp2_model = FSDPv2(m)
    fsdp2_model = JittableModule(fsdp2_model)

    jax_params = fsdp2_model.shard_fsdp_style(fsdp2_model.params)
    jax_buffers = fsdp2_model.shard_fsdp_style(fsdp2_model.buffers)
    jax_inputs = fsdp2_model.sharded_device_put(jax_view(xla_env.to_xla(input)), fsdp2_model.y_sharding)

    def loss(jax_weights, jax_buffers, jax_inputs):
        jax_weights = jax.lax.with_sharding_constraint(jax_weights, fsdp2_model.replicated)  # fsdpv2
        weights, buffers, inputs = torch_view((jax_weights, jax_buffers, jax_inputs))
        output_fsdp2 = fsdp2_model.functional_call("forward", weights, buffers, inputs)
        return jax_view(output_fsdp2)

    # Use jit_step for jax.jit, otherwise it will run with eager mode
    @partial(
        jax.jit,
        donate_argnums=(0, 1),
        out_shardings=(fsdp2_model.y_sharding),
    )
    def step(jax_weights, jax_buffers, jax_inputs):
        return loss(jax_weights, jax_buffers, jax_inputs)

    step_lowered = step.lower(jax_params, jax_buffers, jax.ShapeDtypeStruct(jax_inputs.shape, jnp.dtype('float32'), sharding=fsdp2_model.y_sharding))
    print(step_lowered.as_text())
    #step_compiled = step_lowered.compile()
    #output_fsdp2 = step_compiled(jax_params, jax_buffers, jax_inputs)
    
    #output_fsdp2 = step(jax_params, jax_buffers, jax_inputs)

    #print(output_cpu)
    #print(output_fsdp2)

if __name__ == "__main__":
    main_fsdp2_spmd(ToyModel(), torch.randn(1024, 1024))
