# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import flax.nnx as nnx
import jax
import jax.numpy as jnp

"""
Demonstrates how to hook into compile options to use Codegen, from Jax
"""


class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.A = nnx.Linear(32, 128, rngs=rngs)
        self.B = nnx.Linear(128, 64, rngs=rngs)

    def __call__(self, x):
        x = self.A(x)
        x = nnx.relu(x)
        x = self.B(x)
        x = nnx.tanh(x)
        return jnp.sum(x**2)


# Initialize model on CPU.
with jax.default_device(jax.devices("cpu")[0]):
    model = Model(rngs=nnx.Rngs(0))
    key = jax.random.key(1)
    x = jax.random.normal(key, (32, 32))
    graphdef, state = nnx.split(model)


# Define forward pass.
def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)


# Set up compile options to trigger code generation.
options = {
    # Code generation options
    "backend": "codegen_py",
    # Optimizer options
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # Tensor dumping options
    # "dump_inputs": True,
    "export_path": "model",
}

# Compile the model. Make sure to pass the code generation options.
fun = jax.jit(
    forward,
    compiler_options=options,
)

# Run the model. This triggers code generation.
fun(graphdef, state, x)
