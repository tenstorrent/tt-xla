# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into serialization to use Codegen, from Jax
You should strongly prefer using codegen via compile options
But for completeness we show how to do it via serialization too
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tt_alchemist
from tt_jax import serialize_compiled_artifacts_to_disk


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


# Serialize the compiled artifacts to disk.
serialize_compiled_artifacts_to_disk(
    forward, graphdef, state, x, output_prefix="model/model"
)

# Generate C++ code from the TTIR.
tt_alchemist.generate_cpp(
    input_file="model/model_ttir.mlir",
    output_dir="model/cpp",
    local=False,
)

# Generate Python code from the TTIR.
tt_alchemist.generate_python(
    input_file="model/model_ttir.mlir",
    output_dir="model/py",
    local=False,
)
