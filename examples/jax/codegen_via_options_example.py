# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.A = nnx.Linear(32, 128, rngs=rngs)
        self.B = nnx.Linear(128, 64, rngs=rngs)

    def __call__(self, x):
        x = self.A(x)
        x = self.B(x)
        x = nnx.tanh(x)
        return jnp.sum(x**2)


with jax.default_device(jax.devices("cpu")[0]):
    model = Model(rngs=nnx.Rngs(0))
    key = jax.random.key(1)
    x = jax.random.normal(key, (32, 32))
    graphdef, state = nnx.split(model)


def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)


fun = jax.jit(
    forward,
    compiler_options={"backend": "codegen_py", "export_path": "jax_codegen_example"},
)
fun(graphdef, state, x)
