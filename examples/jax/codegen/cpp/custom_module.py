# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from tt_jax import codegen_cpp

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


# Define forward pass.
def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)


def main():
    # Initialize model on CPU.
    with jax.default_device(jax.devices("cpu")[0]):
        model = Model(rngs=nnx.Rngs(0))
        key = jax.random.key(1)
        x = jax.random.normal(key, (32, 32))
        graphdef, state = nnx.split(model)

    # Any compile options you could specify when executing the model normally can also be used with codegen.
    extra_options = {
        # "optimization_level": 0,  # Levels 0, 1, and 2 are supported
    }

    codegen_cpp(
        forward, graphdef, state, x, export_path="model", compiler_options=extra_options
    )


def test_codegen_creates_model_folder():
    """Test that codegen_py creates the model folder and clean up afterwards."""
    model_dir = Path("model")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    try:
        main()
        assert model_dir.exists(), "Expected 'model' directory to be created"
    finally:
        if model_dir.exists():
            shutil.rmtree(model_dir)


if __name__ == "__main__":
    main()
