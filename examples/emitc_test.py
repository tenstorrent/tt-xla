# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import jax
import os
import sys
import jax._src.xla_bridge as xb
import io
import pickle
from jax.experimental import serialize_executable
import ttxla_tools
import tt_alchemist


def example_computation(x):
    return jnp.sum(x**2)


with jax.default_device(jax.devices("cpu")[0]):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (32, 32))

ttxla_tools.serialize_function(example_computation, "model/model", x)
tt_alchemist.generate_cpp(
    input_file="model/model_ttir.mlir",
    output_dir="model/cpp",
    local=True,
)
