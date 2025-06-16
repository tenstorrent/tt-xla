# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

from jax import grad, jit, vmap
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import jax
import os
import sys
import jax._src.xla_bridge as xb

# Register cpu and tt plugin. tt plugin is registered with higher priority; so
# program will execute on tt device if not specified otherwise.
def initialize():
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


initialize()

with jax.default_device(jax.devices("cpu")[0]):
    t1 = jax.random.normal(jax.random.PRNGKey(0), (2, 2))

real_gelu = jax.nn.gelu


def mygelu(x, approximate=True):
    if approximate:
        return jax.lax.composite(
            lambda x: real_gelu(x, approximate=True),
            "tt.gelu_tanh",
        )(x)
    else:
        return jax.lax.composite(
            lambda x: real_gelu(x, approximate=False),
            "tt.gelu",
        )(x)


jax.nn.gelu = mygelu


@jit
def composite_fn(x):
    return jax.nn.gelu(x)


result = composite_fn(t1)
print("Result of composite gelu function:", result)
