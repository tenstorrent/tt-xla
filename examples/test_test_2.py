# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from jax import grad, jit, vmap
import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import jax
import os
import sys
import jax._src.xla_bridge as xb


def register_pjrt_plugin():
    """Registers TT PJRT plugin."""

    plugin_path = os.path.join(
        os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
    )
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Could not find TT PJRT plugin at {plugin_path}")

    xb.register_plugin("tt", library_path=plugin_path)
    jax.config.update("jax_platforms", "tt,cpu")


register_pjrt_plugin()


def my_func(x):
    return x**2 + 1


x = jnp.array([1.0, 2.0, 3.0])

jitted_func = jax.jit(my_func, compiler_options={"aaaaa": "bbbbb", "cccccc": "dddddd"})

print(jitted_func(x))
