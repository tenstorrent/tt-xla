# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from tools import serialize_function_to_binary
import os
import jax._src.xla_bridge as xb
import sys

# Register cpu and tt plugin. tt plugin is registered with higher priority; so
# program will execute on tt device if not specified otherwise.
def initialize():
    path = os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")

initialize()

def my_func(x):
    return x**2 + 1

# print("Result:", result)
x = jnp.array([1.0, 2.0, 3.0])

serialize_function_to_binary(my_func, "compiled_executable_3.ttnn", x)