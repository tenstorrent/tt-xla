# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import jax
import os
import sys
import jax._src.xla_bridge as xb


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


def add_function(x, y):
    """Simple function to add two arrays."""
    return x + y


# Explicitly JIT compile the function
jit_add = jax.jit(add_function, compiler_options={"codegen_cpp": "True"})

if __name__ == "__main__":
    # Create some test data
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])

    # Call the JIT-compiled function
    result = jit_add(a, b)

    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result: {result}")
