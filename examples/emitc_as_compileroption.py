# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp


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
