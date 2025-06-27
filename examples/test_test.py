# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import os
from tests.infra.utils import serialize_function_to_binary


@jax.jit
def my_func(x):
    return x**2 + 1

# print("Result:", result)
x = jnp.array([1.0, 2.0, 3.0])

flatbuffer_binary = serialize_function_to_binary(my_func, x)

with open("compiled_executable_3.ttnn", "wb") as f:
    f.write(flatbuffer_binary)