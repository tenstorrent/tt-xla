# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
from tests.infra.utils import serialize_function_to_binary

def my_func(x):
    return x**2 + 1

# Call once to trigger compilation
x = jnp.array([1.0, 2.0, 3.0])

binary = serialize_function_to_binary(my_func, x)

with open("compiled_executable_3.ttnn", "wb") as f:
    f.write(binary)

# print(f"Serialized binary written to compiled_executable.ttnn ({len(serialized)} bytes)")
