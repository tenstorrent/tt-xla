# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp
from tt_jax import serialize_function_to_disk

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])


def add(x, y):
    return x + y


serialize_function_to_disk("output/add", add, a, b)
