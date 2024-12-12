# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import nnx

# Convenience alias. Could be used to represent jax.Array, torch.Tensor, etc.
Tensor = jax.Array

# Convenience alias. Could be used to represent nnx.Module, torch.nn.Module, etc.
Model = nnx.Module


def random_tensor(shape: tuple, dtype=jnp.float32, random_seed: int = 0) -> jax.Array:
    """Generates random tensor of `shape` and `dtype` on CPU."""
    prng_key = jax.random.key(random_seed)
    return jax.random.uniform(key=prng_key, shape=shape, dtype=dtype)
