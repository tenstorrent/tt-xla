# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from .device_runner import run_on_cpu


@run_on_cpu
def random_tensor(shape: tuple, dtype=jnp.float32, random_seed: int = 0) -> jax.Array:
    """Generates random tensor of `shape` and `dtype` on CPU."""
    prng_key = jax.random.key(random_seed)
    return jax.random.uniform(key=prng_key, shape=shape, dtype=dtype)


def compare_pcc(
    device_output: jax.Array, golden_output: jax.Array, required_pcc: float = 0.99
) -> bool:
    # If tensors are really close, pcc will be nan. Handle that before calculating pcc.
    if compare_allclose(device_output, golden_output, 1e-3, 1e-3):
        return True

    pcc = jnp.corrcoef(device_output.flatten(), golden_output.flatten())
    return jnp.min(pcc) >= required_pcc


def compare_atol(
    device_output: jax.Array, golden_output: jax.Array, required_atol: float = 1e-2
) -> bool:
    atol = jnp.max(jnp.abs(device_output - golden_output))
    return atol <= required_atol


def compare_equal(device_output: jax.Array, golden_output: jax.Array) -> bool:
    return (device_output == golden_output).all()


def compare_allclose(
    device_output: jax.Array, golden_output: jax.Array, rtol=1e-2, atol=1e-2
) -> bool:
    allclose = jnp.allclose(device_output, golden_output, rtol=rtol, atol=atol)
    return allclose
