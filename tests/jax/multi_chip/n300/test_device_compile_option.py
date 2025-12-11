# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest


def eltwise_add(x, y):
    return x + y


@pytest.mark.push
@pytest.mark.nightly
def test_jit_with_device_option():
    """
    Test jax.jit with device= option to target a specific TT device.

    This test verifies that compilation and execution work correctly when
    explicitly specifying a device (tt_devices[1]) via jax.jit's device parameter.
    The same function is run on both CPU and TT device to verify correctness.
    """
    tt_devices = jax.devices("tt")

    # Create inputs on CPU
    with jax.default_device(jax.devices("cpu")[0]):
        prng_key = jax.random.PRNGKey(0)
        x = jax.random.uniform(prng_key, shape=(32, 32))
        prng_key, subkey = jax.random.split(prng_key)
        y = jax.random.uniform(subkey, shape=(32, 32))

    # Run on CPU
    cpu_compiled_fn = jax.jit(eltwise_add, device=jax.devices("cpu")[0])
    cpu_result = cpu_compiled_fn(x, y)

    # Run on TT device
    x_tt = jax.device_put(x, tt_devices[1])
    y_tt = jax.device_put(y, tt_devices[1])
    tt_compiled_fn = jax.jit(eltwise_add, device=tt_devices[1])
    tt_result = tt_compiled_fn(x_tt, y_tt)

    # Verify TT result is on the correct device
    assert tt_result.devices() == {tt_devices[1]}

    # Compare results
    tt_result_cpu = jax.device_put(tt_result, jax.devices("cpu")[0])
    assert jnp.allclose(tt_result_cpu, cpu_result, rtol=1e-2, atol=1e-2)
