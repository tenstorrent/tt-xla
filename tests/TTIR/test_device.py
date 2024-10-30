# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from infrastructure import random_input_tensor


def test_num_devices():
    devices = jax.devices()
    assert len(devices) == 1


def test_to_device():
    cpu_array = random_input_tensor((32, 32))
    device = jax.devices()[0]
    tt_array = jax.device_put(cpu_array, device)
    assert tt_array.device.device_kind == "wormhole"


def test_input_on_device():
    def module_add(a, b):
        return a + b

    tt_device = jax.devices()[0]
    cpu_param = random_input_tensor((1000, 32, 32))
    tt_param = jax.device_put(cpu_param, tt_device)

    graph = jax.jit(module_add)
    cpu_activation_0 = random_input_tensor((1000, 32, 32))
    cpu_activation_1 = random_input_tensor((1000, 32, 32))
    tt_activation_0 = jax.device_put(cpu_activation_0, tt_device)
    tt_activation_1 = jax.device_put(cpu_activation_1, tt_device)

    res0 = graph(tt_activation_0, tt_param)
    res1 = graph(tt_activation_1, tt_param)
    res0_host = jax.device_put(res0, cpu_activation_0.device)
    res2 = graph(res0, res1)

    res0_cpu = graph(cpu_activation_0, cpu_param)
    res1_cpu = graph(cpu_activation_1, cpu_param)
    res2_cpu = graph(res0_cpu, res1_cpu)

    res0_host = jax.device_put(res0, res0_cpu.device)
    res1_host = jax.device_put(res1, res1_cpu.device)
    res2_host = jax.device_put(res2, res2_cpu.device)

    assert jnp.allclose(res0_host, res0_cpu, atol=1e-2)
    assert jnp.allclose(res1_host, res1_cpu, atol=1e-2)
    assert jnp.allclose(res2_host, res2_cpu, atol=1e-2)
