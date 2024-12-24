# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import random_tensor
from infra.device_connector import DeviceConnector

# ---------- Fixtures ----------


@pytest.fixture(autouse=True)
def initialize_device():
    """
    Initialize available devices by registering PJRT plugin.

    PJRT plugin is registered during DeviceConnector instantiation. By providing global
    singleton instance of the connector in device_connector.py we solved the problem of
    automatic plugin registration.

    We can simulate it here by calling the constructor directly.

    NOTE we cannot manually do `xla_bridge.register_plugin` here because it will
    report `PJRT_Api already exists for device type tt`.
    """
    DeviceConnector()


@pytest.fixture
def cpu() -> jax.Device:
    return jax.devices("cpu")[0]


@pytest.fixture
def tt_device() -> jax.Device:
    return jax.devices("tt")[0]


def is_cpu(device: jax.Device) -> bool:
    return repr(device).startswith("CpuDevice")


def is_tt_device(device: jax.Device) -> bool:
    return repr(device).startswith("TTDevice")


# ---------- Tests ----------


def test_devices_are_connected():
    cpus = jax.devices("cpu")

    assert len(cpus) > 0, f"Expected at least one CPU to be connected"
    assert is_cpu(cpus[0])

    tt_devices = jax.devices("tt")

    assert len(tt_devices) > 0, f"Expected at least one TT device to be connected"
    assert is_tt_device(tt_devices[0])


def test_put_tensor_on_device(cpu: jax.Device, tt_device: jax.Device):
    # `random_tensor` is executed on cpu due to `@run_on_cpu` decorator so we don't have
    # to put it explicitly on cpu, but we will just for demonstration purposes.
    x = random_tensor((32, 32))

    x = jax.device_put(x, cpu)
    assert is_cpu(x.device)

    x = jax.device_put(x, tt_device)
    assert is_tt_device(x.device)


def test_device_output_comparison(cpu: jax.Device, tt_device: jax.Device):
    @jax.jit  # Apply jit to this function.
    def add(x: jax.Array, y: jax.Array):
        return x + y

    x, y = random_tensor((32, 32)), random_tensor((32, 32))

    cpu_x = jax.device_put(x, cpu)
    cpu_y = jax.device_put(y, cpu)
    cpu_res = add(cpu_x, cpu_y)
    assert is_cpu(cpu_res.device)

    tt_device_x = jax.device_put(x, tt_device)
    tt_device_y = jax.device_put(y, tt_device)
    tt_device_res = add(tt_device_x, tt_device_y)
    assert is_tt_device(tt_device_res.device)

    # We want to do comparison on CPU, so we put tt_device_res on CPU before that.
    tt_device_res = jax.device_put(tt_device_res, cpu)

    assert jax.numpy.allclose(cpu_res, tt_device_res, atol=1e-2)
