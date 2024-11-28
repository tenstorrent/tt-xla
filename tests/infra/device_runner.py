# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Sequence

import jax

from .device_connector import DeviceType, connector
from .test_module import TestModule


class DeviceRunner:
    @staticmethod
    def run_on_tt_device(module: TestModule) -> jax.Array:
        """Runs test module on TT device."""
        return DeviceRunner._run_on_device(module, DeviceType.TT)

    @staticmethod
    def run_on_cpu(module: TestModule) -> jax.Array:
        """Runs test module on CPU."""
        return DeviceRunner._run_on_device(module, DeviceType.CPU)

    @staticmethod
    def run_on_gpu(module: TestModule) -> jax.Array:
        """Runs test module on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def put_on_tt_device(*tensors: jax.Array) -> Sequence[jax.Array]:
        """Puts `tensors` on TT device."""
        return DeviceRunner._put_on_device(tensors, DeviceType.TT)

    @staticmethod
    def put_on_cpu(*tensors: jax.Array) -> Sequence[jax.Array]:
        """Puts `tensors` on CPU."""
        return DeviceRunner._put_on_device(tensors, DeviceType.CPU)

    @staticmethod
    def put_on_gpu(*tensors: jax.Array) -> Sequence[jax.Array]:
        """Puts `tensors` on GPU."""
        raise NotImplementedError("Support for GPUs not implemented")

    @staticmethod
    def _run_on_device(
        module: TestModule,
        device_type: DeviceType,
        raise_ex_if_jit_failed: bool = False,
    ) -> jax.Array:
        """
        Runs test module on device.

        If jitted graph fails to run, will try to run non-jitted graph if
        `raise_ex_if_jit_failed` is False.
        """
        device = connector.connect_device(device_type)

        # TODO is there a better way to check if function can be jitted than runtime fail?
        try:
            graph = module.get_jit_graph()
            inputs = DeviceRunner._put_on_device(module.get_inputs(), device_type)

            with jax.default_device(device):
                return graph(*inputs)

        except Exception as e:
            if raise_ex_if_jit_failed:
                raise e

            with jax.default_device(device):
                return module()

    @staticmethod
    def _put_on_device(
        tensors: Sequence[jax.Array], device_type: DeviceType
    ) -> Sequence[jax.Array]:
        """Puts `tensors` on device identified by `device_type`."""
        device = connector.connect_device(device_type)
        return [jax.device_put(t, device) for t in tensors]


# ----- Convenience decorators -----


def run_on_tt_device(f: Callable):
    """Runs any decorated function on TT device."""

    def wrapper(*args, **kwargs):
        module = TestModule(f, args=args, kwargs=kwargs)
        return DeviceRunner.run_on_tt_device(module)

    return wrapper


def run_on_cpu(f: Callable):
    """Runs any decorated function on CPU."""

    def wrapper(*args, **kwargs):
        module = TestModule(f, args=args, kwargs=kwargs)
        return DeviceRunner.run_on_cpu(module)

    return wrapper


def run_on_gpu(f: Callable):
    """Runs any decorated function on GPU."""

    def wrapper(*args, **kwargs):
        module = TestModule(f, args=args, kwargs=kwargs)
        return DeviceRunner.run_on_gpu(module)

    return wrapper
