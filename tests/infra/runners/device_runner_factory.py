# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from connectors import DeviceConnectorFactory
from utilities.types import Framework

from .device_runner import DeviceRunner
from .jax_device_runner import JaxDeviceRunner
from .torch_device_runner import TorchDeviceRunner


class DeviceRunnerFactory:
    """Factory creating DeviceRunners based on provided framework."""

    # -------------------- Public methods --------------------

    def __init__(self, framework: Framework) -> None:
        self._framework = framework

    def create_runner(self) -> DeviceRunner:
        connector = DeviceConnectorFactory(self._framework).create_connector()

        if self._framework == Framework.JAX:
            return JaxDeviceRunner(connector)
        elif self._framework == Framework.TORCH:
            return TorchDeviceRunner(connector)
        else:
            raise ValueError(f"Unsupported framework {self._framework}")
