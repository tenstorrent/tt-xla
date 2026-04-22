# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra.connectors import DeviceConnectorFactory
from infra.utilities.types import Framework

from .device_runner import DeviceRunner


class DeviceRunnerFactory:
    """Factory creating DeviceRunners based on provided framework."""

    @staticmethod
    def create_runner(framework: Framework) -> DeviceRunner:
        connector = DeviceConnectorFactory.create_connector(framework)

        if framework == Framework.JAX:
            from .jax_device_runner import JaxDeviceRunner

            return JaxDeviceRunner(connector)
        elif framework == Framework.TORCH:
            from .torch_device_runner import TorchDeviceRunner

            return TorchDeviceRunner(connector)
        else:
            raise ValueError(f"Unsupported framework {framework}")
