# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from utilities.types import Framework

from .device_connector import DeviceConnector
from .jax_device_connector import JaxDeviceConnector
from .torch_device_connector import TorchDeviceConnector


class DeviceConnectorFactory:
    """Factory creating DeviceConnectors based on provided framework."""

    # -------------------- Public methods --------------------

    def __init__(self, framework: Framework) -> None:
        self._framework = framework

    def create_connector(self) -> DeviceConnector:
        if self._framework == Framework.JAX:
            return JaxDeviceConnector()
        elif self._framework == Framework.TORCH:
            return TorchDeviceConnector()
        else:
            raise ValueError(f"Unsupported framework {self._framework}")
