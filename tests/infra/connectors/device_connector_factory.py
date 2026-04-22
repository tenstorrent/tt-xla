# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra.utilities import Framework

from .device_connector import DeviceConnector
from .torch_device_connector import TorchDeviceConnector, torch_device_connector

jax_device_connector = None


class DeviceConnectorFactory:
    """Factory creating DeviceConnectors based on provided framework."""

    # -------------------- Public methods --------------------

    @staticmethod
    def create_connector(framework: Framework) -> DeviceConnector:
        if framework == Framework.JAX:
            global jax_device_connector
            try:
                from .jax_device_connector import JaxDeviceConnector
            except Exception as e:
                raise RuntimeError(
                    "JAX device connector is unavailable in this environment"
                ) from e

            if jax_device_connector is None:
                jax_device_connector = JaxDeviceConnector()

            return jax_device_connector
        elif framework == Framework.TORCH:
            global torch_device_connector

            if torch_device_connector is None:
                torch_device_connector = TorchDeviceConnector()

            return torch_device_connector
        else:
            raise ValueError(f"Unsupported framework {framework}")
