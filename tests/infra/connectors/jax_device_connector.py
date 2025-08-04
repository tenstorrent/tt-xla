# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax._src.xla_bridge as xb
from infra.utilities import Device

from .device_connector import DeviceConnector, DeviceType


class JaxDeviceConnector(DeviceConnector):
    """Device connector used with JAX."""

    # @override
    def _register_plugin(self, wheel_plugin_path: str, build_plugin_path: str) -> None:
        """
        Registers TT plugin which will make TTDevice available in JAX.

        For wheel installs plugin is self-registered so this only updates the JAX config.
        If wheel plugin is not available, registers the plugin from build directory.
        """
        try:
            if wheel_plugin_path is not None:
                # Wheel-installed plugin is present, it will self-register on demand.
                self._update_jax_config()
                return

            # No wheel plugin found, fall back to local build.
            xb.register_plugin(DeviceType.TT.value, library_path=build_plugin_path)
            self._update_jax_config()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize TT PJRT plugin for JAX from wheel or local build."
            ) from e

    def _update_jax_config(self) -> None:
        # Allocating enough CPU devices so we can create various meshes depending on
        # which TT device tests are running. Can't be set to exact number of TT
        # devices because after calling `jax.devices` function this config update
        # doesn't work anymore. This needs to be done before any other `jax` commands.
        jax.config.update("jax_num_cpu_devices", 8)
        # Update available platforms.
        jax.config.update("jax_platforms", self._supported_devices_str())

    def _supported_devices_str(self) -> str:
        """Returns comma separated list of supported devices as a string."""
        # Note no space, only comma.
        return ",".join([device.value for device in self._supported_devices()])

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        return len(jax.devices(device_type.value))

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        return jax.devices(device_type.value)[device_num]

    def get_tt_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns TT device mesh with specified `shape` and `axis_names`."""
        tt_devices = jax.devices(DeviceType.TT.value)
        return jax.make_mesh(shape, axis_names, devices=tt_devices)

    def get_cpu_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns CPU mesh with specified `shape` and `axis_names`."""
        cpu_devices = jax.devices(DeviceType.CPU.value)
        return jax.make_mesh(shape, axis_names, devices=cpu_devices)


# Global singleton instance.
jax_device_connector: JaxDeviceConnector = None
