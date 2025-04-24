# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import jax
import jax._src.xla_bridge as xb
from infra.utilities import Device
from jax._src.lib import xla_client

from .device_connector import DeviceConnector, DeviceType


class JaxDeviceConnector(DeviceConnector):
    """Device connector used with JAX."""

    # -------------------- Public methods --------------------

    def get_tt_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns TT device mesh with specified `shape` and `axis_names`."""
        tt_devices = jax.devices(DeviceType.TT.value)
        return jax.make_mesh(shape, axis_names, devices=tt_devices)

    def get_cpu_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns CPU mesh with specified `shape` and `axis_names`."""
        cpu_devices = jax.devices(DeviceType.CPU.value)
        return jax.make_mesh(shape, axis_names, devices=cpu_devices)

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        return jax.devices(device_type.value)[device_num]

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        return len(jax.devices(device_type.value))

    # @override
    def _register_plugin(self, plugin_path: str) -> None:
        """
        Registers TT plugin which will make TTDevice available in JAX.

        For source builds, loads the plugin from build directory. For wheel installs,
        imports the prebuilt plugin which self-registers.
        """
        try:
            # Try and see if plugin was installed from a wheel.
            # First check if 'jax_plugins' package exists to avoid ModuleNotFoundError.
            jax_plugins_spec = importlib.util.find_spec("jax_plugins")

            if jax_plugins_spec is not None:
                # Check if the wheel-installed jax plugin exists.
                plugin_spec = importlib.util.find_spec("jax_plugins.pjrt_plugin_tt")

                if plugin_spec is not None:
                    # Wheel-installed plugin is present, it will self-register on demand.
                    self._update_jax_config()
                    return

            # No wheel plugin found, fall back to local build.
            xb.register_plugin(DeviceType.TT.value, library_path=plugin_path)
            self._update_jax_config()

        except Exception as e:
            raise RuntimeError(
                "Failed to initialize TT PJRT plugin for JAX from wheel or local build."
            ) from e

    # -----------------

    def _supported_devices_str(self) -> str:
        """Returns comma separated list of supported devices as a string."""
        # Note no space, only comma.
        return ",".join([device.value for device in self._supported_devices()])

    def _update_jax_config(self) -> None:
        # Allocating enough CPU devices so we can create various meshes depending on
        # which TT device tests are running. Can't be set to exact number of TT
        # devices because after calling `jax.devices` function this config update
        # doesn't work anymore. This needs to be done before any other `jax` commands.
        jax.config.update("jax_num_cpu_devices", 8)
        # Update available platforms.
        jax.config.update("jax_platforms", self._supported_devices_str())


# Global singleton instance.
jax_device_connector: JaxDeviceConnector = None
