# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from functools import reduce
from typing import Generator

import jax
import jax._src.xla_bridge as xb
from utilities.types import Device

from .device_connector import DeviceConnector, DeviceType


class JaxDeviceConnector(DeviceConnector):
    """Device connector used with JAX."""

    # -------------------- Public methods --------------------

    def get_tt_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns TTDevice mesh with specified `shape` and `axis_names`."""
        tt_devices = jax.devices(DeviceType.TT.value)
        return jax.make_mesh(shape, axis_names, devices=tt_devices)

    def get_cpu_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns cpu device mesh with specified `shape` and `axis_names`."""
        cpu_devices = jax.devices(DeviceType.CPU.value)
        return jax.make_mesh(shape, axis_names, devices=cpu_devices)

    @contextmanager
    def simulate_cpu_mesh(self, mesh_shape: tuple) -> Generator[None, None, None]:
        """
        Context manager that simulates multiple CPU devices by setting a flag that tells
        XLA to simulate a specific number of host devices.
        This will only set CPU the devices the first time that it is used, due to the way
        XLA uses env flags responsible for enabling CPU virtualization. It configures it
        in the beginning, and all following uses of imported jax lib will be configured
        that way. This is not desired behaviour since in one python run we cannot mix
        multichip and singlechip tests.
        """
        num_virtual_cpus = reduce(lambda x, y: x * y, mesh_shape, 1)
        self._simulate_multiple_cpu_devices(num_virtual_cpus)

        try:
            yield
        finally:
            # Teardown: reset the XLA flags after the block completes
            self._reset_xla_flags()

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        return jax.devices(device_type.value)[device_num]

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        return len(jax.devices(device_type.value))

    # @override
    def _initialize_backend(self) -> None:
        xb.register_plugin(
            DeviceType.TT.value,
            priority=500,
            library_path=self._plugin_path,
            options=None,
        )
        jax.config.update("jax_platforms", self._supported_devices_str())

    # -----------------

    def _simulate_multiple_cpu_devices(self, num_devices: int) -> None:
        """Sets a flag that tells XLA to simulate multiple CPU devices."""
        platfrom_device_count_flag = (
            f" --xla_force_host_platform_device_count={num_devices}"
        )
        os.environ["XLA_FLAGS"] = self._default_xla_flags + platfrom_device_count_flag

    def _reset_xla_flags(self) -> None:
        """Resets XLA flags to default."""
        os.environ["XLA_FLAGS"] = self._default_xla_flags

    def _supported_devices_str(self) -> str:
        """Returns comma separated list of supported devices as a string."""
        # Note no space, only comma.
        return ",".join([device.value for device in self._supported_devices()])
