# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from enum import Enum
from typing import Sequence

import jax
import jax._src.xla_bridge as xb

# Relative path to PJRT plugin for TT devices.
TT_PJRT_PLUGIN_RELPATH = "build/src/tt/pjrt_plugin_tt.so"


class DeviceType(Enum):
    CPU = "cpu"
    TT = "tt"
    GPU = "gpu"


class DeviceConnector:
    """
    Singleton class providing connections to devices on which jax commands will be
    executed.

    As a singleton it is instantiated only once, thus making sure that PJRT plugin is
    registered exactly once. Registration needs to happen before any other jax commands
    are executed. Registering it multiple times would cause error.

    Do not instantiate this class directly. Use provided factory method instead.

    TODO (kmitrovic) see how to make this class a thread safe singleton if needed.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        # Ensure that only one instance of the class is created.
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self) -> None:
        """Don't use directly, use provided factory method instead."""
        # We need to ensure __init__ body is executed once. It will be called each time
        # `DeviceConnector()` is called.
        if self.is_initialized():
            return

        self._initialized = False

        plugin_path = os.path.join(os.getcwd(), TT_PJRT_PLUGIN_RELPATH)
        if not os.path.exists(plugin_path):
            raise FileNotFoundError(
                f"Could not find tt_pjrt C API plugin at {plugin_path}"
            )

        self._plugin_path = plugin_path
        self._initialize_backend()

    def is_initialized(self) -> bool:
        """Checks if connector is already initialized."""
        if hasattr(self, "_initialized") and self._initialized == True:
            return True

        return False

    def get_tt_device_mesh(self, shape: tuple, axis_names: tuple) -> jax.sharding.Mesh:
        """Returns TTDevice mesh with specified `shape` and `axis_names`."""
        tt_devices = jax.devices(DeviceType.TT.value)
        return jax.make_mesh(shape, axis_names, devices=tt_devices)

    def connect_tt_device(self, device_num: int = 0) -> jax.Device:
        """Returns TTDevice handle."""
        return self.connect_device(DeviceType.TT, device_num)

    def connect_cpu(self) -> jax.Device:
        """Returns CPUDevice handle."""
        return self.connect_device(DeviceType.CPU)

    def connect_gpu(self) -> jax.Device:
        """Returns GPUDevice handle."""
        return self.connect_device(DeviceType.GPU)

    def connect_device(
        self, device_type: DeviceType, device_num: int = 0
    ) -> jax.Device:
        """
        Returns handle for device identified by `device_type`.

        If there are multiple available devices of `device_type`, `device_num` makes it
        possible to choose between them. By default, returns first available device.
        """
        assert device_num < self._number_of_devices(device_type)
        assert device_num >= 0

        return jax.devices(device_type.value)[device_num]

    def _number_of_devices(self, device_type: DeviceType) -> int:
        """Returns the number of available devices of specified type."""
        return len(jax.devices(device_type.value))

    def _supported_devices(self) -> Sequence[DeviceType]:
        """Returns list of supported device types."""
        # TODO support GPU
        return [DeviceType.CPU, DeviceType.TT]

    def _supported_devices_str(self) -> str:
        """Returns comma separated list of supported devices as a string."""
        # Note no space, only comma.
        return ",".join([device.value for device in self._supported_devices()])

    def _initialize_backend(self) -> None:
        """
        Registers TT plugin which will make TTDevice available in jax.

        Needs to be called before any other jax command.
        """
        xb.register_plugin(
            DeviceType.TT.value,
            priority=500,
            library_path=self._plugin_path,
            options=None,
        )
        jax.config.update("jax_platforms", self._supported_devices_str())

        self._initialized = True


# `DeviceConnector._initialize_backend` must be executed before anything jax related is
# called. By providing this global instance, that is secured.
device_connector = DeviceConnector()
