# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from utilities.types import Device

# Relative path to PJRT plugin for TT devices.
TT_PJRT_PLUGIN_RELPATH = "build/src/tt/pjrt_plugin_tt.so"


class DeviceType(Enum):
    """Supported devices."""

    CPU = "cpu"
    TT = "tt"


class DeviceConnector(ABC):
    """
    Class providing connections to devices on which framework specific commands will be
    executed.

    It is an abstract base class meant to be derived from to implement framework
    specific behaviour of certain methods.

    It is a per-class singleton, ensuring derived class is instantiated only once, thus
    making sure that PJRT plugin is registered exactly once. Registering it multiple
    times would cause error.
    """

    # Map of derived classes to concrete instances of those classes. Ensures per-class
    # singleton behaviour.
    _instances = {}

    # -------------------- Public methods --------------------

    def connect_tt_device(self, device_num: int = 0) -> Device:
        """Returns TT device handle."""
        return self.connect_device(DeviceType.TT, device_num)

    def connect_cpu(self) -> Device:
        """Returns CPU device handle."""
        return self.connect_device(DeviceType.CPU)

    def connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        """
        Returns handle for device identified by `device_type`.

        If there are multiple available devices of `device_type`, `device_num` makes it
        possible to choose between them. By default, returns first available device.
        """
        assert (
            device_type in self._supported_devices()
        ), f"Unsupported device {device_type}"
        assert device_num >= 0 and device_num < self._number_of_devices(device_type)

        return self._connect_device(device_type, device_num)

    # -------------------- Protected methods --------------------

    # --- For subclasses to override ---

    @abstractmethod
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        """Returns handle for device identified by `device_type`."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _number_of_devices(self, device_type: DeviceType) -> int:
        """Returns the number of available devices of specified type."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize_backend(self) -> None:
        """Registers custom TT plugin which will make TTDevice available."""
        raise NotImplementedError("Subclasses must implement this method")

    # ----------------------------------

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls, *args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]

    def __init__(self) -> None:
        # We need to ensure __init__ body is executed once. It will be called each time
        # object is constructed.
        if self._is_initialized():
            return

        plugin_path = os.path.join(os.getcwd(), TT_PJRT_PLUGIN_RELPATH)
        if not os.path.exists(plugin_path):
            raise FileNotFoundError(
                f"Could not find tt_pjrt C API plugin at {plugin_path}"
            )

        self._plugin_path = plugin_path
        self._initialized = False
        self._default_xla_flags = os.environ.get("XLA_FLAGS", "")

        self._initialize_backend()
        self._initialized = True

    def _supported_devices(self) -> Sequence[DeviceType]:
        """Returns list of supported device types."""
        return [DeviceType.CPU, DeviceType.TT]

    # -------------------- Private methods --------------------

    def _is_initialized(self) -> bool:
        """Checks if connector is already initialized."""
        return hasattr(self, "_initialized") and self._initialized == True
