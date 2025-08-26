# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from infra.utilities import Device


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

    It is a thread-safe singleton, ensuring derived class is instantiated only once,
    thus making sure that PJRT plugin is registered exactly once. Registering it
    multiple times would cause error otherwise.
    """

    # Singleton instance as class attribute.
    _instance: DeviceConnector = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self) -> None:
        pass

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

    def _supported_devices(self) -> Sequence[DeviceType]:
        """Returns list of supported device types."""
        # NOTE The order here is important, JAX will respect that order to choose the
        # default device. That way we don't need to use `with jax.default_device` to
        # generate inputs on CPU etc. It also makes a difference in how JAX puts sharded
        # tensors on device (https://github.com/tenstorrent/tt-xla/issues/542).
        return [DeviceType.CPU, DeviceType.TT]

    @abstractmethod
    def _number_of_devices(self, device_type: DeviceType) -> int:
        """Returns the number of available devices of specified type."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        """Returns handle for device identified by `device_type`."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_number_of_tt_devices(self) -> int:
        """Returns number of available TT devices."""
        return self._number_of_devices(DeviceType.TT)

    def get_number_of_cpus(self) -> int:
        """Returns number of available CPUs."""
        return self._number_of_devices(DeviceType.CPU)
