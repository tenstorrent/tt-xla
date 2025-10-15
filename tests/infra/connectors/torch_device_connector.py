# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra.utilities import Device

from .device_connector import DeviceConnector, DeviceType


class TorchDeviceConnector(DeviceConnector):
    """Device connector used with torch."""

    def __init__(self) -> None:
        super().__init__()
        xr.initialize_cache(self.get_cache_dir())

    @staticmethod
    def get_cache_dir() -> str:
        return f"{os.getcwd()}/tmp/"

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        # Custom TT devices are discovered through XLA plugin. In case of CPUs, we
        # want to fallback to a regular CPU on host, which torch sees natively
        # through `torch.device("cpu")`.
        return (
            torch_xla.device(device_num)
            if device_type == DeviceType.TT
            else torch.device(device_type.value)
        )

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        # Torch does not have an API to retrieve number of CPUs, so we use system
        # lib `os`.
        return (
            len(xm.get_xla_supported_devices())
            if device_type == DeviceType.TT
            else os.cpu_count()
        )


# Global singleton instance.
torch_device_connector: TorchDeviceConnector = None
