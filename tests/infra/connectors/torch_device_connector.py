# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from infra.utilities.types import Device

from .device_connector import DeviceConnector, DeviceType


class TorchDeviceConnector(DeviceConnector):
    """Device connector used with torch."""

    def __init__(self) -> None:
        super().__init__()
        self._tt_runtime_initialized = False

    @staticmethod
    def get_cache_dir() -> str:
        return f"{os.getcwd()}/tmp/"

    def _supported_devices(self):
        return [DeviceType.CPU, DeviceType.CUDA, DeviceType.TT]

    def _ensure_tt_runtime_initialized(self) -> None:
        if self._tt_runtime_initialized:
            return

        import torch_xla
        import torch_xla.runtime as xr

        xr.runtime.set_device_type("TT")
        if not torch_xla._XLAC._xla_computation_cache_is_initialized():
            xr.initialize_cache(self.get_cache_dir())
        self._tt_runtime_initialized = True

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        # Custom TT devices are discovered through XLA plugin. In case of CPUs, we
        # want to fallback to a regular CPU on host, which torch sees natively
        # through `torch.device("cpu")`.
        if device_type == DeviceType.TT:
            self._ensure_tt_runtime_initialized()
            import torch_xla

            return torch_xla.device(device_num)

        return torch.device(device_type.value)

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        # Torch does not have an API to retrieve number of CPUs, so we use system
        # lib `os`.
        if device_type == DeviceType.TT:
            self._ensure_tt_runtime_initialized()
            import torch_xla.core.xla_model as xm

            return len(xm.get_xla_supported_devices())
        if device_type == DeviceType.CUDA:
            return torch.cuda.device_count()
        return os.cpu_count()


# Global singleton instance.
torch_device_connector: TorchDeviceConnector = None
