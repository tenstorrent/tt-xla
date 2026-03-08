# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra.utilities import Device

from .device_connector import DeviceConnector, DeviceType


class TorchDeviceConnector(DeviceConnector):
    """Device connector used with torch."""

    def __init__(self) -> None:
        print(f"\n[DEBUG][TorchDeviceConnector.__init__] CALLED", flush=True)
        super().__init__()
        print(f"[DEBUG][TorchDeviceConnector.__init__] Setting device type to 'TT' via xr.runtime.set_device_type('TT')", flush=True)
        xr.runtime.set_device_type("TT")
        # Only initialize cache if not already initialized (avoids assertion
        # error when tests have already performed XLA operations before using
        # the comparison evaluator)
        if not torch_xla._XLAC._xla_computation_cache_is_initialized():
            cache_dir = self.get_cache_dir()
            print(f"[DEBUG][TorchDeviceConnector.__init__] Initializing XLA computation cache at: {cache_dir}", flush=True)
            xr.initialize_cache(cache_dir)
        else:
            print(f"[DEBUG][TorchDeviceConnector.__init__] XLA computation cache already initialized", flush=True)
        print(f"[DEBUG][TorchDeviceConnector.__init__] DONE", flush=True)

    @staticmethod
    def get_cache_dir() -> str:
        return f"{os.getcwd()}/tmp/"

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        print(f"[DEBUG][TorchDeviceConnector._connect_device] CALLED — device_type={device_type}, device_num={device_num}", flush=True)
        # Custom TT devices are discovered through XLA plugin. In case of CPUs, we
        # want to fallback to a regular CPU on host, which torch sees natively
        # through `torch.device("cpu")`.
        device = (
            torch_xla.device(device_num)
            if device_type == DeviceType.TT
            else torch.device(device_type.value)
        )
        print(f"[DEBUG][TorchDeviceConnector._connect_device] Returning device: {device}", flush=True)
        return device

    # @override
    def _number_of_devices(self, device_type: DeviceType) -> int:
        # Torch does not have an API to retrieve number of CPUs, so we use system
        # lib `os`.
        count = (
            len(xm.get_xla_supported_devices())
            if device_type == DeviceType.TT
            else os.cpu_count()
        )
        print(f"[DEBUG][TorchDeviceConnector._number_of_devices] device_type={device_type}, count={count}", flush=True)
        return count


# Global singleton instance.
torch_device_connector: TorchDeviceConnector = None
