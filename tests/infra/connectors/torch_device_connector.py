# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla.core.xla_model as xm
from infra.utilities import Device
from torch_xla.experimental import plugins

from .device_connector import DeviceConnector, DeviceType


class TTPjrtPlugin(plugins.DevicePlugin):
    """Class necessary to register PJRT plugin with torch."""

    # -------------------- Public methods --------------------

    def __init__(self, plugin_path: str) -> None:
        self._plugin_path = plugin_path
        super().__init__()

    def library_path(self):
        return self._plugin_path


class TorchDeviceConnector(DeviceConnector):
    """Device connector used with torch."""

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _connect_device(self, device_type: DeviceType, device_num: int = 0) -> Device:
        # Custom TT devices are discovered through XLA plugin. In case of CPUs, we
        # want to fallback to a regular CPU on host, which torch sees natively
        # through `torch.device("cpu")`.
        return (
            xm.xla_device(device_num)
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

    # @override
    def _register_plugin(self, plugin_path: str) -> None:
        try:
            os.environ["PJRT_DEVICE"] = DeviceType.TT.value
            os.environ["XLA_STABLEHLO_COMPILE"] = "1"

            plugins.register_plugin(DeviceType.TT.value, TTPjrtPlugin(plugin_path))
        except Exception as e:
            raise RuntimeError("Failed to initialize TT PJRT plugin for Torch.") from e


# Global singleton instance.
torch_device_connector: TorchDeviceConnector = None
