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

    # @override
    def _register_plugin(self, wheel_plugin_path: str, build_plugin_path: str) -> None:
        """
        Registers TT plugin which will make TTDevice available in PyTorch/XLA.

        For wheel installs, registers the plugin installed from wheel. If wheel plugin
        is not available, registers the plugin from build directory.
        """
        try:
            os.environ["PJRT_DEVICE"] = DeviceType.TT.value
            os.environ["XLA_STABLEHLO_COMPILE"] = "1"

            if wheel_plugin_path is None:
                plugin_path = build_plugin_path
            else:
                plugin_path = wheel_plugin_path
                # Export path to metal so it is accessible by the plugin.
                tt_metal_path = os.path.join(
                    os.path.dirname(wheel_plugin_path), "tt-mlir/install/tt-metal"
                )
                os.environ["TT_METAL_HOME"] = str(tt_metal_path)

            plugins.register_plugin(DeviceType.TT.value, TTPjrtPlugin(plugin_path))
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize TT PJRT plugin for PyTorch from wheel or local build."
            ) from e
        
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

    


# Global singleton instance.
torch_device_connector: TorchDeviceConnector = None
