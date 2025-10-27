# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional

from infra.connectors import DeviceConnector, DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import Workload


class DeviceRunner(ABC):
    """
    Class providing methods to put and run workload on any supported device.

    It is an abstract base class meant to be derived from to implement framework
    specific behaviour of certain methods.
    """

    def __init__(self, device_connector: DeviceConnector) -> None:
        self._device_connector = device_connector

    def run_on_cpu(self, workload: Workload) -> Tensor:
        """Runs `workload` on CPU."""
        return self.run_on_device(workload, DeviceType.CPU)

    def run_on_tt_device(self, workload: Workload, device_num: int = 0) -> Tensor:
        """Runs `workload` on TT device."""
        return self.run_on_device(workload, DeviceType.TT, device_num)

    def run_on_device(
        self, workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Tensor:
        """Orchestrates workload execution by connecting to device, transferring data, and running."""

        device = self._device_connector.connect_device(device_type, device_num)
        device_workload = self._put_on_device(workload, device=device)

        return self._run_on_device(device_workload, device)

    def _put_on_device(
        self,
        workload: Workload,
        *,
        device: Optional[Device] = None,
        device_type: Optional[DeviceType] = None,
        device_num: Optional[int] = 0,
    ) -> Workload:
        """Puts `workload` on device and returns it."""
        device = device or self._device_connector.connect_device(
            device_type, device_num
        )
        return self._safely_put_workload_on_device(workload, device)

    @abstractmethod
    def _safely_put_workload_on_device(
        self, workload: Workload, device: Device
    ) -> Workload:
        """Puts workload's args and kwargs on device."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _run_on_device(self, workload: Workload, device: Device) -> Tensor:
        """Executes workload on device using framework-specific execution context."""
        raise NotImplementedError("Subclasses must implement this method")

    def serialize_on_device(
        self,
        workload: Workload,
        output_prefix: str,
        device_type: DeviceType = DeviceType.TT,
        device_num: int = 0,
        compiler_options=None,
    ) -> None:
        """
        Serializes a workload after putting it on the specified device.

        Args:
            workload: The workload to serialize
            output_prefix: Base path and filename prefix for output files
            device_type: The type of device to use (default: TT)
            device_num: The device number (default: 0)
            compiler_options: Optional JAX compiler options dict
        """
        device = self._device_connector.connect_device(device_type, device_num)
        device_workload = self._put_on_device(workload, device=device)
        device_workload.serialize(output_prefix, compiler_options=compiler_options)
