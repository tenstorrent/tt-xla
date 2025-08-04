# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Optional, Sequence

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
        return self._run_on_device(workload, DeviceType.CPU)

    def run_on_tt_device(self, workload: Workload, device_num: int = 0) -> Tensor:
        """Runs `workload` on TT device."""
        return self._run_on_device(workload, DeviceType.TT, device_num)

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

    # Mislim da naredne 4 funkcije ne sluzen nicemu
    ###########################################################################
    def put_on_cpu(self, workload: Workload) -> Workload:
        """Puts `workload` on CPU."""
        return self._put_on_device(workload, device_type=DeviceType.CPU)

    def put_on_tt_device(self, workload: Workload, device_num: int = 0) -> Workload:
        """Puts `workload` on TT device."""
        return self._put_on_device(
            workload, device_type=DeviceType.TT, device_num=device_num
        )

    def put_tensors_on_tt_device(self, *tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on TT device."""
        return self._put_tensors_on_device(DeviceType.TT, tensors)

    def put_tensors_on_cpu(self, *tensors: Tensor) -> Sequence[Tensor]:
        """Puts `tensors` on CPU."""
        return self._put_tensors_on_device(DeviceType.CPU, tensors)

    ###########################################################################

    @abstractmethod
    def _run_on_device(
        self, workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Tensor:
        """Runs `workload` on device identified by `device_type`."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _put_tensors_on_device(
        self, device_type: DeviceType, tensors: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        """Puts `tensors` on device identified by `device_type`."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _safely_put_workload_on_device(
        self, workload: Workload, device: Device
    ) -> Workload:
        """Puts workload's args and kwargs on device."""
        raise NotImplementedError("Subclasses must implement this method")
