# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from connectors import DeviceConnector, DeviceType
from utilities.multichip_utils import MultichipWorkload, ShardingMode
from utilities.types import Device, Tensor
from utilities.workloads.torch_workload import TorchWorkload, Workload

from .device_runner import DeviceRunner


class TorchDeviceRunner(DeviceRunner):
    """Device runner used with torch."""

    # -------------------- Public methods --------------------

    def __init__(self, connector: DeviceConnector) -> None:
        super().__init__(connector)

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _run_on_device(
        self, workload: Workload, device_type: DeviceType, device_num: int = 0
    ) -> Tensor:
        device = self._device_connector.connect_device(device_type, device_num)
        device_workload = self._put_on_device(workload, device=device)

        # TODO this context manager disables gradient calculation to save memory. We
        # will need to enable it for training.
        with torch.no_grad():
            return device_workload.execute()

    # @override
    def _run_on_multichip_device(
        self, multichip_workload: MultichipWorkload, sharding_mode: ShardingMode
    ) -> Tensor:
        raise NotImplementedError("Multichip support not implemented for torch")

    # @override
    def _put_tensors_on_device(
        self, device_type: DeviceType, tensors: Sequence[Tensor]
    ) -> Sequence[Tensor]:
        device = self._device_connector.connect_device(device_type)
        return [t.to(device) for t in tensors]

    # @override
    def _safely_put_workload_on_device(
        self, workload: Workload, device: Device
    ) -> Workload:
        """
        Puts workload's args and kwargs on device only if `.to()` supports it and also
        puts model if workload is carrying once on device. Returns new workload which is
        "on device".
        """
        assert isinstance(workload, TorchWorkload)

        args_on_device = []
        kwargs_on_device = {}

        for arg in workload.args:
            try:
                args_on_device.append(arg.to(device))
            except:
                args_on_device.append(arg)

        for key, value in workload.kwargs.items():
            try:
                kwargs_on_device[key] = value.to(device)
            except:
                kwargs_on_device[key] = value

        if workload.model is not None:
            workload.model.to(device)

        return TorchWorkload.create(
            workload.executable,  # Unchanged.
            workload.model,  # Moved to device if not None.
            args_on_device,
            kwargs_on_device,
        )
