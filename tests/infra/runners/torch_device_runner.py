# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from infra.connectors import DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import Workload

from .device_runner import DeviceRunner


class TorchDeviceRunner(DeviceRunner):
    """Device runner used with torch."""

    # @override
    def _run_on_device(self, workload: Workload, device: Device) -> Tensor:
        # TODO this context manager disables gradient calculation to save memory. We
        # will need to enable it for training.

        with torch.no_grad():
            return workload.execute()

    # @override
    def _safely_put_workload_on_device(
        self, workload: Workload, device: Device
    ) -> Workload:
        """
        Puts workload's args and kwargs on device only if `.to()` supports it and also
        puts model if workload is carrying one on device. Returns new workload which is
        "on device".
        """
        assert workload.is_torch, "Workload must be Torch workload to put on device"

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

        # PyTorch needs executable for comparing CPU and TT device outputs (see comparator.py -> compare() -> _match_data_types())
        return Workload(
            framework=workload.framework,
            model=workload.model,  # Moved to device if not None.
            executable=workload.executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
        )
