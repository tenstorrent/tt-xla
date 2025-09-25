# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils._pytree import tree_map
from infra.connectors import DeviceConnector
from infra.utilities import Device, Tensor
from infra.workloads import Workload

from .device_runner import DeviceRunner


class TorchDeviceRunner(DeviceRunner):
    """Device runner used with torch."""

    def __init__(self, device_connector: DeviceConnector) -> None:
        self.training_mode = False
        super().__init__(device_connector)

    def set_training_mode(self, training_mode: bool = True) -> None:
        self.training_mode = training_mode

    # @override
    def _run_on_device(self, workload: Workload, device: Device) -> Tensor:
        # Provide a context manager to enable or disable gradient calculation.
        with torch.set_grad_enabled(self.training_mode):
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

        def attempt_to_device(x):
            if hasattr(x, "to"):
                return x.to(device)
            return x

        args_on_device = tree_map(attempt_to_device, workload.args)
        kwargs_on_device = tree_map(attempt_to_device, workload.kwargs)

        if workload.model is not None:
            workload.model.to(device)

        if workload.compiled_executable is not None:
            workload.compiled_executable.to(device)

        return Workload(
            framework=workload.framework,
            model=workload.model,  # Moved to device if not None.
            executable=workload.executable,  # Unchanged.
            compiled_executable=workload.compiled_executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
        )
