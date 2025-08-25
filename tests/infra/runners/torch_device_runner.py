# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from infra.connectors import DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import TorchWorkload, Workload

from .device_runner import DeviceRunner

# Move individual tensors to the specified device, while preserving the structure of the input collection.
def to_device(tensor_collection, device: Device):
    result = []
    assert isinstance(
        tensor_collection, (list, tuple)
    ), f"Expected tensor_collection to be a list or tuple, got {type(tensor_collection)}"
    for item in tensor_collection:
        if isinstance(item, (list)):
            result.append(to_device(item, device))
        elif isinstance(item, tuple):
            result.append(to_device(item, device))
        else:
            if not isinstance(item, Tensor):
                result.append(item)
                continue
            result.append(item.to(device))

    result = result if isinstance(tensor_collection, list) else tuple(result)
    return result


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
        assert isinstance(workload, TorchWorkload)

        args_on_device = []
        kwargs_on_device = {}

        args_on_device = to_device(workload.args, device)

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
