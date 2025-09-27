# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
from torch.utils._pytree import tree_map
from infra.connectors import DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import Workload

from .device_runner import DeviceRunner

import torch_xla.distributed.spmd as xs


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

        def attempt_to_device(x):
            if hasattr(x, "to"):
                return x.to(device)
            return x

        args_on_device = tree_map(attempt_to_device, workload.args)
        kwargs_on_device = tree_map(attempt_to_device, workload.kwargs)

        if workload.model is not None:
            workload.model = workload.model.to(device)

        shard_specs = (
            None
            if workload.shard_spec_function is None
            else workload.shard_spec_function(workload.model)
        )
        if shard_specs is not None and device.type != "cpu":
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, xs.get_global_mesh(), shard_spec)

        return Workload(
            framework=workload.framework,
            model=workload.model,  # Moved to device if not None.
            executable=workload.executable,  # Unchanged.
            compiled_executable=workload.compiled_executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
        )
