# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.distributed.spmd as xs
from infra.connectors import DeviceConnector
from infra.utilities import Device, Tensor
from infra.workloads import Workload
from torch.utils._pytree import tree_map

from .device_runner import DeviceRunner


def to_device(x, device):
    """
    Recursively move data structures and objects to the specified device.

    This function handles:
    - Basic Python containers (list, tuple, dict)
    - PyTorch tensors and models (objects with .to() method)
    - Custom objects with attributes (recursively processes all fields)
    - None values and other primitives (returned unchanged)

    Args:
        x: The data structure or object to move to device
        device: The target device (e.g., 'cuda', 'cpu', torch.device)

    Returns:
        The same structure with all compatible elements moved to the device
    """
    if x is None:
        return x
    elif isinstance(x, list):
        return [to_device(item, device) for item in x]
    elif isinstance(x, tuple):
        return tuple(to_device(item, device) for item in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif hasattr(x, "to"):
        return x.to(device)
    # Handle objects with attributes by recursively processing all fields
    elif hasattr(x, "__dict__"):
        for attr_name in x.__dict__:
            attr_value = getattr(x, attr_name)
            setattr(x, attr_name, to_device(attr_value, device))
        return x
    else:
        return x


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

        args_on_device = tree_map(lambda x: to_device(x, device), workload.args)
        kwargs_on_device = tree_map(lambda x: to_device(x, device), workload.kwargs)

        if workload.model is not None:
            workload.model = workload.model.to(device)

        shard_specs = workload.shard_spec_fn and workload.shard_spec_fn(workload.model)
        is_multichip = workload.mesh and len(workload.mesh.device_ids) > 1

        if shard_specs is not None and is_multichip and device.type != "cpu":
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, workload.mesh, shard_spec)

        workload.compiled_executable = to_device(workload.compiled_executable, device)

        return Workload(
            framework=workload.framework,
            model=workload.model,  # Moved to device if not None.
            executable=workload.executable,  # Unchanged.
            compiled_executable=workload.compiled_executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
        )
