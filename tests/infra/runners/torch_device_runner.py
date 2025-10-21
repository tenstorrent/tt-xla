# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
import torch_xla.distributed.spmd as xs
from infra.connectors import DeviceConnector
from infra.utilities import Device, Tensor
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
from torch.utils._pytree import tree_map

from .device_runner import DeviceRunner


def to_device(x, device, depth=5):
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
        depth: Maximum recursion depth (default: 5). When depth reaches 0,
               recursion stops and objects are returned as-is.

    Returns:
        The same structure with all compatible elements moved to the device
    """
    # Stop recursion when maximum depth is reached
    if depth <= 0:
        # Still try to move tensors/models at the final depth level
        if hasattr(x, "to"):
            return x.to(device)
        return x

    if x is None:
        return x
    elif isinstance(x, list):
        return [to_device(item, device, depth - 1) for item in x]
    elif isinstance(x, tuple):
        return tuple(to_device(item, device, depth - 1) for item in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device, depth - 1) for k, v in x.items()}
    elif hasattr(x, "to"):
        return x.to(device)
    # Handle objects with attributes by recursively processing all fields.
    # This is done in-place.
    elif hasattr(x, "__dict__"):
        for attr_name in x.__dict__:
            attr_value = getattr(x, attr_name)
            setattr(x, attr_name, to_device(attr_value, device, depth - 1))
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

        if workload.model is not None and hasattr(workload.model, "to"):
            workload.model = workload.model.to(device)

        shard_specs = None
        if (
            device.type != "cpu"
            and hasattr(workload, "shard_spec_fn")
            and workload.shard_spec_fn
        ):
            sig = inspect.signature(workload.shard_spec_fn)
            param_names = list(sig.parameters.keys())

            # Check if function expects args and kwargs (data parallel)
            if (
                len(param_names) == 2
                and "args" in param_names
                and "kwargs" in param_names
            ):
                shard_specs = workload.shard_spec_fn(args_on_device, kwargs_on_device)
            else:
                assert (
                    workload.model is not None
                ), "Tensor parallel workloads require a nn.Module to shard weights"
                # Do we need to shard actications as well?
                shard_activations = (
                    len(param_names) == 3
                    and "args" in param_names
                    and "kwargs" in param_names
                )
                if shard_activations:
                    shard_specs = workload.shard_spec_fn(
                        workload.model, args_on_device, kwargs_on_device
                    )
                else:
                    shard_specs = workload.shard_spec_fn(workload.model)

        is_multichip = (
            hasattr(workload, "mesh")
            and workload.mesh
            and len(workload.mesh.device_ids) > 1
        )

        if shard_specs is not None and is_multichip and device.type != "cpu":
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, workload.mesh, shard_spec)

        # In the future, we will deprecate `workload.model` and use only
        # `workload.compiled_executable` carrying the model.
        # So we also move it to the device. But we have to check if compiled_executable has '.to' method.
        # If we compiled function, compiled_executable will be a callable
        # which doesn't have `.to()` method (function is not loaded on device).
        workload.compiled_executable = to_device(workload.compiled_executable, device)

        return TorchWorkload(
            model=workload.model,  # Moved to device if not None.
            executable=workload.executable,  # Unchanged.
            compiled_executable=workload.compiled_executable,  # Unchanged.
            args=args_on_device,
            kwargs=kwargs_on_device,
        )
