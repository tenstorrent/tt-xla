# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect

import torch
import torch_xla.distributed.spmd as xs
from infra.connectors import DeviceConnector, DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
from torch.utils._pytree import tree_map

from .device_runner import DeviceRunner


def to_device(x, device, depth=5, moved=None):
    """
    Recursively move data structures and objects to the specified device.

    This function handles:
    - Basic Python containers (list, tuple, dict)
    - PyTorch tensors and models (objects with .to() method)
    - Custom objects with attributes (recursively processes all fields)
    - None values and other primitives (returned unchanged)
    - Class types (returned unchanged as metadata)
    - Aliasing preservation: if the same object appears multiple times,
      it will be moved once and the same moved object will be reused

    Args:
        x: The data structure or object to move to device
        device: The target device (e.g., 'cuda', 'cpu', torch.device)
        depth: Maximum recursion depth (default: 5). When depth reaches 0,
               recursion stops and objects are returned as-is.
        moved: Dict mapping id(original_object) -> moved_object to preserve aliasing.
               Should not be provided by callers (used internally for recursion).

    Returns:
        The same structure with all compatible elements moved to the device
    """
    if moved is None:
        moved = {}

    # If the object has moved and this is an alias, return the original moved object
    obj_id = id(x)
    if obj_id in moved:
        return moved[obj_id]

    # Stop recursion when maximum depth is reached
    if depth <= 0:
        # Still try to move tensors/models at the final depth level
        if hasattr(x, "to"):
            result = x.to(device)
            moved[obj_id] = result
            return result
        return x

    if x is None:
        return x
    elif isinstance(x, list):
        result = [to_device(item, device, depth - 1, moved) for item in x]
        moved[obj_id] = result
        return result
    elif isinstance(x, tuple):
        result = tuple(to_device(item, device, depth - 1, moved) for item in x)
        moved[obj_id] = result
        return result
    elif isinstance(x, dict):
        result = {k: to_device(v, device, depth - 1, moved) for k, v in x.items()}
        moved[obj_id] = result
        return result
    elif hasattr(x, "to"):
        if isinstance(x, type):
            return x
        result = x.to(device)
        # nn.Module.to() is in-place (returns self). Clone tensors that didn't move
        # so mutations in one run don't corrupt the workload's source state.
        if result is x and isinstance(x, torch.Tensor):
            result = result.clone()
        moved[obj_id] = result
        return result
    elif hasattr(x, "__dict__"):
        if callable(x):
            # Compiled torch functions are callable and may have circular refs in
            # __dict__ (dynamo internals), so copy.copy is unsafe. Mutate in-place.
            moved[obj_id] = x  # guard before recursion to break circular refs
            for attr_name in list(x.__dict__):
                setattr(
                    x,
                    attr_name,
                    to_device(getattr(x, attr_name), device, depth - 1, moved),
                )
            if "device" in x.__dict__ and isinstance(
                x.__dict__["device"], (str, torch.device)
            ):
                x.device = device
            return x
        else:
            # Non-callable plain objects (e.g. transformers StaticCache/StaticLayer)
            # are NOT nn.Modules — they have no .to() and get mutated in-place without
            # a copy. Use copy.copy so each run gets a fresh object, preventing
            # accumulated cache state from corrupting subsequent forward passes.
            new_obj = copy.copy(x)
            moved[obj_id] = new_obj  # guard before recursion to break circular refs
            for attr_name in list(x.__dict__):
                setattr(
                    new_obj,
                    attr_name,
                    to_device(getattr(x, attr_name), device, depth - 1, moved),
                )
            if "device" in new_obj.__dict__ and isinstance(
                new_obj.__dict__["device"], (str, torch.device)
            ):
                new_obj.device = device
            return new_obj
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
    def serialize_on_device(
        self,
        workload: Workload,
        output_prefix: str,
        device_type: DeviceType = DeviceType.TT,
        device_num: int = 0,
        compiler_options=None,
    ) -> None:
        with torch.set_grad_enabled(self.training_mode):
            super().serialize_on_device(
                workload,
                output_prefix,
                device_type=device_type,
                device_num=device_num,
                compiler_options=compiler_options,
            )

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

            # We need to tie weights for the model after moving it to the device.
            # For torch_xla this is a known quirk. See: https://docs.pytorch.org/xla/release/r2.8/learn/troubleshoot.html#xla-tensor-quirks
            if hasattr(workload.model, "tie_weights"):
                workload.model.tie_weights()

        is_multichip = (
            hasattr(workload, "mesh")
            and workload.mesh
            and len(workload.mesh.device_ids) > 1
        )

        shard_specs = None
        if (
            is_multichip
            and device.type != "cpu"
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

        if shard_specs is not None:
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
