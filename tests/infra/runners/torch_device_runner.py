# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect

import torch
import torch.nn.utils.parametrize as parametrize
import torch_xla.distributed.spmd as xs
from infra.connectors import DeviceConnector, DeviceType
from infra.utilities import Device, Tensor
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
from loguru import logger
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


def _register_activation_constraints(activation_specs, mesh) -> None:
    """Register forward hooks that constrain each module's OUTPUT sharding.

    ``activation_specs`` maps a module -> output partition spec (``None`` means
    fully replicate). Hooks are idempotent and registered at most once per module
    (this runs on every execution). The constraint only fires for XLA tensors, so
    it is a no-op on the CPU golden path.
    """
    if not activation_specs:
        return
    from tt_torch.sharding import sharding_constraint_tensor

    def _make_hook(spec):
        def _hook(_module, _inputs, out):
            if not torch.is_tensor(out) or out.device.type != "xla":
                return out
            resolved = spec if spec is not None else tuple([None] * out.dim())
            return sharding_constraint_tensor(out, mesh, resolved)

        return _hook

    for module, spec in activation_specs.items():
        if getattr(module, "_tt_activation_constraint_hooked", False):
            continue
        module.register_forward_hook(_make_hook(spec))
        module._tt_activation_constraint_hooked = True


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

    @staticmethod
    def _apply_weight_dtype_overrides(model: torch.nn.Module, config) -> bool:
        """Register the weight-dtype parametrization on `model`, once.

        Returns True if the model carries the weight-dtype parametrization when
        this returns -- either because it was already registered by an earlier
        device placement, or because it was registered here.

        The "already registered" check looks for WeightDtypeParametrization
        specifically rather than for any parametrization: models may ship with
        their own (e.g. nn.utils.parametrizations.weight_norm / orthogonal), and
        treating those as ours would silently skip the override.
        """
        from tt_torch.weight_dtype import (
            WeightDtypeParametrization,
            apply_weight_dtype_overrides,
        )

        already_applied = any(
            isinstance(p, WeightDtypeParametrization)
            for m in model.modules()
            if parametrize.is_parametrized(m)
            for param_list in m.parametrizations.values()
            for p in param_list
        )
        if already_applied:
            return True

        applied = apply_weight_dtype_overrides(model, config)
        if not applied:
            # The config resolved to no parameters. Report it loudly: silently
            # continuing would drop the dtype override *and* the weight tying.
            logger.warning(
                f"Weight dtype config {config} matched no parameters on "
                f"{type(model).__name__}; running without dtype overrides."
            )
            return False

        logger.info(f"Applied {len(applied)} weight dtype overrides from {config}")
        return True

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

            weight_dtype_config = getattr(workload, "weight_dtype_config", None)
            parametrized = False
            if weight_dtype_config is not None and device.type != "cpu":
                # Inference weight-dtype overrides parametrize the model's
                # weights so a bfp8 custom_call is emitted into the on-device
                # trace. This must run AFTER device placement so the
                # parametrization is alive and on-device when torch.compile
                # traces (that is what emits the annotation). Placement runs on
                # every invocation, so guard against re-registering.
                parametrized = self._apply_weight_dtype_overrides(
                    workload.model, weight_dtype_config
                )

            # We need to tie weights for the model after moving it to the device.
            # For torch_xla this is a known quirk. See: https://docs.pytorch.org/xla/release/r2.8/learn/troubleshoot.html#xla-tensor-quirks
            #
            # Tying is mutually exclusive with the weight-dtype parametrization:
            # `weight` is no longer a Parameter, so tie_weights() would route
            # through the parametrization's right_inverse and copy values
            # instead of aliasing. Skip the re-tie only when the model really is
            # parametrized -- for forward-only inference the untied weights hold
            # identical values, leaving numerics unchanged. If the override did
            # not apply (no matching weights, or a config that resolved to
            # nothing), we must still tie.
            if not parametrized and hasattr(workload.model, "tie_weights"):
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

        # Apply activation sharding constraints (on intermediate module OUTPUTS)
        # as forward hooks. Complements the weight mark_sharding above. Registered
        # once per module (this method runs on every execution, e.g. perf warmup).
        if (
            is_multichip
            and device.type != "cpu"
            and getattr(workload, "activation_shard_spec_fn", None)
            and workload.model is not None
        ):
            _register_activation_constraints(
                workload.activation_shard_spec_fn(workload.model), workload.mesh
            )

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
