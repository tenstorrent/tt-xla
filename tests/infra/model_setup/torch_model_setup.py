# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model lifecycle helpers extracted from TorchModelTester.

These functions handle model configuration, dtype casting, workload creation,
and training flows for PyTorch models. They are pure functions that take explicit
parameters instead of reading from self._* attributes.
"""

from __future__ import annotations

import collections
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import torch
import torch_xla
import torch_xla.runtime as xr
from infra.evaluators import ComparisonResult
from infra.utilities import Framework
from infra.workloads import TorchWorkload, Workload
from tt_torch.sharding import sharding_constraint_tensor
from ttxla_tools.logging import logger

from .jax_model_setup import RunMode


@contextmanager
def _mask_jax_accelerator():
    """Temporarily hide jax accelerator to avoid inductor issues with no-tensor-input graphs."""
    original_fn = torch.accelerator.is_available

    def masked_is_available():
        try:
            acc = torch.accelerator.current_accelerator()
            if acc and acc.type == "jax":
                return False
        except RuntimeError:
            pass
        return original_fn()

    torch.accelerator.is_available = masked_is_available
    try:
        yield
    finally:
        torch.accelerator.is_available = original_fn


def configure_torch_model(model: torch.nn.Module, run_mode: RunMode) -> None:
    """Configures a PyTorch model for inference or training mode."""
    assert isinstance(model, torch.nn.Module)
    if run_mode == RunMode.INFERENCE:
        model.eval()
    else:
        model.train()


def calculate_model_size(model: torch.nn.Module) -> int | None:
    """Calculate total number of parameters in a PyTorch model."""
    if isinstance(model, torch.nn.Module):
        size = sum(p.numel() for p in model.parameters())
        logger.debug(f"Model size: {size / 1e9}B")
        return size
    logger.debug("Model is not a torch.nn.Module, skipping size calculation")
    return None


def cast_torch_model_dtype(model: torch.nn.Module, dtype) -> torch.nn.Module:
    """Applies dtype override to a PyTorch model. Returns the cast model."""
    if hasattr(model, "to"):
        return model.to(dtype)
    raise TypeError("Model does not have 'to' method to apply dtype.")


def cast_torch_inputs_dtype(inputs, dtype):
    """Applies dtype override to PyTorch inputs, returning new values."""
    return _cast_tensors_to_dtype(inputs, dtype)


def get_torch_forward_method_args(input_activations) -> Sequence[Any]:
    """Returns positional arguments for model's forward pass."""
    if isinstance(input_activations, torch.Tensor):
        return [input_activations]
    if isinstance(input_activations, (tuple, list)):
        return list(input_activations)
    return []


def get_torch_forward_method_kwargs(input_activations) -> Mapping[str, Any]:
    """Returns keyword arguments for model's forward pass."""
    if isinstance(input_activations, collections.abc.Mapping):
        return {**input_activations}
    return {}


def create_torch_model_workload(
    model: torch.nn.Module,
    input_activations,
    run_mode: RunMode = RunMode.INFERENCE,
    compiler_config=None,
    dtype_override=None,
    mesh=None,
    shard_spec_fn=None,
    parallelism=None,
) -> TorchWorkload:
    """Creates a TorchWorkload from a model and its inputs.

    This replaces the model lifecycle that was spread across TorchModelTester.__init__,
    _initialize_components, _cache_model_inputs, _initialize_workload, etc.
    """
    # Configure model
    configure_torch_model(model, run_mode)
    calculate_model_size(model)

    # Apply dtype overrides
    if dtype_override is not None:
        model = cast_torch_model_dtype(model, dtype_override)
        input_activations = cast_torch_inputs_dtype(input_activations, dtype_override)

    # Build args/kwargs from input_activations
    args = get_torch_forward_method_args(input_activations)
    kwargs = get_torch_forward_method_kwargs(input_activations)

    assert (
        len(args) > 0 or len(kwargs) > 0
    ), "Forward method args or kwargs or both must be provided"

    workload = TorchWorkload(
        model=model,
        args=args,
        kwargs=kwargs,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )

    if parallelism is not None:
        from third_party.tt_forge_models.config import Parallelism

        if parallelism == Parallelism.TENSOR_PARALLEL:
            assert (
                workload.shard_spec_fn is not None
            ), "Tensor parallel requires shard specs function"
            assert (
                workload.mesh and len(workload.mesh.device_ids) > 1
            ), "Tensor parallel requires multi-chip mesh"

    return workload


def run_torch_training(
    tester,
    workload: TorchWorkload,
    model: torch.nn.Module,
    unpack_forward_output=None,
    parallelism=None,
) -> Tuple[ComparisonResult, ...]:
    """Runs PyTorch training flow with backward pass.

    Args:
        tester: Tester instance (used for compile, run, compare).
        workload: The model workload.
        model: The original torch.nn.Module (needed for gradient extraction).
        unpack_forward_output: Optional function to unpack model output to a tensor.
        parallelism: Parallelism mode (for tensor parallel gradient sharding).
    """
    framework = tester.framework

    if unpack_forward_output is None:
        unpack_forward_output = lambda x: x

    # Initialize XLA computation client for autograd engine device queues
    torch_xla._XLAC._init_computation_client()

    # Run forward on CPU
    tester._compile_for_cpu(workload)
    with _mask_jax_accelerator():
        cpu_res = tester.device_runner.run_on_cpu(workload)
    cpu_res = unpack_forward_output(cpu_res)

    # Generate random gradient
    random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)

    # Create and run backward on CPU
    cpu_backward_workload = Workload(
        framework=framework,
        executable=cpu_res.backward,
        args=[],
        kwargs={"gradient": random_grad},
    )
    with _mask_jax_accelerator():
        tester.device_runner.run_on_cpu(cpu_backward_workload)

    cpu_grads, cpu_none_grads = _extract_grads(model)
    workload.model.zero_grad()

    # Run forward on TT
    compile_options = {"tt_experimental_compile": False}

    from third_party.tt_forge_models.config import Parallelism

    if parallelism == Parallelism.TENSOR_PARALLEL:
        compile_options["tt_enable_torch_fx_fusion_pass"] = False

    import torch_xla as txla
    from infra.utilities import compile_torch_workload_for_tt_device

    if tester.compiler_config is not None:
        txla.set_custom_compile_options(
            tester.compiler_config.to_torch_compile_options()
        )
    compile_torch_workload_for_tt_device(
        workload=workload, torch_options=compile_options
    )

    tt_res = tester.device_runner.run_on_tt_device(workload)
    tt_res = unpack_forward_output(tt_res)

    # Force graph break
    torch_xla.sync(wait=True)

    # Run backward on TT
    tt_backward_workload = Workload(
        framework=framework,
        executable=tt_res.backward,
        args=[],
        kwargs={"gradient": random_grad},
    )
    tester.device_runner.run_on_tt_device(tt_backward_workload)

    if parallelism == Parallelism.TENSOR_PARALLEL:
        _mark_gradient_sharding(model, workload)

    # Sync gradients
    wanted_grads = [p.grad for p in model.parameters() if p.grad is not None]
    torch_xla._XLAC._xla_sync_multi(
        wanted_grads,
        list(set([p.device.type for p in wanted_grads])),
        wait=True,
    )
    tt_grads, tt_none_grads = _extract_grads(model)

    assert (
        cpu_none_grads == tt_none_grads
    ), f"CPU and TT have different None grad parameters: {cpu_none_grads} != {tt_none_grads}"
    logger.warning(f"Grads: {cpu_none_grads} are None")

    forward_result = tester.evaluator.evaluate(tt_res, cpu_res)
    backward_result = tester.evaluator.evaluate(tt_grads, cpu_grads)

    return backward_result, forward_result


def _extract_grads(
    model: torch.nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
    """Extracts gradients from a model."""
    existing_grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }
    none_grads = set(
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    )
    return existing_grads, none_grads


def _mark_gradient_sharding(model: torch.nn.Module, workload: TorchWorkload):
    """Apply sharding to gradients based on parameter shard specs."""
    assert workload.shard_spec_fn is not None
    assert workload.mesh is not None

    shard_specs = workload.shard_spec_fn(model)
    assert shard_specs is not None

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param not in shard_specs:
            logger.warning(f"Parameter {name} not found in shard specs")
            continue
        shard_spec = shard_specs[param]
        param.grad = sharding_constraint_tensor(param.grad, workload.mesh, shard_spec)


def _cast_tensors_to_dtype(obj, dtype):
    """Recursively cast float tensors in a nested structure to the given dtype."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point:
            return obj.to(dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        cast_items = [_cast_tensors_to_dtype(item, dtype) for item in obj]
        return type(obj)(cast_items)
    elif isinstance(obj, dict):
        return {key: _cast_tensors_to_dtype(value, dtype) for key, value in obj.items()}
    else:
        return obj
