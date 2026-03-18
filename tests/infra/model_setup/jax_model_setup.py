# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""JAX model lifecycle helpers extracted from JaxModelTester.

These functions handle model configuration, dtype casting, workload creation,
and training flows for JAX models. They are pure functions that take explicit
parameters instead of reading from self._* attributes.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen, nnx
from infra.evaluators import ComparisonResult
from infra.utilities import Framework, Model, PyTree, random_tensor
from infra.workloads import Workload
from transformers.modeling_flax_utils import FlaxPreTrainedModel


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


def configure_jax_model(model: Model, run_mode: RunMode) -> None:
    """Configures a JAX model for inference or training mode."""
    assert isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

    if not isinstance(model, nnx.Module):
        return

    if run_mode == RunMode.INFERENCE:
        model.eval()
    else:
        model.train()


def cast_jax_model_dtype(model: Model, dtype) -> None:
    """Applies dtype override to JAX model parameters in-place."""
    assert jnp.issubdtype(dtype, jnp.floating), "Dtype override must be floating point"
    if hasattr(model, "params"):
        model.params = _cast_pytree_to_dtype(model.params, dtype)
    elif isinstance(model, nnx.Module):
        state = nnx.state(model)
        casted_state = _cast_pytree_to_dtype(state, dtype)
        nnx.update(model, casted_state)


def cast_jax_inputs_dtype(input_activations, dtype, input_parameters=None):
    """Applies dtype override to JAX inputs, returning new values."""
    assert jnp.issubdtype(dtype, jnp.floating), "Dtype override must be floating point"
    casted_activations = _cast_pytree_to_dtype(input_activations, dtype)
    casted_params = None
    if input_parameters is not None:
        casted_params = _cast_pytree_to_dtype(input_parameters, dtype)
    return casted_activations, casted_params


def get_jax_input_parameters(model: Model) -> PyTree:
    """Returns input parameters for a JAX model."""
    if isinstance(model, FlaxPreTrainedModel):
        assert hasattr(model, "params")
        return model.params
    elif isinstance(model, nnx.Module):
        return nnx.split(model)[1]
    raise NotImplementedError(
        "Model type not supported. Provide input_parameters explicitly."
    )


def get_jax_forward_method_args(
    model: Model, input_parameters: PyTree, input_activations
) -> Sequence[Any]:
    """Returns positional arguments for model's forward pass."""
    if isinstance(model, linen.Module):
        return [input_parameters, input_activations]
    return []


def get_jax_forward_method_kwargs(
    model: Model,
    input_parameters: PyTree,
    input_activations,
    run_mode: RunMode,
    has_batch_norm: bool = False,
) -> Mapping[str, Any]:
    """Returns keyword arguments for model's forward pass."""
    kwargs = {}
    if isinstance(model, (FlaxPreTrainedModel, nnx.Module)):
        kwargs = {
            "params": input_parameters,
            **input_activations,
        }
        try:
            sig = inspect.signature(model.__call__)
            if "deterministic" in sig.parameters:
                kwargs["deterministic"] = run_mode == RunMode.INFERENCE
            if "train" in sig.parameters:
                kwargs["train"] = run_mode == RunMode.TRAINING
        except Exception:
            pass
    else:
        kwargs = {"train": run_mode == RunMode.TRAINING}
    if run_mode == RunMode.TRAINING and has_batch_norm:
        kwargs["mutable"] = ("batch_stats",)
    return kwargs


def get_jax_static_argnames(
    model: Model, run_mode: RunMode, has_batch_norm: bool = False
) -> Optional[Sequence[str]]:
    """Returns names of arguments which should be treated as static by JIT compiler."""
    static_argnames = []
    sig = inspect.signature(model.__call__)
    if "train" in sig.parameters:
        static_argnames.append("train")
    if "deterministic" in sig.parameters:
        static_argnames.append("deterministic")
    if run_mode == RunMode.TRAINING and has_batch_norm:
        static_argnames.append("mutable")
    return static_argnames


def create_jax_model_workload(
    model: Model,
    input_activations,
    input_parameters: PyTree = None,
    run_mode: RunMode = RunMode.INFERENCE,
    forward_method_name: str = None,
    forward_method_args: Sequence[Any] = None,
    forward_method_kwargs: Mapping[str, Any] = None,
    static_argnames: Sequence[str] = None,
    has_batch_norm: bool = False,
    dtype_override=None,
) -> Workload:
    """Creates a JAX Workload from a model and its inputs.

    This replaces the model lifecycle that was spread across JaxModelTester.__init__,
    _initialize_components, _cache_model_inputs, _initialize_workload, etc.
    """
    # Configure model
    configure_jax_model(model, run_mode)

    # Apply dtype overrides
    if dtype_override is not None:
        cast_jax_model_dtype(model, dtype_override)
        input_activations, input_parameters = cast_jax_inputs_dtype(
            input_activations, dtype_override, input_parameters
        )

    # Get input parameters if not provided
    if input_parameters is None:
        input_parameters = get_jax_input_parameters(model)

    # Get forward method args/kwargs if not provided
    if forward_method_args is None:
        forward_method_args = get_jax_forward_method_args(
            model, input_parameters, input_activations
        )
    if forward_method_kwargs is None:
        forward_method_kwargs = get_jax_forward_method_kwargs(
            model, input_parameters, input_activations, run_mode, has_batch_norm
        )
    if static_argnames is None:
        static_argnames = get_jax_static_argnames(model, run_mode, has_batch_norm)

    assert (
        len(forward_method_args) > 0 or len(forward_method_kwargs) > 0
    ), "Forward method args or kwargs or both must be provided"

    # Build executable
    if isinstance(model, nnx.Module):
        graphdef = nnx.split(model)[0]

        def forward_pass_method(params, **inputs):
            model_ = nnx.merge(graphdef, params)
            return model_(**inputs)

    else:
        method_name = forward_method_name or (
            "apply" if isinstance(model, linen.Module) else "__call__"
        )
        assert hasattr(model, method_name), f"Model does not have method {method_name}"
        forward_pass_method = getattr(model, method_name)

    return Workload(
        framework=Framework.JAX,
        executable=forward_pass_method,
        args=forward_method_args,
        kwargs=forward_method_kwargs,
        static_argnames=static_argnames,
    )


def run_jax_training(
    tester,
    workload: Workload,
    wrapper_model_fn=None,
    has_batch_norm: bool = False,
    model=None,
) -> tuple[ComparisonResult, ...]:
    """Runs JAX training flow with VJP.

    Args:
        tester: Tester instance (used for compile, run, compare).
        workload: The model workload.
        wrapper_model_fn: Optional function that wraps the compiled executable.
            If None, a default wrapper is used.
        has_batch_norm: Whether model has batch norm (affects output unpacking).
        model: The original model (needed for default wrapper if FlaxPreTrainedModel).
    """
    framework = tester.framework

    # Default wrapper
    if wrapper_model_fn is None:

        def wrapper_model_fn(f):
            def wrapped(args, kwargs):
                out = f(*args, **kwargs)
                if has_batch_norm:
                    out = out[0]
                if model is not None and isinstance(model, FlaxPreTrainedModel):
                    out = out.logits
                return out

            return wrapped

    # Create partial with static args
    partial_executable = jax.tree_util.Partial(
        workload.executable,
        **{k: workload.kwargs[k] for k in workload.static_argnames},
    )
    training_workload = Workload(
        framework=framework,
        executable=partial_executable,
        args=workload.args,
        kwargs={
            k: workload.kwargs[k]
            for k in workload.kwargs
            if k not in workload.static_argnames
        },
        static_argnames=[],
    )

    # Compile and run forward + vjp on CPU
    tester._compile_for_cpu(training_workload)
    train_fwd_cpu = Workload(
        framework=framework,
        executable=jax.tree_util.Partial(
            jax.vjp,
            wrapper_model_fn(training_workload.compiled_executable),
        ),
        args=[training_workload.args, training_workload.kwargs],
    )
    cpu_forward_out, cpu_pullback = tester.device_runner.run_on_cpu(train_fwd_cpu)

    # Compile and run forward + vjp on TT device
    tester._compile_for_tt_device(training_workload)
    train_fwd_tt = Workload(
        framework=framework,
        executable=jax.tree_util.Partial(
            jax.vjp,
            wrapper_model_fn(training_workload.compiled_executable),
        ),
        args=[training_workload.args, training_workload.kwargs],
    )
    tt_forward_out, tt_pullback = tester.device_runner.run_on_tt_device(train_fwd_tt)

    # Create random gradient with same shape as output
    random_grad = random_tensor(
        cpu_forward_out.shape,
        dtype=cpu_forward_out.dtype,
        framework=framework,
    )

    # Run pullback on CPU
    pullback_workload_cpu = Workload(
        framework=framework,
        executable=cpu_pullback,
        args=[random_grad],
    )
    grads_cpu = tester.device_runner.run_on_cpu(pullback_workload_cpu)

    # Run pullback on TT device
    pullback_workload_tt = Workload(
        framework=framework,
        executable=tt_pullback,
        args=[random_grad],
    )
    grads_tt = tester.device_runner.run_on_tt_device(pullback_workload_tt)

    # Compare forward results and gradients
    forward_comparison = tester.evaluator.evaluate(tt_forward_out, cpu_forward_out)
    gradients_comparison = tester.evaluator.evaluate(grads_tt, grads_cpu)

    return (gradients_comparison, forward_comparison)


def _cast_pytree_to_dtype(pytree, dtype):
    """Recursively cast floating-point array leaves in a JAX pytree to dtype."""

    def cast_leaf(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(cast_leaf, pytree)
