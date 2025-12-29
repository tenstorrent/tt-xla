# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen, nnx
from huggingface_hub import snapshot_download
from infra.comparators import ComparisonConfig
from infra.utilities import (
    Framework,
    Model,
    PyTree,
    compile_jax_workload_for_cpu,
    compile_jax_workload_for_tt_device,
    random_tensor,
)
from infra.workloads import Workload
from loguru import logger
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from tests.infra.testers.compiler_config import CompilerConfig

from .model_tester import ModelTester, RunMode


class JaxModelTester(ModelTester):
    """
    Abstract base class all single chip `jax` model testers must inherit.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations(self) -> Sequence[Any]
    _get_forward_method_name(self) -> str # Optional, has default behaviour.
    _get_static_argnames(self) -> Sequence[str] # Optional, has default behaviour.
    _get_input_parameters(self) -> PyTree # Optional, has default behaviour.
    # One of or both:
    _get_forward_method_args(self) -> Sequence[Any] # Optional, has default behaviour.
    _get_forward_method_kwargs(self) -> Mapping[str, Any] # Optional, has default behaviour.
    ```
    """

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
        has_batch_norm: bool = False,
        dtype_override=None,
    ) -> None:

        self._input_activations: Dict | Sequence[Any] = None
        self._input_parameters: PyTree = None
        self._has_batch_norm = has_batch_norm

        super().__init__(
            comparison_config, run_mode, Framework.JAX, compiler_config, dtype_override
        )

    # @override
    def _configure_model_for_inference(self) -> None:
        assert isinstance(self._model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(self._model, nnx.Module):
            # TODO find another way to do this since model.eval() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        self._model.eval()

    # @override
    def _configure_model_for_training(self) -> None:
        assert isinstance(self._model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(self._model, nnx.Module):
            # TODO find another way to do this since model.train() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        self._model.train()

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations = self._get_input_activations()
        self._input_parameters = self._get_input_parameters()

    def _get_input_parameters(self) -> PyTree:
        """
        Returns input parameters.

        By default returns existing model parameters for the HF FlaxPreTrainedModel.
        """

        if isinstance(self._model, FlaxPreTrainedModel):
            assert hasattr(self._model, "params")
            return self._model.params
        elif isinstance(self._model, nnx.Module):
            return nnx.split(self._model)[1]

        raise NotImplementedError("Subclasses must implement this method.")

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()
        forward_static_args = self._get_static_argnames()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"

        if isinstance(self._model, nnx.Module):
            graphdef = nnx.split(self._model)[0]

            def forward_pass_method(state, inputs):
                model_ = nnx.merge(graphdef, state)
                return model_(inputs)

        else:
            forward_method_name = self._get_forward_method_name()
            assert hasattr(
                self._model, forward_method_name
            ), f"Model does not have method {forward_method_name}"

            forward_pass_method = getattr(self._model, forward_method_name)

        self._workload = Workload(
            framework=self._framework,
            executable=forward_pass_method,
            args=args,
            kwargs=kwargs,
            static_argnames=forward_static_args,
        )

    def _get_forward_method_args(self) -> Sequence[Any]:
        """
        Returns positional arguments for model's forward pass.

        By default returns input parameters and activations for the Flax linen models,
        and empty list for other type of models.
        """
        if isinstance(self._model, (linen.Module, nnx.Module)):
            return [self._input_parameters, self._input_activations]

        return []

    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Returns keyword arguments for model's forward pass.

        By default returns input parameters and activations for the HF
        FlaxPreTrainedModel, and empty dict for other type of models.
        """
        kwargs = {}
        if isinstance(self._model, FlaxPreTrainedModel):
            kwargs = {
                "params": self._input_parameters,
                **self._input_activations,
            }

            # Only add 'deterministic' if the model accepts it
            try:
                sig = inspect.signature(self._model.__call__)
                if "deterministic" in sig.parameters:
                    # deterministic=True means inference (no dropout), deterministic=False means training
                    kwargs["deterministic"] = (
                        True if self._run_mode == RunMode.INFERENCE else False
                    )
                if "train" in sig.parameters:
                    kwargs["train"] = (
                        False if self._run_mode == RunMode.INFERENCE else True
                    )
            except:
                pass
        elif isinstance(self._model, nnx.Module):
            pass
        else:
            kwargs = {"train": False if self._run_mode == RunMode.INFERENCE else True}
        if self._run_mode == RunMode.TRAINING and self._has_batch_norm:
            kwargs["mutable"] = ("batch_stats",)
        return kwargs

    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        """
        Returns names of arguments which should be treated as static by JIT compiler.

        Static arguments are those which are not replaced with Tracer objects by the JIT
        but rather are used as is, which is needed if control flow or shapes depend on
        them. See:
        https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables

        By default no arguments are static.
        """
        static_argnames = []
        sig = inspect.signature(self._model.__call__)
        if "train" in sig.parameters:
            static_argnames.append("train")
        if "deterministic" in sig.parameters:
            static_argnames.append("deterministic")
        if self._run_mode == RunMode.TRAINING and self._has_batch_norm:
            static_argnames.append("mutable")
        return static_argnames

    def _compile_for_tt_device(self, workload: Workload) -> None:
        """Compile JAX workload for TT device."""
        compile_jax_workload_for_tt_device(
            workload, self._compiler_config.to_jax_compiler_options()
        )

    def _compile_for_cpu(self, workload: Workload) -> None:
        """Compile JAX workload for CPU."""
        compile_jax_workload_for_cpu(workload)

    def _wrapper_model(self, f):
        def model(args, kwargs):
            out = f(*args, **kwargs)
            if self._has_batch_norm and self._run_mode == RunMode.TRAINING:
                out = out[0]
            if isinstance(self._model, FlaxPreTrainedModel):
                out = out.logits
            return out

        return model

    # @override
    def _test_training(self):
        """
        Steps:
        1. Create partial with static args
        2. Compile workloads for CPU and TT device
        3. Create partial with vjp of model
        4. Run forward on CPU and TT device
        5. Create random gradient with same shape as output
        6. Run pullback on CPU and TT device
        7. Compare forward results and gradients
        """

        # Wrapper to convert kwargs to args and return logits if model is HF
        is_hf_model = isinstance(self._model, FlaxPreTrainedModel)

        # Create partial with static args
        partial_executable = jax.tree_util.Partial(
            self._workload.executable,
            **{k: self._workload.kwargs[k] for k in self._workload.static_argnames},
        )
        training_workload = Workload(
            framework=self._framework,
            executable=partial_executable,
            args=self._workload.args,
            kwargs={
                k: self._workload.kwargs[k]
                for k in self._workload.kwargs
                if k not in self._workload.static_argnames
            },
            static_argnames=[],
        )

        # Compile workloads for CPU with vjp of model
        self._compile_for_cpu(training_workload)
        train_fwd_cpu = Workload(
            framework=self._framework,
            executable=jax.tree_util.Partial(
                jax.vjp,
                self._wrapper_model(training_workload.compiled_executable),
            ),
            args=[training_workload.args, training_workload.kwargs],
        )
        cpu_forward_out, cpu_pullback = self._run_on_cpu(train_fwd_cpu)

        # Compile workloads for TT device with vjp of model
        self._compile_for_tt_device(training_workload)
        train_fwd_tt = Workload(
            framework=self._framework,
            executable=jax.tree_util.Partial(
                jax.vjp,
                self._wrapper_model(training_workload.compiled_executable),
            ),
            args=[training_workload.args, training_workload.kwargs],
        )
        tt_forward_out, tt_pullback = self._run_on_tt_device(train_fwd_tt)

        # Create random gradient with same shape as output
        random_grad = random_tensor(
            cpu_forward_out.shape,
            dtype=cpu_forward_out.dtype,
            framework=self._framework,
        )

        # Run pullback on CPU
        pullback_workload_cpu = Workload(
            framework=self._framework,
            executable=cpu_pullback,
            args=[random_grad],
        )
        grads_cpu = self._run_on_cpu(pullback_workload_cpu)

        # Run pullback on TT device
        pullback_workload_tt = Workload(
            framework=self._framework,
            executable=tt_pullback,
            args=[random_grad],
        )
        grads_tt = self._run_on_tt_device(pullback_workload_tt)

        # Compare forward results and gradients
        forward_comparison = self._compare(tt_forward_out, cpu_forward_out)
        gradients_comparison = self._compare(grads_tt, grads_cpu)

        return (gradients_comparison, forward_comparison)

    # @override
    def _apply_model_dtype(self) -> None:
        """Applies dtype_override to the model parameters."""
        # assert that dtype is a floating point dtype
        assert jnp.issubdtype(
            self._dtype_override, jnp.floating
        ), "Dtype override must be floating point"
        # For JAX models, we typically cast the parameters rather than the model itself
        if hasattr(self._model, "params"):
            self._model.params = self._cast_pytree_to_dtype(
                self._model.params, self._dtype_override
            )
        elif isinstance(self._model, nnx.Module):
            # For NNX modules, cast the state
            state = nnx.state(self._model)
            casted_state = self._cast_pytree_to_dtype(state, self._dtype_override)
            nnx.update(self._model, casted_state)

    # @override
    def _apply_inputs_dtype(self) -> None:
        """Applies dtype_override to inputs, only casting float tensors."""
        # assert that dtype is a floating point dtype
        assert jnp.issubdtype(
            self._dtype_override, jnp.floating
        ), "Dtype override must be floating point"
        self._input_activations = self._cast_pytree_to_dtype(
            self._input_activations, self._dtype_override
        )
        if self._input_parameters is not None:
            self._input_parameters = self._cast_pytree_to_dtype(
                self._input_parameters, self._dtype_override
            )

    def _cast_pytree_to_dtype(self, pytree, dtype):
        """Recursively cast floating-point array leaves in a JAX pytree to `dtype`."""

        def cast_leaf(x):
            # Works for jax.Array, numpy.ndarray, and other array-likes with dtype
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        return jax.tree_util.tree_map(cast_leaf, pytree)
