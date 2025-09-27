# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Mapping, Optional, Sequence

import jax
import os
import shutil

from flax import linen, nnx
from huggingface_hub import snapshot_download
from infra.comparators import ComparisonConfig
from tests.infra.testers.compiler_config import CompilerConfig
from infra.utilities import Framework, Model, PyTree, random_tensor
from infra.workloads import Workload
from loguru import logger
from transformers.modeling_flax_utils import FlaxPreTrainedModel

from .model_tester import ModelTester, RunMode


class JaxModelTester(ModelTester):
    """
    Abstract base class all single chip `jax` model testers must inherit.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations(self) -> Sequence[Any]
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
    ) -> None:

        self._input_activations: Dict | Sequence[Any] = None
        self._input_parameters: PyTree = None

        super().__init__(comparison_config, run_mode, Framework.JAX, compiler_config)

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
        Returns input parameters (nnx models run without passing params as args/kwargs).
        """
        if isinstance(self._model, nnx.Module):
            _, state = nnx.split(self._model)
            return state

        elif isinstance(self._model, (FlaxPreTrainedModel, linen.Module)):
            # Check if params are already attached to model (custom initialization)
            if hasattr(self._model, "params"):
                return self._model.params
            # Otherwise, subclasses must implement parameter initialization
            raise NotImplementedError(
                "Subclasses must implement this method for linen models without pre-attached params."
            )

        raise NotImplementedError(
            "Not supported model type. Supported are nnx.Module, linen.Module and FlaxPreTrainedModel."
        )

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
            graphdef, _ = nnx.split(self._model)

            def forward_pass_method(state, inputs):
                model = nnx.merge(graphdef, state)
                return model(inputs)

        elif isinstance(self._model, FlaxPreTrainedModel):
            forward_pass_method = getattr(self._model, "__call__")

        elif isinstance(self._model, linen.Module):
            forward_pass_method = getattr(self._model, "apply")

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
        """
        if isinstance(self._model, (linen.Module, nnx.Module)):
            return [self._input_parameters, self._input_activations]

        return []

    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Returns keyword arguments for model's forward pass.
        """
        if isinstance(self._model, FlaxPreTrainedModel):
            return {
                "params": self._input_parameters,
                **self._input_activations,
            }

        # TODO: add support for training-specific kwargs for different types of models. https://github.com/tenstorrent/tt-xla/issues/1388
        return {}

    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        """
        Returns names of arguments which should be treated as static by JIT compiler.

        Static arguments are those which are not replaced with Tracer objects by the JIT
        but rather are used as is, which is needed if control flow or shapes depend on
        them. See:
        https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables

        By default no arguments are static.
        """
        return []

    # @override
    def _compile_for_tt_device(self, workload: Workload) -> None:
        """JIT-compiles model's forward pass into optimized kernels."""
        compiler_options = self._compiler_config.to_jax_compiler_options()

        self._jit_compile_workload(workload, compiler_options=compiler_options)

    # @override
    def _compile_for_cpu(self, workload: Workload) -> None:
        """JIT-compiles model's forward pass into optimized kernels."""
        self._jit_compile_workload(workload)

    def _jit_compile_workload(self, workload: Workload, **jit_options) -> None:
        assert workload.is_jax, "Workload must be JAX workload to compile"

        workload.compiled_executable = jax.jit(
            workload.executable, static_argnames=workload.static_argnames, **jit_options
        )

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

        def wrapper_model(f):
            def model(args, kwargs):
                out = f(*args, **kwargs)
                if is_hf_model:
                    out = out.logits
                return out

            return model

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
                jax.vjp, wrapper_model(training_workload.executable)
            ),
            args=[training_workload.args, training_workload.kwargs],
        )
        cpu_forward_out, cpu_pullback = self._run_on_cpu(train_fwd_cpu)

        # Compile workloads for TT device with vjp of model
        self._compile_for_tt_device(training_workload)
        train_fwd_tt = Workload(
            framework=self._framework,
            executable=jax.tree_util.Partial(
                jax.vjp, wrapper_model(training_workload.executable)
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
        self._compare(tt_forward_out, cpu_forward_out)
        self._compare(grads_tt, grads_cpu)
