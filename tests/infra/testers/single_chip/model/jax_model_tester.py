# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import linen, nnx
from infra.evaluators import ComparisonConfig
from infra.utilities import (
    Framework,
    Model,
    PyTree,
    compile_jax_workload_for_cpu,
    compile_jax_workload_for_tt_device,
    random_tensor,
)
from infra.workloads import Workload
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
            return

        self._model.eval()

    # @override
    def _configure_model_for_training(self) -> None:
        assert isinstance(self._model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(self._model, nnx.Module):
            return

        self._model.train()

    # @override
    def _cache_model_inputs(self) -> None:
        self._input_activations = self._get_input_activations()
        self._input_parameters = self._get_input_parameters()

    def _get_input_parameters(self) -> PyTree:
        if isinstance(self._model, FlaxPreTrainedModel):
            assert hasattr(self._model, "params")
            return self._model.params
        elif isinstance(self._model, nnx.Module):
            return nnx.split(self._model)[1]

        raise NotImplementedError("Subclasses must implement this method.")

    # @override
    def _initialize_workload(self) -> None:
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()
        forward_static_args = self._get_static_argnames()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), "Forward method args or kwargs or both must be provided"

        if isinstance(self._model, nnx.Module):
            graphdef = nnx.split(self._model)[0]

            def forward_pass_method(params, **inputs):
                model_ = nnx.merge(graphdef, params)
                return model_(**inputs)

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
        if isinstance(self._model, linen.Module):
            return [self._input_parameters, self._input_activations]
        return []

    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        kwargs = {}
        if isinstance(self._model, (FlaxPreTrainedModel, nnx.Module)):
            kwargs = {
                "params": self._input_parameters,
                **self._input_activations,
            }

            try:
                sig = inspect.signature(self._model.__call__)
                if "deterministic" in sig.parameters:
                    kwargs["deterministic"] = (
                        True if self._run_mode == RunMode.INFERENCE else False
                    )
                if "train" in sig.parameters:
                    kwargs["train"] = (
                        False if self._run_mode == RunMode.INFERENCE else True
                    )
            except:
                pass
        else:
            kwargs = {"train": False if self._run_mode == RunMode.INFERENCE else True}
        if self._run_mode == RunMode.TRAINING and self._has_batch_norm:
            kwargs["mutable"] = ("batch_stats",)
        return kwargs

    def _get_static_argnames(self) -> Optional[Sequence[str]]:
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
        compile_jax_workload_for_tt_device(
            workload, self._compiler_config.to_jax_compiler_options()
        )

    def _compile_for_cpu(self, workload: Workload) -> None:
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
        is_hf_model = isinstance(self._model, FlaxPreTrainedModel)

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

        random_grad = random_tensor(
            cpu_forward_out.shape,
            dtype=cpu_forward_out.dtype,
            framework=self._framework,
        )

        pullback_workload_cpu = Workload(
            framework=self._framework,
            executable=cpu_pullback,
            args=[random_grad],
        )
        grads_cpu = self._run_on_cpu(pullback_workload_cpu)

        pullback_workload_tt = Workload(
            framework=self._framework,
            executable=tt_pullback,
            args=[random_grad],
        )
        grads_tt = self._run_on_tt_device(pullback_workload_tt)

        forward_comparison = self._compare(tt_forward_out, cpu_forward_out)
        gradients_comparison = self._compare(grads_tt, grads_cpu)

        return (gradients_comparison, forward_comparison)

    # @override
    def _apply_model_dtype(self) -> None:
        assert jnp.issubdtype(
            self._dtype_override, jnp.floating
        ), "Dtype override must be floating point"
        if hasattr(self._model, "params"):
            self._model.params = self._cast_pytree_to_dtype(
                self._model.params, self._dtype_override
            )
        elif isinstance(self._model, nnx.Module):
            state = nnx.state(self._model)
            casted_state = self._cast_pytree_to_dtype(state, self._dtype_override)
            nnx.update(self._model, casted_state)

    # @override
    def _apply_inputs_dtype(self) -> None:
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
        def cast_leaf(x):
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        return jax.tree_util.tree_map(cast_leaf, pytree)
