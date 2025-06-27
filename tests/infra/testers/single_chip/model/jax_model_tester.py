# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Mapping, Optional, Sequence

import jax
from flax import linen, nnx
from infra.comparators import ComparisonConfig
from infra.utilities import Framework, Model, PyTree
from infra.workloads import JaxWorkload, Workload, WorkloadFactory
from op_by_op_infra.pydantic_models import OpTest
from op_by_op_infra.workflow import run_op_by_op_workflow
from transformers.modeling_flax_utils import FlaxPreTrainedModel

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

    # -------------------- Protected methods --------------------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        # Placeholders for objects that will be set during
        # `_initialize_all_components`. Easier to spot if located in constructor instead
        # of dynamically creating them somewhere in methods.
        self._input_activations: Dict | Sequence[Any] = None
        self._input_parameters: PyTree = None

        super().__init__(comparison_config, run_mode, Framework.JAX)

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

    def _get_input_parameters(self) -> PyTree:
        """
        Returns input parameters.

        By default returns existing model parameters for the HF FlaxPreTrainedModel.
        """
        if isinstance(self._model, FlaxPreTrainedModel):
            assert hasattr(self._model, "params")
            return self._model.params

        raise NotImplementedError("Subclasses must implement this method.")

    # --- Overrides ---

    # @override
    def _get_forward_method_args(self) -> Sequence[Any]:
        """
        Returns positional arguments for model's forward pass.

        By default returns input parameters and activations for the Flax linen models,
        and empty list for other type of models.
        """
        if isinstance(self._model, linen.Module):
            return [self._input_parameters, self._input_activations]

        return []

    # @override
    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        """
        Returns keyword arguments for model's forward pass.

        By default returns input parameters and activations for the HF
        FlaxPreTrainedModel, and empty dict for other type of models.
        """
        if isinstance(self._model, FlaxPreTrainedModel):
            return {
                "params": self._input_parameters,
                **self._input_activations,
            }

        return {}

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()
        forward_static_args = self._get_static_argnames()
        forward_method_name = self._get_forward_method_name()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"
        assert hasattr(
            self._model, forward_method_name
        ), f"Model does not have {forward_method_name} method provided."

        forward_pass_method = getattr(self._model, forward_method_name)

        self._workload = WorkloadFactory.create_workload(
            self._framework,
            executable=forward_pass_method,
            model=self._model,
            args=args,
            kwargs=kwargs,
            static_argnames=forward_static_args,
        )

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations = self._get_input_activations()
        self._input_parameters = self._get_input_parameters()

    # @override
    @staticmethod
    def _configure_model_for_inference(model: Model) -> None:
        assert isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(model, nnx.Module):
            # TODO find another way to do this since model.eval() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        model.eval()

    # @override
    @staticmethod
    def _configure_model_for_training(model: Model) -> None:
        assert isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel))

        if not isinstance(model, nnx.Module):
            # TODO find another way to do this since model.train() does not exist, maybe
            # by passing train param as kwarg to __call__.
            return

        model.train()

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model's forward pass into optimized kernels."""
        assert isinstance(workload, JaxWorkload)

        workload.executable = jax.jit(
            workload.executable, static_argnames=workload.static_argnames
        )
        return workload

    # @override
    def _test_inference_op_by_op(
        self,
        compile_before_split: bool = False,
        compile_each_submodule_after_split: bool = False,
        *,
        frontend: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> List[OpTest]:
        compiled_workload = self._compile(self._workload)
        assert isinstance(compiled_workload, JaxWorkload)

        return run_op_by_op_workflow(
            module=compiled_workload.as_mlir_module_str(),
            compile_before_split=compile_before_split,
            compile_each_submodule_after_split=compile_each_submodule_after_split,
            frontend=frontend,
            model_name=model_name,
        )
