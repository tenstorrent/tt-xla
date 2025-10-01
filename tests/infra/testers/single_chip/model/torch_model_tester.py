# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Any, Dict, Mapping, Sequence, Callable

import torch
import torch_xla
import torch_xla.runtime as xr

from infra.comparators import ComparisonConfig
from tests.infra.testers.compiler_config import CompilerConfig
from infra.utilities import Framework
from infra.workloads import Workload
import os

from .model_tester import ModelTester, RunMode


# class FunctionModule(torch.nn.Module):
#     """A wrapper to convert a function into a torch.nn.Module."""

#     def __init__(self, func: Callable):
#         super().__init__()
#         self.func = func

#     def forward(self, *args, **kwargs):
#         return self.func(*args, **kwargs)


class TorchModelTester(ModelTester):
    """
    Abstract base class all single chip `torch` model testers must inherit.

    Derived classes must provide implementations of:
    ```
    _get_model(self) -> Model
    _get_input_activations(self) -> Sequence[Any]
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

        super().__init__(comparison_config, run_mode, Framework.TORCH, compiler_config)
        # Set custom compile options if provided.
        # Use explicit API for passing compiler options.
        if compiler_config is not None:
            torch_xla.set_custom_compile_options(
                compiler_config.to_torch_compile_options()
            )

    def _ensure_is_executable(self) -> None:
        """Ensures that self._model is a torch.nn.Module, wrapping it if necessary."""

        assert isinstance(self._model, torch.nn.Module) or callable(
            self._model
        ), f"Model must be either a torch.nn.Module or a callable, got {type(self._model)}"

        # if not isinstance(self._model, torch.nn.Module):
        #     self._model = FunctionModule(self._model)

    # @override
    def _configure_model_for_inference(self) -> None:
        self._ensure_is_executable()
        if isinstance(self._model, torch.nn.Module):
            self._model.eval()

    # @override
    def _configure_model_for_training(self) -> None:
        self._ensure_is_executable()
        if isinstance(self._model, torch.nn.Module):
            self._model.train()

    # @override
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        self._input_activations = self._get_input_activations()

    # @override
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        # Prepack model's forward pass and its arguments into a `Workload.`
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), f"Forward method args or kwargs or both must be provided"

        self._workload = Workload(
            framework=self._framework, executable=self._model, args=args, kwargs=kwargs
        )
        self._workload.mesh = self._get_mesh()
        self._workload.shard_spec_fn = self._get_shard_specs_function()

        self._enable_xla_spmd_if_needed()

    # If model has shard specs and running on multichip mesh, then convert StableHLO
    # to Shardy dialect and initialize XLA SPMD runtime.
    def _enable_xla_spmd_if_needed(self) -> None:
        has_shard_specs = self._workload.shard_spec_fn is not None
        is_multichip = self._workload.mesh and len(self._workload.mesh.device_ids) > 1

        if has_shard_specs and is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()

    # @override
    def _get_forward_method_args(self) -> Sequence[Any]:
        if isinstance(self._input_activations, torch.Tensor):
            return [self._input_activations]
        if isinstance(self._input_activations, (tuple, list)):
            return list(self._input_activations)
        return []

    # @override
    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        if isinstance(self._input_activations, collections.abc.Mapping):
            return {**self._input_activations}
        return {}

    # @override
    def _compile_for_cpu(self, workload: Workload) -> None:
        """Compiles `workload` for CPU."""
        self._compile(workload)

    def _compile(self, workload: Workload) -> None:
        """JIT-compiles model's forward pass into optimized kernels.

        Compiles for inductor backend by default.
        """
        self._compile_for_backend(workload, backend="inductor")

    # @override
    def _compile_for_tt_device(self, workload: Workload) -> None:
        """Compiles `workload` for TT device."""
        self._compile_for_backend(workload, backend="tt")

    def _compile_for_backend(self, workload: Workload, backend: str) -> None:
        """JIT-compiles model into optimized kernels."""
        assert workload.is_torch and workload.executable is not None

        workload.compiled_executable = torch.compile(workload.executable, backend=backend)
