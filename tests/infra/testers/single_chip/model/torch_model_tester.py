# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import collections
import os
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch_xla
import torch_xla.runtime as xr
from infra.comparators import ComparisonConfig
from infra.utilities import Framework
from infra.workloads import TorchWorkload, Workload

from tests.infra.comparators.comparator import ComparisonResult
from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.config import Parallelism

from .model_tester import ModelTester, RunMode


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
        parallelism=None,
        dtype_override=None,
    ) -> None:

        self._input_activations: Dict | Sequence[Any] = None
        self._parallelism = parallelism

        super().__init__(
            comparison_config,
            run_mode,
            Framework.TORCH,
            compiler_config,
            dtype_override,
        )
        # Set custom compile options if provided.
        # Use explicit API for passing compiler options.
        if compiler_config is not None:
            torch_xla.set_custom_compile_options(
                compiler_config.to_torch_compile_options()
            )

    # @override
    def _configure_model(self) -> None:
        self._device_runner.set_training_mode(self._run_mode == RunMode.TRAINING)
        super()._configure_model()

    # @override
    def _configure_model_for_inference(self) -> None:
        assert isinstance(self._model, torch.nn.Module)
        self._model.eval()

    # @override
    def _configure_model_for_training(self) -> None:
        assert isinstance(self._model, torch.nn.Module)
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

        self._workload = TorchWorkload(
            model=self._model,
            args=args,
            kwargs=kwargs,
            mesh=self._get_mesh(),
            shard_spec_fn=self._get_shard_specs_function(),
        )

        if self._parallelism == Parallelism.TENSOR_PARALLEL:
            self._assert_tensor_parallel_valid()

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

    def _assert_tensor_parallel_valid(self) -> bool:
        """Ensure the workload supports tensor parallelism (multi-chip mesh and shard specs)."""
        fn = self._workload.shard_spec_fn
        assert callable(fn), "Tensor parallel requires shard specs function"

        shard_specs = fn(self._model)
        assert (
            shard_specs
        ), "Tensor parallel requires shard specs function to return non-empty specs"

        mesh = self._workload.mesh
        assert (
            mesh and len(mesh.device_ids) > 1
        ), "Tensor parallel requires multi-chip mesh"

        return True

    def _compile_for_backend(self, workload: Workload, backend: str) -> None:
        """JIT-compiles model into optimized kernels."""
        assert workload.is_torch and workload.model is not None

        workload.model.compile(backend=backend)

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        # Run forward on CPU
        # TODO: Needs further investigation https://github.com/tenstorrent/tt-xla/issues/1391
        # self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)

        # Generate random gradient
        random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)

        # Create and run backward on CPU
        cpu_backward_workload = Workload(
            framework=self._framework,
            executable=cpu_res.backward,
            args=[],
            kwargs={"gradient": random_grad},
        )
        self._run_on_cpu(cpu_backward_workload)

        cpu_grads = {name: p.grad.clone() for name, p in self._model.named_parameters()}
        self._workload.model.zero_grad()

        # Run forward on TT
        # TODO: Needs further investigation https://github.com/tenstorrent/tt-xla/issues/1391
        # self._compile_for_tt_device(self._workload)
        tt_res = self._run_on_tt_device(self._workload)
        # Force graph break so we can differentiate between forward and backward
        torch_xla.sync(wait=True)

        # Run backward on TT
        tt_backward_workload = Workload(
            framework=self._framework,
            executable=tt_res.backward,
            args=[],
            kwargs={"gradient": random_grad},
        )
        self._run_on_tt_device(tt_backward_workload)
        torch_xla.sync(wait=True)

        tt_grads = {
            name: p.grad.cpu().clone() for name, p in self._model.named_parameters()
        }

        forward_result = self._compare(tt_res, cpu_res)
        backward_result = self._compare(tt_grads, cpu_grads)

        # Only the first result is recorded in the report properties,
        # and only want to report on the backward result
        return backward_result, forward_result

    # @override
    def _apply_model_dtype(self) -> None:
        """Applies dtype_override to the model."""
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._dtype_override)
        else:
            raise TypeError("Model does not have 'to' method to apply dtype.")

    # @override
    def _apply_inputs_dtype(self) -> None:
        """Applies dtype_override to inputs, only casting float tensors."""
        self._input_activations = self._cast_tensors_to_dtype(
            self._input_activations, self._dtype_override
        )

    def _cast_tensors_to_dtype(self, obj, dtype):
        """Recursively cast float tensors in a nested structure to the given dtype."""
        if isinstance(obj, torch.Tensor):
            # Only cast floating point tensors, leave integer tensors unchanged
            if obj.dtype.is_floating_point:
                return obj.to(dtype)
            return obj
        elif isinstance(obj, (list, tuple)):
            cast_items = [self._cast_tensors_to_dtype(item, dtype) for item in obj]
            return type(obj)(cast_items)
        elif isinstance(obj, dict):
            return {
                key: self._cast_tensors_to_dtype(value, dtype)
                for key, value in obj.items()
            }
        else:
            return obj
