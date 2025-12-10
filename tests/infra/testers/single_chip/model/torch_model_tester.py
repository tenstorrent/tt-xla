# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import collections
import os
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
from infra.comparators import ComparisonConfig
from infra.utilities import Framework
from infra.workloads import TorchWorkload, Workload
from loguru import logger

from tests.infra.comparators.comparator import ComparisonResult
from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.config import Parallelism

from .model_tester import ModelTester, RunMode


@contextmanager
def _mask_jax_accelerator():
    """Temporarily hide jax accelerator to avoid inductor issues with no-tensor-input graphs.

    When torchax is imported (via torch_xla's mark_sharding), it registers 'jax' as a PyTorch
    accelerator. This causes inductor to fail when compiling graphs with no tensor inputs,
    as it tries to call torch.accelerator.current_device_index() which isn't supported for jax.
    """
    original_fn = torch.accelerator.is_available

    def masked_is_available():
        try:
            acc = torch.accelerator.current_accelerator()
            # current_accelerator() returns device(type='jax'), need to check .type
            if acc.type == "jax":
                return False
        except RuntimeError:
            pass
        return original_fn()

    torch.accelerator.is_available = masked_is_available
    try:
        yield
    finally:
        torch.accelerator.is_available = original_fn


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
            assert (
                self._workload.shard_spec_fn is not None
            ), "Tensor parallel requires shard specs function"
            assert (
                self._workload.mesh and len(self._workload.mesh.device_ids) > 1
            ), "Tensor parallel requires multi-chip mesh"

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
    def _run_on_cpu(self, compiled_workload: Workload) -> torch.Tensor:
        """Runs workload on CPU with jax accelerator masked.

        Uses _mask_jax_accelerator because torch.compile with inductor is lazy -
        actual compilation happens during execution, not during torch.compile() call.
        """
        with _mask_jax_accelerator():
            return super()._run_on_cpu(compiled_workload)

    # @override
    def _compile_for_tt_device(self, workload: Workload) -> None:
        """Compiles `workload` for TT device."""
        self._compile_for_backend(workload, backend="tt")

    def _compile_for_backend(self, workload: Workload, backend: str) -> None:
        """JIT-compiles model into optimized kernels."""
        assert workload.is_torch and workload.model is not None

        workload.compiled_executable = torch.compile(workload.model, backend=backend)

    def _unpack_forward_output(self, output: Any) -> torch.Tensor:
        """
        Unwraps model output to a single tensor.
        In base case, we assume the output is a single tensor.
        """
        return output

    def _extract_grads(
        self, model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
        """
        Extracts gradients from a model and returns a dictionary of gradients and a dictionary of None gradients.
        """
        # TODO: Right now, we only extract gradients for parameters that have a gradient.
        # In the future, we should extract gradients for all parameters that require grad is True.
        #
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

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        # Run forward on CPU
        # TODO: Needs further investigation https://github.com/tenstorrent/tt-xla/issues/1391
        # self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)
        cpu_res = self._unpack_forward_output(cpu_res)

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

        cpu_grads, cpu_none_grads = self._extract_grads(self._model)
        self._workload.model.zero_grad()

        # Run forward on TT
        # TODO: Needs further investigation https://github.com/tenstorrent/tt-xla/issues/1391
        # self._compile_for_tt_device(self._workload)
        tt_res = self._run_on_tt_device(self._workload)
        tt_res = self._unpack_forward_output(tt_res)

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
        # TODO: Adding explicit sync to ensure view of gradients are not computed without reason
        # https://github.com/tenstorrent/tt-xla/issues/1466
        wanted_grads = [p.grad for p in self._model.parameters() if p.grad is not None]
        torch_xla._XLAC._xla_sync_multi(
            wanted_grads,
            list(set([p.device.type for p in wanted_grads])),
            wait=True,
        )
        tt_grads, tt_none_grads = self._extract_grads(self._model)

        assert (
            cpu_none_grads == tt_none_grads
        ), f"CPU and TT have different None grad parameters: {cpu_none_grads} != {tt_none_grads}"
        logger.warning(f"Grads: {cpu_none_grads} are None")

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
