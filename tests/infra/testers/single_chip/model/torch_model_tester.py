# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import collections
from contextlib import contextmanager
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import torch
import torch_xla
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig
from infra.utilities import (
    Framework,
    compile_torch_workload_for_cpu,
    compile_torch_workload_for_tt_device,
)
from infra.workloads import TorchWorkload, Workload
from tt_torch.sharding import sharding_constraint_tensor
from ttxla_tools.logging import logger

from tests.infra.evaluators import ComparisonResult
from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.config import Parallelism

from .model_tester import ModelTester, RunMode


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

    Attributes:
        _model_size: Stores the model size in number of parameters
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
        self._model_size = None

        super().__init__(
            comparison_config,
            run_mode,
            Framework.TORCH,
            compiler_config,
            dtype_override,
        )
        # Set custom compile options if provided.
        if compiler_config is not None:
            torch_xla.set_custom_compile_options(
                compiler_config.to_torch_compile_options()
            )

    # @override
    def _configure_model(self) -> None:
        self._device_runner.set_training_mode(self._run_mode == RunMode.TRAINING)
        super()._configure_model()
        self._calculate_model_size()

    # @override
    def _configure_model_for_inference(self) -> None:
        assert isinstance(self._model, torch.nn.Module)
        self._model.eval()

    # @override
    def _configure_model_for_training(self) -> None:
        assert isinstance(self._model, torch.nn.Module)
        self._model.train()

    def _calculate_model_size(self) -> None:
        if isinstance(self._model, torch.nn.Module):
            self._model_size = sum(p.numel() for p in self._model.parameters())
            logger.debug(f"Model size: {self._model_size / 1e9}B")
        else:
            logger.debug("Model is not a torch.nn.Module, skipping size calculation")
            self._model_size = None

    # @override
    def _cache_model_inputs(self) -> None:
        self._input_activations = self._get_input_activations()

    # @override
    def _initialize_workload(self) -> None:
        args = self._get_forward_method_args()
        kwargs = self._get_forward_method_kwargs()

        assert (
            len(args) > 0 or len(kwargs) > 0
        ), "Forward method args or kwargs or both must be provided"

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

    def _compile_for_cpu(self, workload: Workload) -> None:
        compile_torch_workload_for_cpu(workload)

    def _run_on_cpu(self, compiled_workload: Workload) -> torch.Tensor:
        with _mask_jax_accelerator():
            return super()._run_on_cpu(compiled_workload)

    def _compile_for_tt_device(self, workload: Workload, options=None) -> None:
        compile_torch_workload_for_tt_device(workload=workload, torch_options=options)

    def _unpack_forward_output(self, output: Any) -> torch.Tensor:
        return output

    def _extract_grads(
        self, model: torch.nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
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

    def mark_gradient_sharding(self, model: torch.nn.Module):
        assert (
            self._workload.shard_spec_fn is not None
        ), "Shard spec function must be provided for tensor parallel training"
        assert (
            self._workload.mesh is not None
        ), "Mesh must be provided for tensor parallel training"

        shard_specs = self._workload.shard_spec_fn(self._model)
        assert (
            shard_specs is not None
        ), "Shard specs must be provided for tensor parallel training"

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if param not in shard_specs:
                logger.warning(f"Parameter {name} not found in shard specs")
                continue
            shard_spec = shard_specs[param]
            param.grad = sharding_constraint_tensor(
                param.grad, self._workload.mesh, shard_spec
            )

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        torch_xla._XLAC._init_computation_client()

        # Run forward on CPU
        self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)
        cpu_res = self._unpack_forward_output(cpu_res)

        random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)

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
        compile_options = {"tt_experimental_compile": False}

        if self._parallelism == Parallelism.TENSOR_PARALLEL:
            compile_options["tt_enable_torch_fx_fusion_pass"] = False

        self._compile_for_tt_device(self._workload, compile_options)
        tt_res = self._run_on_tt_device(self._workload)
        tt_res = self._unpack_forward_output(tt_res)

        torch_xla.sync(wait=True)

        tt_backward_workload = Workload(
            framework=self._framework,
            executable=tt_res.backward,
            args=[],
            kwargs={"gradient": random_grad},
        )
        self._run_on_tt_device(tt_backward_workload)

        if self._parallelism == Parallelism.TENSOR_PARALLEL:
            self.mark_gradient_sharding(self._model)

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

        return backward_result, forward_result

    # @override
    def _apply_model_dtype(self) -> None:
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._dtype_override)
        else:
            raise TypeError("Model does not have 'to' method to apply dtype.")

    # @override
    def _apply_inputs_dtype(self) -> None:
        self._input_activations = self._cast_tensors_to_dtype(
            self._input_activations, self._dtype_override
        )

    def _cast_tensors_to_dtype(self, obj, dtype):
        if isinstance(obj, torch.Tensor):
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
