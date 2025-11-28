# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

from infra.comparators import ComparisonConfig, ComparisonResult
from infra.utilities import Framework, Mesh, Model, ShardSpec, Tensor
from infra.workloads import Workload
from loguru import logger

from tests.infra.testers.compiler_config import CompilerConfig

from ...base_tester import BaseTester


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


class ModelTester(BaseTester, ABC):
    """Abstract base class all single chip model testers must inherit."""

    def __init__(
        self,
        comparison_config: ComparisonConfig,
        run_mode: RunMode,
        framework: Framework,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._run_mode = run_mode
        self._dtype_override = dtype_override

        self._model: Model = None
        self._workload: Workload = None

        self._perf_measurements: list[dict[str, float]] = []

        super().__init__(comparison_config, framework)
        self._initialize_components()

    def _initialize_components(self) -> None:
        self._initialize_model()
        self._set_model_dtype()
        self._cache_model_inputs()
        self._set_inputs_dtype()
        self._initialize_workload()

    def _initialize_model(self) -> None:
        """
        Initializes `self._model`

        It is also important that model is configured before it is prepacked into a
        Workload during `_initialize_workload`.
        """
        # Store model instance.
        self._model = self._get_model()
        # Configure it.
        self._configure_model()

    def _get_shard_specs_function(self) -> Optional[Callable[[Model], ShardSpec]]:
        """Optional: returns shard specs function if required; otherwise None."""
        return None

    def _get_mesh(self) -> Optional[Mesh]:
        """Optional: returns mesh if required; otherwise None."""
        return None

    @abstractmethod
    def _get_model(self) -> Model:
        """Returns model instance."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _configure_model(self) -> None:
        """
        Configures model for inference *or* training, depending on chosen run mode.
        """
        if self._run_mode == RunMode.INFERENCE:
            self._configure_model_for_inference()
        else:
            self._configure_model_for_training()

    @abstractmethod
    def _configure_model_for_inference(self) -> None:
        """Configures `model` for inference."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _configure_model_for_training(self) -> None:
        """Configures `model` for training."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _cache_model_inputs(self) -> None:
        """Caches model inputs."""
        raise NotImplementedError("Subclasses must implement this method")

    def _set_model_dtype(self) -> None:
        """Sets model dtype if dtype_override is provided."""
        if self._dtype_override is not None:
            self._apply_model_dtype()

    def _set_inputs_dtype(self) -> None:
        """Sets inputs dtype if dtype_override is provided."""
        if self._dtype_override is not None:
            self._apply_inputs_dtype()

    def _apply_model_dtype(self) -> None:
        """Applies dtype to model. Base implementation does nothing."""
        raise NotImplementedError("Subclasses must implement this method")

    def _apply_inputs_dtype(self) -> None:
        """Applies dtype to inputs. Base implementation does nothing."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize_workload(self) -> None:
        """Initializes `self._workload`."""
        raise NotImplementedError("Subclasses must implement this method")

    def _get_forward_method_name(self) -> str:
        """
        Returns string name of model's forward pass method.

        Returns "__call__" by default which is the most common one. "forward" and
        "apply" are also common.
        """
        return "__call__"

    def test(self) -> Tuple[ComparisonResult, ...]:
        """Tests the model depending on test type with which tester was configured."""
        if self._run_mode == RunMode.INFERENCE:
            return self._test_inference()
        else:
            return self._test_training()

    def _test_inference(self) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)

        self._compile_for_tt_device(self._workload)
        e2e_perf_stats = self._test_e2e_perf()
        list.append(self._perf_measurements, e2e_perf_stats)
        tt_res = self._run_on_tt_device(self._workload)

        return (self._compare(tt_res, cpu_res),)

    def _test_e2e_perf(self) -> dict[str, float]:
        warmup_iters = 3
        perf_iters = 2

        # warmup runs
        for _ in range(warmup_iters):
            _ = self._run_on_tt_device(self._workload)

        # e2e perf
        tt_start = time.perf_counter()
        for _ in range(perf_iters):
            tt_res = self._run_on_tt_device(self._workload)
        tt_end = time.perf_counter()
        tt_total_time = tt_end - tt_start

        avg_time = tt_total_time / perf_iters

        perf_stats = {
            "warmup_iters": warmup_iters,
            "perf_iters": perf_iters,
            "total_time": tt_total_time,
            "avg_time": avg_time,
        }

        return perf_stats

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._device_runner.run_on_cpu(compiled_workload)

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> ComparisonResult:
        """Compares device with golden output and returns the result."""
        return self._comparator.compare(device_out, golden_out)

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running training on TT device and on CPU and comparing the
        forward results and gradients. Implementation is framework-specific.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize_on_device(self, output_prefix: str) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            output_prefix: Base path and filename prefix for output files
        """
        if self._workload is None:
            self._initialize_workload()

        # Get compiler options based on framework
        if self._framework == Framework.JAX:
            compiler_options = self._compiler_config.to_jax_compiler_options()
        elif self._framework == Framework.TORCH:
            compiler_options = self._compiler_config.to_torch_compile_options()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

        self._device_runner.serialize_on_device(
            self._workload, output_prefix, compiler_options=compiler_options
        )
