# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

from infra.evaluators import ComparisonConfig, ComparisonResult
from infra.utilities import Framework, Mesh, Model, ShardSpec, Tensor
from infra.workloads import Workload

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.tester import Tester


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value


class ModelTester(ABC):
    """Abstract base class all single chip model testers must inherit.

    Uses a Tester internally for device runner, evaluator, and filecheck handling
    instead of inheriting from BaseTester.
    """

    def __init__(
        self,
        comparison_config: ComparisonConfig,
        run_mode: RunMode,
        framework: Framework,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
    ) -> None:
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._run_mode = run_mode
        self._dtype_override = dtype_override
        self._framework = framework

        self._model: Model = None
        self._workload: Workload = None

        self._disable_perf_measurement = (
            os.environ.get("DISABLE_PERF_MEASUREMENT", "0") == "1"
        )
        self._perf_measurements: list[dict[str, float]] = []

        # Use Tester for device runner, evaluator, and filecheck handling
        self._tester = Tester(
            framework=framework,
            comparison_config=comparison_config,
            compiler_config=compiler_config,
        )
        self._comparison_config = comparison_config
        self._device_runner = self._tester.device_runner
        self._evaluator = self._tester.evaluator

        self._initialize_components()

    def _initialize_components(self) -> None:
        self._initialize_model()
        self._set_model_dtype()
        self._cache_model_inputs()
        self._set_inputs_dtype()
        self._initialize_workload()

    def _initialize_model(self) -> None:
        self._model = self._get_model()
        self._configure_model()

    def _get_shard_specs_function(self) -> Optional[Callable[[Model], ShardSpec]]:
        return None

    def _get_mesh(self) -> Optional[Mesh]:
        return None

    @abstractmethod
    def _get_model(self) -> Model:
        raise NotImplementedError("Subclasses must implement this method.")

    def _configure_model(self) -> None:
        if self._run_mode == RunMode.INFERENCE:
            self._configure_model_for_inference()
        else:
            self._configure_model_for_training()

    @abstractmethod
    def _configure_model_for_inference(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _configure_model_for_training(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _cache_model_inputs(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _set_model_dtype(self) -> None:
        if self._dtype_override is not None:
            self._apply_model_dtype()

    def _set_inputs_dtype(self) -> None:
        if self._dtype_override is not None:
            self._apply_inputs_dtype()

    def _apply_model_dtype(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _apply_inputs_dtype(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _initialize_workload(self) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    def _get_forward_method_name(self) -> str:
        return "__call__"

    def test(self, request=None) -> Tuple[ComparisonResult, ...]:
        if self._run_mode == RunMode.INFERENCE:
            return self._test_inference(request=request)
        else:
            return self._test_training()

    def _test_inference(self, request=None) -> Tuple[ComparisonResult, ...]:
        self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)

        self._compile_for_tt_device(self._workload)

        if not self._disable_perf_measurement:
            e2e_perf_stats = self._test_e2e_perf()
            list.append(self._perf_measurements, e2e_perf_stats)

        tt_res = self._run_on_tt_device(self._workload)

        if request:
            self._tester.handle_filecheck_and_serialization(request, self._workload)

        return (self._compare(tt_res, cpu_res),)

    def _test_e2e_perf(self) -> dict[str, float]:
        warmup_iters_count = 3
        perf_iters_count = 2

        for _ in range(warmup_iters_count):
            _ = self._run_on_tt_device(self._workload)

        perf_times = []
        for _ in range(perf_iters_count):
            iter_start = time.perf_counter()
            self._run_on_tt_device(self._workload)
            iter_end = time.perf_counter()
            perf_times.append(iter_end - iter_start)

        tt_total_time = sum(perf_times)
        avg_time = tt_total_time / perf_iters_count

        return {
            "warmup_iters_count": warmup_iters_count,
            "perf_iters_count": perf_iters_count,
            "total_time": tt_total_time,
            "avg_time": avg_time,
            "perf_times": perf_times,
        }

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        return self._device_runner.run_on_cpu(compiled_workload)

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> ComparisonResult:
        return self._evaluator.evaluate(device_out, golden_out)

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize_on_device(
        self, workload: Workload = None, output_prefix: str = None
    ) -> None:
        if workload is None:
            if self._workload is None:
                self._initialize_workload()
            workload = self._workload

        self._tester.serialize_on_device(workload, output_prefix)

    def serialize_compilation_artifacts(
        self, test_name: str, workload: Workload
    ) -> None:
        self._tester.serialize_compilation_artifacts(test_name, workload)

    def handle_filecheck_and_serialization(self, request, workload: Workload) -> None:
        self._tester.handle_filecheck_and_serialization(request, workload)
