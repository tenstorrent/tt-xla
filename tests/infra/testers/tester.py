# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified tester for all comparison-based testing (ops, graphs, models, multichip)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
import torch_xla
from infra.evaluators import ComparisonConfig, EvaluatorFactory
from infra.runners import DeviceRunnerFactory
from infra.utilities import (
    Framework,
    Mesh,
    Tensor,
    compile_jax_workload_for_cpu,
    compile_jax_workload_for_tt_device,
    compile_torch_workload_for_cpu,
    compile_torch_workload_for_tt_device,
    random_tensor,
    sanitize_test_name,
)
from infra.workloads import Workload
from infra.workloads.torch_workload import TorchWorkload
from jax._src.typing import DTypeLike

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.utilities.filecheck_utils import (
    run_filecheck,
    validate_filecheck_results,
)

FILECHECK_DIR = Path(__file__).parent.parent.parent / "filecheck"


class Tester:
    """Unified tester for all comparison-based testing.

    Handles ops, graphs, models (single-chip and multichip) for both JAX and Torch.
    The tester receives fully-formed Workloads and does not know how they were created.
    """

    def __init__(
        self,
        framework: Framework,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        compiler_config: CompilerConfig = None,
        torch_options: dict = None,
    ) -> None:
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._framework = framework
        self._comparison_config = comparison_config
        self._compiler_config = compiler_config
        self._torch_options = torch_options if torch_options is not None else {}
        self._enable_perf_measurement = (
            os.environ.get("ENABLE_OP_TEST_PERF_MEASUREMENT", "0") == "1"
        )

        # Initialize device runner and evaluator
        self._device_runner = DeviceRunnerFactory.create_runner(self._framework)
        self._evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type="comparison",
            framework=self._framework,
            comparison_config=self._comparison_config,
        )

    @property
    def device_runner(self):
        return self._device_runner

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def framework(self):
        return self._framework

    @property
    def compiler_config(self):
        return self._compiler_config

    # ---- Core test loop ----

    def test(
        self, workload: Workload, cpu_workload: Workload = None, request=None
    ) -> None:
        """Core test: compile+run on CPU, compile+run on device, compare results.

        Args:
            workload: Workload to run on TT device.
            cpu_workload: Workload to run on CPU. If None, uses workload for both.
            request: pytest request fixture for filecheck/serialization.
        """
        cpu_wl = cpu_workload if cpu_workload is not None else workload

        self._compile_for_cpu(cpu_wl)
        cpu_res = self._device_runner.run_on_cpu(cpu_wl)

        self._compile_for_tt_device(workload)
        tt_res = self._device_runner.run_on_tt_device(workload)

        self._evaluator.evaluate(tt_res, cpu_res)

        if self._enable_perf_measurement:
            self._test_e2e_perf(workload)

        if request:
            self.handle_filecheck_and_serialization(request, workload)

    def test_with_random_inputs(
        self,
        f: Callable,
        input_shapes: Sequence[tuple],
        minval: float = 0.0,
        maxval: float = 1.0,
        dtype: str | DTypeLike | torch.dtype = "float32",
        request=None,
    ) -> None:
        """Tests f with random inputs by running on TT device and CPU and comparing."""
        inputs = [
            random_tensor(
                shape,
                minval=minval,
                maxval=maxval,
                dtype=dtype,
                framework=self._framework,
            )
            for shape in input_shapes
        ]
        workload = Workload(framework=self._framework, executable=f, args=inputs)
        self.test(workload, request=request)

    # ---- Compilation ----

    def _compile_for_cpu(self, workload: Workload) -> None:
        if self._framework == Framework.JAX:
            compile_jax_workload_for_cpu(workload)
        else:
            compile_torch_workload_for_cpu(workload)

    def _compile_for_tt_device(self, workload: Workload) -> None:
        if self._framework == Framework.JAX:
            compile_jax_workload_for_tt_device(
                workload, self._compiler_config.to_jax_compiler_options()
            )
        else:
            if self._compiler_config is not None:
                torch_xla.set_custom_compile_options(
                    self._compiler_config.to_torch_compile_options()
                )
            compile_torch_workload_for_tt_device(workload, self._torch_options)

    # ---- Performance measurement ----

    def _test_e2e_perf(self, workload: Workload) -> None:
        warmup_iters_count = 3
        perf_iters_count = 2

        for _ in range(warmup_iters_count):
            _ = self._device_runner.run_on_tt_device(workload)

        perf_times = []
        for _ in range(perf_iters_count):
            iter_start = time.perf_counter_ns()
            self._device_runner.run_on_tt_device(workload)
            iter_end = time.perf_counter_ns()
            perf_times.append(iter_end - iter_start)

        tt_total_time = sum(perf_times)
        avg_time = tt_total_time / perf_iters_count
        self._print_e2e_perf_stats(perf_times, avg_time, tt_total_time)

    @staticmethod
    def _print_e2e_perf_stats(
        perf_times: list[float], avg_time: float, total_time: float
    ) -> None:
        print("====================================================================")
        print("| BENCHMARK:  ")
        print("--------------------------------------------------------------------")
        total_iter = len(perf_times)
        for i, perf_time in enumerate(perf_times, 1):
            print(f"| Iteration {i}/{total_iter}: {perf_time / 1e6:.04} ms")
        print(f"| e2e_perf-avg_time: {avg_time / 1e6:.04} ms")
        print(f"| e2e_perf-total_time: {total_time / 1e6:.04} ms")
        print("====================================================================")

    # ---- Serialization and filecheck ----

    def serialize_on_device(self, workload: Workload, output_prefix: str) -> None:
        """Serializes a workload on TT device with proper compiler configuration."""
        if self._framework == Framework.JAX:
            compiler_options = self._compiler_config.to_jax_compiler_options()
        elif self._framework == Framework.TORCH:
            compiler_options = self._compiler_config.to_torch_compile_options()
            self._compile_for_tt_device(workload)
        else:
            compiler_options = None

        self._device_runner.serialize_on_device(
            workload, output_prefix, compiler_options=compiler_options
        )

    def serialize_compilation_artifacts(
        self, test_name: str, workload: Workload
    ) -> None:
        clean_name = sanitize_test_name(test_name)
        output_prefix = f"output_artifact/{clean_name}"
        self.serialize_on_device(workload, output_prefix)

    def handle_filecheck_and_serialization(self, request, workload: Workload) -> None:
        """Serializes workload if --serialize flag is set or filecheck patterns are specified,
        then runs filecheck validation if patterns are provided."""
        if not request:
            return

        test_id = request.node.name

        serialize = request.config.getoption("--serialize", False)

        filecheck_marker = request.node.get_closest_marker("filecheck")
        pattern_files = (
            filecheck_marker.args[0]
            if filecheck_marker and filecheck_marker.args
            else None
        )

        if serialize or pattern_files:
            self.serialize_compilation_artifacts(test_name=test_id, workload=workload)

        if pattern_files:
            self._run_filecheck(pattern_files, test_id=test_id)

    def _run_filecheck(self, pattern_files: list, test_id: str) -> None:
        self._validate_filecheck_mark(
            pattern_files, test_id=test_id, where="pytest mark"
        )

        filecheck_results = run_filecheck(
            test_node_name=test_id,
            irs_filepath="output_artifact",
            pattern_files=pattern_files,
        )
        validate_filecheck_results(filecheck_results)

    def _validate_filecheck_mark(
        self, pattern_files, *, test_id: str, where: str
    ) -> None:
        if not pattern_files:
            return
        if not isinstance(pattern_files, list):
            print(
                f"WARNING: 'filecheck' mark should pass a list in {where}. Found: {type(pattern_files).__name__}"
            )
            return
        for pattern_file in pattern_files:
            if not isinstance(pattern_file, str):
                print(
                    f"WARNING: filecheck entry should be a string in {where}. Found: {type(pattern_file).__name__}"
                )
                continue
            pattern_path = FILECHECK_DIR / pattern_file
            if not pattern_path.exists():
                print(
                    f"WARNING: filecheck pattern file not found: {pattern_path}\n         Referenced in test '{test_id}'"
                )
