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
        print(f"\n[DEBUG][ModelTester.__init__] CALLED", flush=True)
        print(f"  comparison_config = {comparison_config}", flush=True)
        print(f"  run_mode = {run_mode}", flush=True)
        print(f"  framework = {framework}", flush=True)
        print(f"  compiler_config = {compiler_config}", flush=True)
        print(f"  dtype_override = {dtype_override}", flush=True)
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._run_mode = run_mode
        self._dtype_override = dtype_override

        self._model: Model = None
        self._workload: Workload = None

        self._disable_perf_measurement = (
            os.environ.get("DISABLE_PERF_MEASUREMENT", "0") == "1"
        )
        self._perf_measurements: list[dict[str, float]] = []
        print(f"[DEBUG][ModelTester.__init__] disable_perf_measurement={self._disable_perf_measurement}", flush=True)

        super().__init__(
            evaluator_type="comparison",
            comparison_config=comparison_config,
            framework=framework,
        )
        self._initialize_components()
        print(f"[DEBUG][ModelTester.__init__] DONE", flush=True)

    def _initialize_components(self) -> None:
        print(f"\n[DEBUG][ModelTester._initialize_components] CALLED — starting 5-step initialization", flush=True)
        print(f"[DEBUG][ModelTester._initialize_components] Step 1/5: _initialize_model()", flush=True)
        self._initialize_model()
        print(f"[DEBUG][ModelTester._initialize_components] Step 2/5: _set_model_dtype()", flush=True)
        self._set_model_dtype()
        print(f"[DEBUG][ModelTester._initialize_components] Step 3/5: _cache_model_inputs()", flush=True)
        self._cache_model_inputs()
        print(f"[DEBUG][ModelTester._initialize_components] Step 4/5: _set_inputs_dtype()", flush=True)
        self._set_inputs_dtype()
        print(f"[DEBUG][ModelTester._initialize_components] Step 5/5: _initialize_workload()", flush=True)
        self._initialize_workload()
        print(f"[DEBUG][ModelTester._initialize_components] DONE — all 5 steps complete", flush=True)

    def _initialize_model(self) -> None:
        """
        Initializes `self._model`

        It is also important that model is configured before it is prepacked into a
        Workload during `_initialize_workload`.
        """
        print(f"\n[DEBUG][ModelTester._initialize_model] CALLED", flush=True)
        # Store model instance.
        self._model = self._get_model()
        print(f"[DEBUG][ModelTester._initialize_model] Got model: type={type(self._model).__name__}", flush=True)
        # Configure it.
        self._configure_model()
        print(f"[DEBUG][ModelTester._initialize_model] Model configured for {self._run_mode}", flush=True)

    def _get_shard_specs_function(self) -> Optional[Callable[[Model], ShardSpec]]:
        """Optional: returns shard specs function if required; otherwise None."""
        print(f"[DEBUG][ModelTester._get_shard_specs_function] CALLED — returning None (base implementation)", flush=True)
        return None

    def _get_mesh(self) -> Optional[Mesh]:
        """Optional: returns mesh if required; otherwise None."""
        print(f"[DEBUG][ModelTester._get_mesh] CALLED — returning None (base implementation)", flush=True)
        return None

    @abstractmethod
    def _get_model(self) -> Model:
        """Returns model instance."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _configure_model(self) -> None:
        """
        Configures model for inference *or* training, depending on chosen run mode.
        """
        print(f"\n[DEBUG][ModelTester._configure_model] CALLED — run_mode={self._run_mode}", flush=True)
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
        print(f"[DEBUG][ModelTester._set_model_dtype] CALLED — dtype_override={self._dtype_override}", flush=True)
        if self._dtype_override is not None:
            self._apply_model_dtype()
        else:
            print(f"[DEBUG][ModelTester._set_model_dtype] No dtype_override, skipping", flush=True)

    def _set_inputs_dtype(self) -> None:
        """Sets inputs dtype if dtype_override is provided."""
        print(f"[DEBUG][ModelTester._set_inputs_dtype] CALLED — dtype_override={self._dtype_override}", flush=True)
        if self._dtype_override is not None:
            self._apply_inputs_dtype()
        else:
            print(f"[DEBUG][ModelTester._set_inputs_dtype] No dtype_override, skipping", flush=True)

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

    def test(self, request=None) -> Tuple[ComparisonResult, ...]:
        """Tests the model depending on test type with which tester was configured."""
        print(f"\n{'='*80}", flush=True)
        print(f"[DEBUG][ModelTester.test] CALLED — run_mode={self._run_mode}", flush=True)
        print(f"{'='*80}", flush=True)
        if self._run_mode == RunMode.INFERENCE:
            return self._test_inference(request=request)
        else:
            return self._test_training()

    def _test_inference(self, request=None) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        from tests.infra.testers.base_tester import _debug_summarize

        print(f"\n[DEBUG][ModelTester._test_inference] === STEP 1: Compile for CPU ===", flush=True)
        self._compile_for_cpu(self._workload)

        print(f"\n[DEBUG][ModelTester._test_inference] === STEP 2: Run on CPU (golden reference) ===", flush=True)
        cpu_res = self._run_on_cpu(self._workload)
        print(f"[DEBUG][ModelTester._test_inference] CPU result: {_debug_summarize(cpu_res)}", flush=True)

        print(f"\n[DEBUG][ModelTester._test_inference] === STEP 3: Compile for TT device ===", flush=True)
        self._compile_for_tt_device(self._workload)

        if not self._disable_perf_measurement:
            print(f"\n[DEBUG][ModelTester._test_inference] === STEP 4: E2E Perf measurement ===", flush=True)
            e2e_perf_stats = self._test_e2e_perf()
            list.append(self._perf_measurements, e2e_perf_stats)
            print(f"[DEBUG][ModelTester._test_inference] Perf stats: avg_time={e2e_perf_stats.get('avg_time', 'N/A'):.4f}s", flush=True)

        print(f"\n[DEBUG][ModelTester._test_inference] === STEP 5: Run on TT device ===", flush=True)
        tt_res = self._run_on_tt_device(self._workload)
        print(f"[DEBUG][ModelTester._test_inference] TT result: {_debug_summarize(tt_res)}", flush=True)

        if request:
            self.handle_filecheck_and_serialization(request, self._workload)

        print(f"\n[DEBUG][ModelTester._test_inference] === STEP 6: Compare TT vs CPU ===", flush=True)
        comparison = self._compare(tt_res, cpu_res)
        print(f"[DEBUG][ModelTester._test_inference] Comparison result: {comparison}", flush=True)
        return (comparison,)

    def _test_e2e_perf(self) -> dict[str, float]:
        warmup_iters_count = 3
        perf_iters_count = 2

        # warmup runs
        for _ in range(warmup_iters_count):
            _ = self._run_on_tt_device(self._workload)

        # e2e perf
        perf_times = []
        for _ in range(perf_iters_count):
            iter_start = time.perf_counter()
            tt_res = self._run_on_tt_device(self._workload)
            iter_end = time.perf_counter()
            perf_times.append(iter_end - iter_start)

        tt_total_time = sum(perf_times)
        avg_time = tt_total_time / perf_iters_count

        perf_stats = {
            "warmup_iters_count": warmup_iters_count,
            "perf_iters_count": perf_iters_count,
            "total_time": tt_total_time,
            "avg_time": avg_time,
            "perf_times": perf_times,
        }

        return perf_stats

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        print(f"[DEBUG][ModelTester._run_on_cpu] CALLED — delegating to device_runner.run_on_cpu()", flush=True)
        return self._device_runner.run_on_cpu(compiled_workload)

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        print(f"[DEBUG][ModelTester._run_on_tt_device] CALLED — delegating to device_runner.run_on_tt_device()", flush=True)
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> ComparisonResult:
        """Compares device with golden output and returns the result."""
        print(f"[DEBUG][ModelTester._compare] CALLED — comparing device output vs golden output", flush=True)
        return self._evaluator.evaluate(device_out, golden_out)

    def _test_training(self) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running training on TT device and on CPU and comparing the
        forward results and gradients. Implementation is framework-specific.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize_on_device(
        self, workload: Workload = None, output_prefix: str = None
    ) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            workload: Workload to serialize (if None, uses self._workload)
            output_prefix: Base path and filename prefix for output files
        """
        # Use provided workload or fall back to self._workload
        if workload is None:
            if self._workload is None:
                self._initialize_workload()
            workload = self._workload

        # Get compiler options based on framework
        if self._framework == Framework.JAX:
            compiler_options = self._compiler_config.to_jax_compiler_options()
        elif self._framework == Framework.TORCH:
            compiler_options = self._compiler_config.to_torch_compile_options()
        else:
            raise ValueError(f"Unsupported framework: {self._framework}")

        self._device_runner.serialize_on_device(
            workload, output_prefix, compiler_options=compiler_options
        )
