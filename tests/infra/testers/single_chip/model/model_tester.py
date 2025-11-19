# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional, Tuple

from infra.comparators import ComparisonConfig, ComparisonResult
from infra.utilities import Framework, Mesh, Model, ShardSpec, Tensor
from infra.workloads import Workload

from tests.infra.testers.compiler_config import CompilerConfig

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow

from ...base_tester import BaseTester


class RunMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"

    def __str__(self) -> str:
        return self.value

class ExecutionGranularity(Enum):
    FULL = "full"
    OP_BY_OP = "op_by_op"

    def __str__(self) -> str:
        return self.value


class ModelTester(BaseTester, ABC):
    """Abstract base class all single chip model testers must inherit."""

    def __init__(
        self,
        comparison_config: ComparisonConfig,
        run_mode: RunMode,
        execution_granularity: ExecutionGranularity,
        framework: Framework,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
        record_property=None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._run_mode = run_mode
        self._execution_granularity = execution_granularity
        self._dtype_override = dtype_override

        self._model: Model = None
        self._workload: Workload = None

        super().__init__(comparison_config, framework, record_property)
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
            result = self._test_inference()
        else:
            result = self._test_training() 

        import os
        import glob

        irs_dir = os.path.join(self._compiler_config.export_path, "irs")
        pattern = os.path.join(irs_dir, "shlo_compiler*.mlir")

        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern}")

        # If there are multiple, pick the newest one
        ir_file_path = max(matches, key=os.path.getmtime)

        try:
            with open(ir_file_path, 'r') as f:
                module = f.read()
        except (FileNotFoundError, IOError, OSError) as e:
            print(f"Warning: Could not read IR dump file {ir_file_path}: {e}")
            module = ""
        
        print("Running op by op tests for module:")
        print(module)
        results = self._test_op_by_op(module=module)
        for resulty in results:
            self._record_property(f"OpTest model for: {resulty.op_name}", model_to_dict(resulty))
        return result


    def _test_op_by_op(self, module: str, compile_before_split: bool = False, compile_each_submodule_after_split: bool = False, *, frontend: Optional[str] = "tt-xla", model_name: Optional[str] = None) -> List[OpTest]:
        """
        Tests the model on op by op basis.
        To enable showing progress of the workflow, set env var `SHOW_WORKFLOW_PROGRESS=ON`

        Parameters
        ----------
        module: Module | str
            Original MLIR module (or module str) processed by the workflow.
        compile_before_split: bool
            If True, compiles the module before splitting.
            NOTE if True `compile_each_submodule_after_split` cannot be True.
        compile_each_submodule_after_split: bool
            If True, compiles each submodule after splitting.
            NOTE if True `compile_before_split` cannot be True.
        frontend: Optional[str]
            Name of the frontend using op by op infra.
        model_name: Optional[str]
            Name of the ML model which was passed as original MLIR module to the workflow.

        Returns
        -------
        List[OpTest]
            List of `OpTest` pydantic models
        """
        return run_op_by_op_workflow(
            module=module,
            compile_before_split=compile_before_split,
            compile_each_submodule_after_split=compile_each_submodule_after_split,
            frontend=frontend,
            model_name=model_name,
        )

    def _test_inference(self) -> Tuple[ComparisonResult, ...]:
        """
        Tests the model by running inference on TT device and on CPU and comparing the
        results.
        """
        self._compile_for_cpu(self._workload)
        cpu_res = self._run_on_cpu(self._workload)

        self._compile_for_tt_device(self._workload)
        tt_res = self._run_on_tt_device(self._workload)

        return (self._compare(tt_res, cpu_res),)

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
