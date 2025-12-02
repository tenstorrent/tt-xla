# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import glob
import os
import subprocess
import sys

import pytest
from infra import RunMode
from infra.testers.compiler_config import CompilerConfig
from infra.testers.single_chip.model import (
    DynamicLoader,
    JaxDynamicLoader,
    TorchDynamicLoader,
)

from tests.infra.comparators.comparator import Comparator, ComparisonResult
from tests.infra.utilities.filecheck_utils import *
from tests.infra.utilities.types import Framework
from tests.runner.requirements import RequirementsManager
from tests.runner.test_config.torch import PLACEHOLDER_MODELS
from tests.runner.test_utils import (
    ModelTestConfig,
    ModelTestStatus,
    fix_venv_isolation,
    record_model_test_properties,
    update_test_metadata_for_exception,
)
from tests.runner.testers import (
    DynamicJaxModelTester,
    DynamicJaxMultiChipModelTester,
    DynamicTorchModelTester,
)
from tests.utils import BringupStatus
from third_party.tt_forge_models.config import Parallelism

from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import run_op_by_op_workflow

from typing import List, Optional

def _test_op_by_op(module: str, compile_before_split: bool = False, compile_each_submodule_after_split: bool = False, *, frontend: Optional[str] = "tt-xla", model_name: Optional[str] = None) -> List[OpTest]:
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

# Setup test discovery using TorchDynamicLoader and JaxDynamicLoader
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT_TORCH, test_entries_torch = TorchDynamicLoader.setup_test_discovery(
    PROJECT_ROOT
)
MODELS_ROOT_JAX, test_entries_jax = JaxDynamicLoader.setup_test_discovery(PROJECT_ROOT)

# Combine all test entries for unified tests
all_test_entries = test_entries_torch + test_entries_jax

def _run_model_test_impl(
    test_entry,
    run_mode: RunMode,
    parallelism: Parallelism,
    framework: Framework,
    request,
    record_property,
    test_metadata: ModelTestConfig,
    capteesys,
    compiler_config: CompilerConfig = None,
    **kwargs,  # Extra fixtures like clear_torchxla_computation_cache
) -> None:
    """Core implementation for running a model test.

    This function contains the shared logic for both JAX and Torch tests.
    It's extracted to avoid duplication between test_all_models_jax/torch.
    """
    # Fix venv isolation issue: ensure venv packages take precedence over system packages
    fix_venv_isolation()

    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path):

        # Get the model loader and model info from desired model, variant.
        loader = ModelLoader(variant=variant)
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"Running {request.node.nodeid} - {model_info.name}", flush=True)

        ir_dump_path = ""
        # Dump all collected IRs if --dump-irs option is enabled
        if request.config.getoption("--dump-irs", default=False):
            ir_dump_path = os.path.join(PROJECT_ROOT, "collected_irs", model_info.name)

        if compiler_config is None:
            compiler_config = CompilerConfig(export_path=ir_dump_path)

        succeeded = False
        comparison_result = None
        tester = None
        filecheck_results = None

        try:
            # Only run the actual model test if not marked for skip. The record properties
            # function in finally block will always be called and handles the pytest.skip.
            if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
                # Framework-specific tester creation
                if framework == Framework.TORCH:
                    tester = DynamicTorchModelTester(
                        run_mode,
                        loader=loader,
                        comparison_config=test_metadata.to_comparison_config(),
                        compiler_config=compiler_config,
                        parallelism=parallelism,
                    )
                elif framework == Framework.JAX:
                    if parallelism == Parallelism.TENSOR_PARALLEL:
                        tester = DynamicJaxMultiChipModelTester(
                            model_loader=loader,
                            run_mode=run_mode,
                            comparison_config=test_metadata.to_comparison_config(),
                            compiler_config=compiler_config,
                        )
                    else:
                        tester = DynamicJaxModelTester(
                            run_mode,
                            loader=loader,
                            comparison_config=test_metadata.to_comparison_config(),
                            compiler_config=compiler_config,
                        )
                else:
                    raise ValueError(f"Unknown framework: {framework}")

                comparison_result = tester.test()

                # Check if filecheck patterns are specified
                pattern_files = (
                    test_metadata.filechecks
                    if hasattr(test_metadata, "filechecks")
                    else None
                )

                # Serialize if --serialize flag is set OR if pattern files are specified
                # Serializing IR to disk is required for FileCheck
                if (
                    request.config.getoption("--serialize", default=False)
                    or pattern_files
                ):
                    tester.serialize_compilation_artifacts(request.node.name)

                # All results must pass for the test to succeed
                succeeded = all(result.passed for result in comparison_result)

                # Run FileCheck on generated IR files if test succeeded
                if succeeded and pattern_files:
                    filecheck_results = run_filecheck(
                        test_node_name=request.node.name,
                        irs_filepath="output_artifact",
                        pattern_files=pattern_files,
                    )

                # Trigger assertion after comparison_result is cached, and
                #     fallthrough to finally block on failure.
                Comparator._assert_on_results(comparison_result)
                validate_filecheck_results(filecheck_results)

        except Exception as e:
            out = capteesys.readouterr().out
            err = capteesys.readouterr().err
            # Record runtime failure info so it can be reflected in report properties
            update_test_metadata_for_exception(test_metadata, e, stdout=out, stderr=err)
            raise
        finally:
            # If there are multiple comparison results, only record the first one because the
            #     DB only supports single comparison result for now
            if comparison_result is not None and len(comparison_result) > 0:
                if len(comparison_result) > 1:
                    print(
                        f"{len(comparison_result)} comparison results found for {request.node.nodeid}, only recording the first one."
                    )
                comparison_result = comparison_result[0]

            comparison_config = tester._comparison_config if tester else None

            # If we mark tests with xfail at collection time, then this isn't hit.
            # Always record properties and handle skip/xfail cases uniformly
            record_model_test_properties(
                record_property,
                request,
                model_info=model_info,
                test_metadata=test_metadata,
                run_mode=run_mode,
                parallelism=parallelism,
                test_passed=succeeded,
                comparison_result=comparison_result,
                comparison_config=comparison_config,
            )


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "run_mode",
    [
        pytest.param(RunMode.INFERENCE, id="inference", marks=pytest.mark.inference),
        pytest.param(RunMode.TRAINING, id="training", marks=pytest.mark.training),
    ],
)
@pytest.mark.parametrize(
    "parallelism",
    [
        pytest.param(
            Parallelism.SINGLE_DEVICE,
            id="single_device",
            marks=pytest.mark.single_device,
        ),
        pytest.param(
            Parallelism.DATA_PARALLEL,
            id="data_parallel",
            marks=pytest.mark.data_parallel,
        ),
        pytest.param(
            Parallelism.TENSOR_PARALLEL,
            id="tensor_parallel",
            marks=pytest.mark.tensor_parallel,
        ),
    ],
)
@pytest.mark.parametrize(
    "test_entry",
    test_entries_torch,
    ids=DynamicLoader.create_test_id_generator(MODELS_ROOT_TORCH),
)
def test_all_models_torch(
    test_entry,
    run_mode,
    parallelism,
    record_property,
    test_metadata,
    request,
    capteesys,
    clear_torchxla_computation_cache,
):
    """PyTorch model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.TORCH,
        request=request,
        record_property=record_property,
        test_metadata=test_metadata,
        capteesys=capteesys,
        clear_torchxla_computation_cache=clear_torchxla_computation_cache,
    )


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "run_mode",
    [
        pytest.param(RunMode.INFERENCE, id="inference", marks=pytest.mark.inference),
        pytest.param(RunMode.TRAINING, id="training", marks=pytest.mark.training),
    ],
)
@pytest.mark.parametrize(
    "parallelism",
    [
        pytest.param(
            Parallelism.SINGLE_DEVICE,
            id="single_device",
            marks=pytest.mark.single_device,
        ),
        # TODO(kmabee): Add when data_parallel is supported next.
        # pytest.param(
        #     Parallelism.DATA_PARALLEL,
        #     id="data_parallel",
        #     marks=pytest.mark.data_parallel,
        # ),
        pytest.param(
            Parallelism.TENSOR_PARALLEL,
            id="tensor_parallel",
            marks=pytest.mark.tensor_parallel,
        ),
    ],
)
@pytest.mark.parametrize(
    "test_entry",
    test_entries_jax,
    ids=DynamicLoader.create_test_id_generator(MODELS_ROOT_JAX),
)
def test_all_models_jax(
    test_entry,
    run_mode,
    parallelism,
    record_property,
    test_metadata,
    request,
    capteesys,
):
    """JAX model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.JAX,
        request=request,
        record_property=record_property,
        test_metadata=test_metadata,
        capteesys=capteesys,
    )


@pytest.mark.model_test
@pytest.mark.op_by_op
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "run_mode",
    [
        pytest.param(RunMode.INFERENCE, id="inference", marks=pytest.mark.inference),
        pytest.param(RunMode.TRAINING, id="training", marks=pytest.mark.training),
    ],
)
@pytest.mark.parametrize(
    "parallelism",
    [
        pytest.param(
            Parallelism.SINGLE_DEVICE,
            id="single_device",
            marks=pytest.mark.single_device,
        ),
        pytest.param(
            Parallelism.DATA_PARALLEL,
            id="data_parallel",
            marks=pytest.mark.data_parallel,
        ),
        pytest.param(
            Parallelism.TENSOR_PARALLEL,
            id="tensor_parallel",
            marks=pytest.mark.tensor_parallel,
        ),
    ],
)
@pytest.mark.parametrize(
    "test_entry",
    all_test_entries,
    ids=lambda e: DynamicLoader.generate_test_id(
        e, MODELS_ROOT_TORCH if e.framework == Framework.TORCH else MODELS_ROOT_JAX
    ),
)
def test_all_models_op_by_op(test_entry, run_mode, parallelism, record_property, request):
    """Run model tests in op-by-op mode.

    This test spawns a subprocess that executes the model test with --dump-irs flag to collect StableHLO IR.
    Then it executes each op individually.

    
    # Determine which underlying test function to call
    """
    if test_entry.framework == Framework.TORCH:
        test_func = "test_all_models_torch"
        models_root = MODELS_ROOT_TORCH
    else:
        test_func = "test_all_models_jax"
        models_root = MODELS_ROOT_JAX

    # Construct the test ID for the underlying test
    test_id = DynamicLoader.generate_test_id(test_entry, models_root)
    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)
    """

    # Build the full nodeid for pytest to execute
    nodeid = (
        f"tests/runner/test_models.py::{test_func}"
        f"[{test_id}-{parallelism.value}-{run_mode.value}]"
    )

    # Construct pytest command
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        nodeid,
        "-v",
        "--dump-irs",  # Custom flag to indicate op-by-op mode
    ]

    # Run the subprocess
    result = subprocess.run(
        pytest_cmd,
        cwd=PROJECT_ROOT,
        capture_output=False,  # Show output in real-time
        text=True,
    )

    # Check if subprocess test passed
    if result.returncode != 0:
        pytest.fail(
            f"Op-by-op test failed with exit code {result.returncode}\n"
            f"Test: {nodeid}"
        ) """

    irs_dir = os.path.join("collected_irs", model_info.name, "irs")
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

    results = _test_op_by_op(module=module)
    for resulty in results:
        record_property(f"OpTest model for: {resulty.op_name}", model_to_dict(resulty))


# A test to generate placeholder model reports for models not yet added to tt-forge-models
@pytest.mark.model_test
@pytest.mark.placeholder
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "model_name",
    list(PLACEHOLDER_MODELS.keys()),
    ids=lambda x: x,
)
def test_placeholder_models(model_name, record_property, request):

    from third_party.tt_forge_models.config import ModelGroup

    class DummyModelInfo:
        def __init__(self, name):
            self._name = name
            self.group = ModelGroup.RED

        @property
        def name(self):
            return self._name

        def to_report_dict(self):
            return {}

    cfg = PLACEHOLDER_MODELS.get(model_name) or {}
    model_test_config_data = {
        "bringup_status": cfg.get("bringup_status", BringupStatus.NOT_STARTED),
        "reason": cfg.get("reason", "Not yet started or WIP"),
    }

    # Sanitize model name to lower case, replace spaches and slashes with underscores
    model_name_lc = model_name.lower().replace(" ", "_").replace("/", "_")

    model_info = DummyModelInfo(model_name_lc)
    test_metadata = ModelTestConfig(data=model_test_config_data, arch=None)

    record_model_test_properties(
        record_property,
        request,
        model_info=model_info,
        test_metadata=test_metadata,
        run_mode=RunMode.INFERENCE,
        parallelism=Parallelism.SINGLE_DEVICE,
    )
