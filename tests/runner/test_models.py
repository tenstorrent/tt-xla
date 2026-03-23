# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import subprocess
import sys
import warnings
from typing import List, Optional

import pytest
import torch
from infra import RunMode
from infra.testers.compiler_config import CompilerConfig
from infra.testers.single_chip.model import (
    DynamicLoader,
    JaxDynamicLoader,
    TorchDynamicLoader,
)

from tests.infra.evaluators import ComparisonResult, Evaluator
from tests.infra.utilities.filecheck_utils import *
from tests.infra.utilities.types import Framework
from tests.runner.requirements import RequirementsManager
from tests.runner.test_config.torch import PLACEHOLDER_MODELS
from tests.runner.test_utils import (
    ModelTestConfig,
    RunPhase,
    find_dumped_ir_files,
    record_model_test_properties,
)
from tests.runner.testers import (
    DynamicJaxModelTester,
    DynamicJaxMultiChipModelTester,
    DynamicTorchModelTester,
)
from tests.utils import BringupStatus
from third_party.tt_forge_models.config import (
    ModelGroup,
    ModelInfo,
    ModelSource,
    Parallelism,
)

# Setup test discovery using TorchDynamicLoader and JaxDynamicLoader
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT_TORCH, test_entries_torch = TorchDynamicLoader.setup_test_discovery(
    PROJECT_ROOT
)
MODELS_ROOT_JAX, test_entries_jax = JaxDynamicLoader.setup_test_discovery(PROJECT_ROOT)
all_test_entries = test_entries_torch + test_entries_jax


def _run_model_test_impl(
    test_entry,
    run_mode: RunMode,
    parallelism: Parallelism,
    framework: Framework,
    test_metadata: ModelTestConfig,
    request,
    run_phase: RunPhase = RunPhase.DEFAULT,
    compiler_config: CompilerConfig = None,
    **kwargs,  # Extra fixtures like clear_torchxla_computation_cache
) -> None:
    """Core implementation for running a model test.

    This function contains the shared logic for both JAX and Torch tests.
    It's extracted to avoid duplication between test_all_models_jax/torch.

    Test results are stored on request.node for the pytest_runtest_makereport
    hook to pick up for property recording and failure classification.
    """
    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    # Get the model loader and model info from desired model, variant.
    loader = ModelLoader(variant=variant)
    model_info = ModelLoader.get_model_info(variant=variant)
    print(f"Running {request.node.nodeid} - {model_info.name}", flush=True)

    # Store early data on node (always available for hook, even on crash)
    node = request.node
    node._model_info = model_info
    node._weights_dtype = None
    node._framework = framework
    node._run_mode = run_mode
    node._run_phase = run_phase
    node._parallelism = parallelism
    node._loader = loader

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path, framework=str(framework)):

        ir_dump_path = ""
        # Dump all collected IRs if --dump-irs option is enabled
        if request.config.getoption("--dump-irs", default=False):
            ir_dump_path = os.path.join(PROJECT_ROOT, "collected_irs", model_info.name)

        if compiler_config is None:
            compiler_config = CompilerConfig()
        if test_metadata.enable_weight_bfp8_conversion:
            compiler_config.experimental_weight_dtype = "bfp8"
            node._weights_dtype = "bfp8"

        if ir_dump_path:
            compiler_config.export_path = ir_dump_path
            compiler_config.export_model_name = model_info.name

        # Framework-specific tester creation
        if framework == Framework.TORCH:
            tester = DynamicTorchModelTester(
                run_mode,
                run_phase=run_phase,
                loader=loader,
                comparison_config=test_metadata.to_comparison_config(),
                compiler_config=compiler_config,
                parallelism=parallelism,
                test_metadata=test_metadata,
            )
        elif framework == Framework.JAX:
            if (
                parallelism == Parallelism.TENSOR_PARALLEL
                or parallelism == Parallelism.DATA_PARALLEL
            ):
                tester = DynamicJaxMultiChipModelTester(
                    model_loader=loader,
                    run_mode=run_mode,
                    comparison_config=test_metadata.to_comparison_config(),
                    compiler_config=compiler_config,
                    parallelism=parallelism,
                )
            else:
                if model_info.source.name == ModelSource.EASYDEL.name:
                    # In EasyDel, single-device models use multi-chip setup with (1,1) mesh
                    tester = DynamicJaxMultiChipModelTester(
                        model_loader=loader,
                        comparison_config=test_metadata.to_comparison_config(),
                        num_devices=1,
                        compiler_config=compiler_config,
                        parallelism=parallelism,
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

        # Add filecheck marker dynamically if patterns are specified in test metadata
        if hasattr(test_metadata, "filechecks") and test_metadata.filechecks:
            node.add_marker(pytest.mark.filecheck(test_metadata.filechecks))

        comparison_result = tester.test(request=request)

        # All results must pass for the test to succeed
        succeeded = all(result.passed for result in comparison_result)

        # Store post-execution results for hook
        node._comparison_result = list(comparison_result)
        node._test_passed = succeeded
        node._comparison_config = tester._comparison_config
        node._model_size = getattr(tester, "_model_size", None)
        node._tester = tester

        Evaluator._assert_on_results(comparison_result)


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
    test_metadata,
    request,
    clear_torchxla_computation_cache,
):
    """PyTorch model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.TORCH,
        request=request,
        test_metadata=test_metadata,
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
    test_entries_jax,
    ids=DynamicLoader.create_test_id_generator(MODELS_ROOT_JAX),
)
def test_all_models_jax(
    test_entry,
    run_mode,
    parallelism,
    test_metadata,
    request,
):
    """JAX model test - delegates to shared implementation."""
    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        parallelism=parallelism,
        framework=Framework.JAX,
        request=request,
        test_metadata=test_metadata,
    )


# LLM Specific tests for decode and prefill phases. Separate test to avoid impacting
# original test names in test_all_models_torch and no need for collection-time deselection logic.

# Build list of (test_entry, run_phase) pairs based on loader capabilities
_llm_test_params = []
for entry in test_entries_torch:
    ModelLoader = entry.variant_info[1]
    if hasattr(ModelLoader, "load_inputs_decode"):
        _llm_test_params.append((entry, RunPhase.LLM_DECODE))
    if hasattr(ModelLoader, "load_inputs_prefill"):
        _llm_test_params.append((entry, RunPhase.LLM_PREFILL))


def _generate_llm_test_id(param_tuple):
    """Generate test ID for LLM tests."""
    entry, phase = param_tuple
    return f"{DynamicLoader.generate_test_id(entry, MODELS_ROOT_TORCH)}-{phase.value}"


def _generate_sequence_length_id(sequence_length):
    """Generate test ID component for sequence_length."""
    return f"seq_{sequence_length}" if sequence_length is not None else "seq_1"


def _generate_batch_size_id(batch_size):
    """Generate test ID component for batch_size."""
    return f"batch_{batch_size}"


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.llm
@pytest.mark.parametrize(
    "run_mode",
    [
        pytest.param(RunMode.INFERENCE, id="inference", marks=pytest.mark.inference),
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
            Parallelism.TENSOR_PARALLEL,
            id="tensor_parallel",
            marks=pytest.mark.tensor_parallel,
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2], ids=_generate_batch_size_id)
@pytest.mark.parametrize(
    "sequence_length",
    [None, 128, 1024, 2048, 4096, 8192],
    ids=_generate_sequence_length_id,
)
@pytest.mark.parametrize(
    "test_entry_and_phase",
    _llm_test_params,
    ids=_generate_llm_test_id,
)
def test_llms_torch(
    test_entry_and_phase,
    sequence_length,
    batch_size,
    run_mode,
    parallelism,
    test_metadata,
    request,
    clear_torchxla_computation_cache,
):
    """PyTorch LLM model test (decode/prefill phases) - delegates to shared implementation."""
    test_entry, run_phase = test_entry_and_phase

    if run_phase == RunPhase.LLM_DECODE:
        # Decode tests don't parametrize on sequence length (default is seq_len = 1).
        if sequence_length is not None:
            pytest.skip("Decode tests do not support sequence_length parameterization")
        # Decode tests for now run only batch_size = 1.
        if batch_size != 1:
            pytest.skip("Decode tests currently only support batch_size=1")
        request.node.add_marker(pytest.mark.llm_decode)

    if run_phase == RunPhase.LLM_PREFILL:
        # Sequence length should be specified for prefill tests.
        if sequence_length is None:
            pytest.skip("Sequence length must be specified for prefill tests")
        request.node.add_marker(pytest.mark.llm_prefill)

    test_metadata.batch_size = batch_size
    test_metadata.seq_len = sequence_length

    _run_model_test_impl(
        test_entry=test_entry,
        run_mode=run_mode,
        run_phase=run_phase,
        parallelism=parallelism,
        framework=Framework.TORCH,
        request=request,
        test_metadata=test_metadata,
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
    ],
)
@pytest.mark.parametrize(
    "test_entry",
    all_test_entries,
    ids=lambda e: DynamicLoader.generate_test_id(
        e, MODELS_ROOT_TORCH if e.framework == Framework.TORCH else MODELS_ROOT_JAX
    ),
)
def test_all_models_op_by_op(
    test_entry, run_mode, parallelism, record_property, request
):
    """Run model tests in op-by-op mode.

    This test spawns a subprocess that executes the model test with --dump-irs flag to collect StableHLO IR.
    Then it executes each op wrapped in a module individually.
    """
    # Import op_by_op_infra only when this test runs (not at module level)
    from op_by_op_infra.pydantic_models import OpTest, model_to_dict
    from op_by_op_infra.workflow import run_op_by_op_workflow

    # Construct a pytest command for the subprocess.
    # Determine the correct test function name based on the framework.
    if test_entry.framework == Framework.TORCH:
        test_function_name = "test_all_models_torch"
        models_root = MODELS_ROOT_TORCH
    else:
        test_function_name = "test_all_models_jax"
        models_root = MODELS_ROOT_JAX

    # Generate the model id and get model metadata.
    model_test_id = DynamicLoader.generate_test_id(test_entry, models_root)
    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)

    # Construct the pytest node ID (unique test identifier) to run in subprocess.
    pytest_node_id = (
        f"tests/runner/test_models.py::{test_function_name}"
        f"[{model_test_id}-{parallelism.value}-{run_mode.value}]"
    )

    pytest_cmd = [sys.executable, "-m", "pytest", pytest_node_id, "-sv", "--dump-irs"]

    subprocess_result = subprocess.run(
        pytest_cmd,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True,
    )

    artifacts_dir = os.path.join(PROJECT_ROOT, "collected_irs", model_info.name)
    matches = find_dumped_ir_files(artifacts_dir)

    results = []
    for ir_file_path in matches:
        try:
            with open(ir_file_path, "r") as f:
                module = f.read()
        except (FileNotFoundError, IOError, OSError) as e:
            pytest.fail(
                f"Op-by-op test failed because IR file couldn't be read.\n"
                f"Test: {pytest_node_id}\n"
                f"File: {ir_file_path}"
            )

        module_results = run_op_by_op_workflow(
            module=module,
            compile_before_split=False,
            compile_each_submodule_after_split=False,
            frontend="tt-xla",
            model_name=model_info.name,
        )
        results.extend(module_results)

    for op_result in results:
        record_property(
            f"OpTest model for: {op_result.op_name}", model_to_dict(op_result)
        )

    shutil.rmtree(artifacts_dir)

    if subprocess_result.returncode != 0:
        warnings.warn(
            f"IR collection subprocess completed with exit code {subprocess_result.returncode}, some IR module might be missing from analysis.",
        )

    failed_operations = sum(1 for r in results if not r.success)
    if failed_operations > 0:
        pytest.fail(
            f"Test failed: {failed_operations} operation(s) failed out of {len(results)} total operations",
            pytrace=False,
        )


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
    weights_dtype = None
    record_model_test_properties(
        record_property,
        request.node,
        model_info=model_info,
        test_metadata=test_metadata,
        run_mode=RunMode.INFERENCE,
        parallelism=Parallelism.SINGLE_DEVICE,
        weights_dtype=weights_dtype,
    )
