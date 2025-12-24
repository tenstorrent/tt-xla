# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import socket

import pytest
from infra import RunMode
from infra.testers.single_chip.model import (
    DynamicLoader,
    JaxDynamicLoader,
    TorchDynamicLoader,
)

from tests.infra.comparators.comparator import Comparator, ComparisonResult
from tests.infra.utilities.filecheck_utils import *
from tests.runner.requirements import RequirementsManager
from tests.runner.test_config.torch import PLACEHOLDER_MODELS
from tests.runner.test_utils import (
    ModelTestConfig,
    ModelTestStatus,
    create_benchmark_result,
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
from third_party.tt_forge_models.config import ModelSource, Parallelism

# Setup test discovery using TorchDynamicLoader and JaxDynamicLoader
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT_TORCH, test_entries_torch = TorchDynamicLoader.setup_test_discovery(
    PROJECT_ROOT
)
MODELS_ROOT_JAX, test_entries_jax = JaxDynamicLoader.setup_test_discovery(PROJECT_ROOT)


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
    "op_by_op",
    [None],
    ids=["full"],  # When op-by-op flow is required/supported, add here.
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
    op_by_op,
    parallelism,
    record_property,
    test_metadata,
    request,
    captured_output_fixture,
    clear_torchxla_computation_cache,
):
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

        succeeded = False
        comparison_result = None
        tester = None
        filecheck_results = None

        try:
            # Only run the actual model test if not marked for skip. The record properties
            # function in finally block will always be called and handles the pytest.skip.
            if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
                tester = DynamicTorchModelTester(
                    run_mode,
                    loader=loader,
                    comparison_config=test_metadata.to_comparison_config(),
                    parallelism=parallelism,
                )

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
            captured = captured_output_fixture.readouterr()
            # Record runtime failure info so it can be reflected in report properties
            update_test_metadata_for_exception(
                test_metadata, e, stdout=captured.out, stderr=captured.err
            )
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

            # prints perf benchmark results to console
            # Dumps perf benchmark results to JSON report if --perf-report-dir is given
            measurements = getattr(tester, "_perf_measurements", None)
            output_dir = request.config.getoption("--perf-report-dir")
            create_benchmark_result(
                full_model_name=model_info.name,
                output_dir=output_dir,
                perf_id=request.config.getoption("--perf-id"),
                measurements=measurements,
                model_type="generic",
                training=False,
                model_info=model_info.name,
                device_name=socket.gethostname(),
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
    "op_by_op",
    [None],
    ids=["full"],  # When op-by-op flow is required/supported, add here.
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
    op_by_op,
    parallelism,
    record_property,
    test_metadata,
    request,
    captured_output_fixture,
):
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

        succeeded = False
        comparison_result = None
        tester = None
        filecheck_results = None

        try:
            # Only run the actual model test if not marked for skip. The record properties
            # function in finally block will always be called and handles the pytest.skip.
            if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
                if (
                    parallelism == Parallelism.TENSOR_PARALLEL
                    or parallelism == Parallelism.DATA_PARALLEL
                ):
                    tester = DynamicJaxMultiChipModelTester(
                        model_loader=loader,
                        run_mode=run_mode,
                        comparison_config=test_metadata.to_comparison_config(),
                    )
                else:
                    if model_info.source.name == ModelSource.EASYDEL.name:
                        # In EasyDel, single-device models use multi-chip setup with (1,1) mesh
                        tester = DynamicJaxMultiChipModelTester(
                            model_loader=loader,
                            comparison_config=test_metadata.to_comparison_config(),
                            num_devices=1,
                        )
                    else:
                        tester = DynamicJaxModelTester(
                            run_mode,
                            loader=loader,
                            comparison_config=test_metadata.to_comparison_config(),
                        )

                comparison_result = tester.test()

                # Check if filecheck patterns are specified
                pattern_files = (
                    test_metadata.filechecks
                    if hasattr(test_metadata, "filechecks")
                    else None
                )

                # Serialize if --serialize flag is set OR if pattern files are specified
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
            captured = captured_output_fixture.readouterr()
            update_test_metadata_for_exception(
                test_metadata, e, stdout=captured.out, stderr=captured.err
            )
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
                test_passed=succeeded,
                parallelism=parallelism,
                comparison_result=comparison_result,
                comparison_config=comparison_config,
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

    record_model_test_properties(
        record_property,
        request,
        model_info=model_info,
        test_metadata=test_metadata,
        run_mode=RunMode.INFERENCE,
        parallelism=Parallelism.SINGLE_DEVICE,
    )
