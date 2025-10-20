# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
from infra import RunMode

from tests.infra.comparators.comparator import Comparator, ComparisonResult
from tests.runner.requirements import RequirementsManager
from tests.runner.test_config import PLACEHOLDER_MODELS
from tests.runner.test_utils import (
    DynamicTorchModelTester,
    ModelTestConfig,
    ModelTestStatus,
    create_test_id_generator,
    record_model_test_properties,
    setup_test_discovery,
    update_test_metadata_for_exception,
)
from tests.utils import BringupStatus
from third_party.tt_forge_models.config import Parallelism

# Setup test discovery using utility functions
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT, test_entries = setup_test_discovery(PROJECT_ROOT)


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
    test_entries,
    ids=create_test_id_generator(MODELS_ROOT),
)
def test_all_models(
    test_entry,
    run_mode,
    op_by_op,
    parallelism,
    record_property,
    test_metadata,
    request,
    capteesys,
):

    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    # Dump ModelTestConfig test_metadata for debugging to file
    # FIXME
    try:
        # Output directory can be overridden via env; default under project artifacts
        output_dir = os.environ.get(
            "TT_XLA_TEST_METADATA_DIR",
            os.path.join(PROJECT_ROOT, ".artifacts", "test_metadata"),
        )
        os.makedirs(output_dir, exist_ok=True)

        # Use fully-parameterized pytest nodeid as the base name, sanitized for filesystem
        nodeid = request.node.nodeid
        safe_name = "".join(c if c.isalnum() else "_" for c in nodeid)
        arch_value = getattr(test_metadata, "arch", None)
        arch_str = str(arch_value) if arch_value is not None else ""
        arch_safe = "".join(c if c.isalnum() else "_" for c in arch_str)
        prefix = f"{arch_safe}__" if arch_safe else ""
        output_path = os.path.join(output_dir, f"{prefix}{safe_name}.json")

        def _serialize(value):
            try:
                from enum import Enum

                if isinstance(value, Enum):
                    return value.value
            except Exception:
                pass
            if isinstance(value, dict):
                return {str(k): _serialize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_serialize(v) for v in value]
            return value

        payload = {
            "test_nodeid": nodeid,
            "resolved": {
                "arch": _serialize(getattr(test_metadata, "arch", None)),
                "status": _serialize(getattr(test_metadata, "status", None)),
                "required_pcc": _serialize(
                    getattr(test_metadata, "required_pcc", None)
                ),
                "assert_pcc": _serialize(getattr(test_metadata, "assert_pcc", None)),
                "assert_atol": _serialize(getattr(test_metadata, "assert_atol", None)),
                "required_atol": _serialize(
                    getattr(test_metadata, "required_atol", None)
                ),
                "assert_allclose": _serialize(
                    getattr(test_metadata, "assert_allclose", None)
                ),
                "allclose_rtol": _serialize(
                    getattr(test_metadata, "allclose_rtol", None)
                ),
                "allclose_atol": _serialize(
                    getattr(test_metadata, "allclose_atol", None)
                ),
                "batch_size": _serialize(getattr(test_metadata, "batch_size", None)),
                "reason": _serialize(getattr(test_metadata, "reason", None)),
                "bringup_status": _serialize(
                    getattr(test_metadata, "bringup_status", None)
                ),
                "markers": _serialize(getattr(test_metadata, "markers", None)),
                "supported_archs": _serialize(
                    getattr(test_metadata, "supported_archs", None)
                ),
            },
            "raw_data": _serialize(getattr(test_metadata, "data", {})),
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Failed to write test metadata to file: {e}", flush=True)

    # Exit early after dumping metadata
    return

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path):

        # Get the model loader and model info from desired model, variant.
        loader = ModelLoader(variant=variant)
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"Running {request.node.nodeid} - {model_info.name}", flush=True)

        succeeded = False
        comparison_result = None
        tester = None

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

                # All results must pass for the test to succeed
                succeeded = all(result.passed for result in comparison_result)

                # Trigger assertion after comparison_result is cached, and
                #     fallthrough to finally block on failure.
                Comparator._assert_on_results(comparison_result)

        except Exception as e:
            err = capteesys.readouterr().err
            # Record runtime failure info so it can be reflected in report properties
            update_test_metadata_for_exception(test_metadata, e, stderr=err)
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
