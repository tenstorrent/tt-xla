# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from tests.runner.test_utils import (
    ModelTestStatus,
    DynamicTorchModelTester,
    setup_test_discovery,
    create_test_id_generator,
    record_model_test_properties,
    update_test_metadata_for_exception,
)
from tests.runner.requirements import RequirementsManager
from infra import RunMode
from tests.utils import BringupStatus

# Setup test discovery using utility functions
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT, test_entries = setup_test_discovery(PROJECT_ROOT)


@pytest.mark.model_test
@pytest.mark.no_auto_properties
@pytest.mark.parametrize(
    "run_mode",
    [RunMode.INFERENCE],
    ids=["inference"],
)
@pytest.mark.parametrize(
    "op_by_op",
    [None],
    ids=["full"],  # When op-by-op flow is required/supported, add here.
)
@pytest.mark.parametrize(
    "test_entry",
    test_entries,
    ids=create_test_id_generator(MODELS_ROOT),
)
def test_all_models(
    test_entry, run_mode, op_by_op, record_property, test_metadata, request, capfd
):

    loader_path = test_entry.path
    variant, ModelLoader = test_entry.variant_info

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path):

        # Get the model loader and model info from desired model, variant.
        loader = ModelLoader(variant=variant)
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"Running {request.node.nodeid} - {model_info.name}", flush=True)

        succeeded = False
        try:
            # Only run the actual model test if not marked for skip. The record properties
            # function in finally block will always be called and handles the pytest.skip.
            if test_metadata.status != ModelTestStatus.NOT_SUPPORTED_SKIP:
                tester = DynamicTorchModelTester(
                    run_mode,
                    loader=loader,
                    comparison_config=test_metadata.to_comparison_config(),
                )

                tester.test()
                succeeded = True

        except Exception as e:
            err = capfd.readouterr().err
            # Record runtime failure info so it can be reflected in report properties
            update_test_metadata_for_exception(test_metadata, e, stderr=err)
            raise
        finally:
            # If we mark tests with xfail at collection time, then this isn't hit.
            # Always record properties and handle skip/xfail cases uniformly
            record_model_test_properties(
                record_property,
                request,
                model_info=model_info,
                test_metadata=test_metadata,
                run_mode=run_mode,
                test_passed=succeeded,
            )
