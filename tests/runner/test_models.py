# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import gc
from tt_torch.tools.utils import CompilerConfig, CompileDepth
from tests.runner.test_utils import (
    ModelStatus,
    import_model_loader_and_variant,
    DynamicTester,
    setup_test_discovery,
    create_test_id_generator,
    record_model_test_properties,
)
from tests.runner.requirements import RequirementsManager
from infra import RunMode

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
    ids=["full"],
    # FIXME - Consider adding when op-by-op flow is working/supported in tt-xla.
    # [OpByOpBackend.STABLEHLO, OpByOpBackend.TORCH, None],
    # ids=["op_by_op_stablehlo", "op_by_op_torch", "full"],
)
@pytest.mark.parametrize(
    "test_entry",
    test_entries,
    ids=create_test_id_generator(MODELS_ROOT),
)
def test_all_models(
    test_entry, run_mode, op_by_op, record_property, test_metadata, request
):
    loader_path = test_entry["path"]
    variant_info = test_entry["variant_info"]

    # Ensure per-model requirements are installed, and roll back after the test
    with RequirementsManager.for_loader(loader_path):
        # FIXME - Consider cleaning this up, avoid call to import_model_loader_and_variant.
        if variant_info:
            # Unpack the tuple we stored earlier
            variant, ModelLoader, ModelVariant = variant_info
        else:
            # For models without variants
            ModelLoader, _ = import_model_loader_and_variant(loader_path, MODELS_ROOT)
            variant = None

        cc = CompilerConfig()
        cc.enable_consteval = True
        cc.consteval_parameters = True
        # FIXME - Add back when op-by-op flow is working/supported in tt-xla.
        # if op_by_op:
        #     cc.compile_depth = CompileDepth.EXECUTE_OP_BY_OP
        #     cc.op_by_op_backend = op_by_op

        # Use the variant from the test_entry parameter
        loader = ModelLoader(variant=variant)

        # Get model name from the ModelLoader's ModelInfo
        # FIXME - Consider catching exceptions here and still reporting on failed tests.
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"model_name: {model_info.name} status: {test_metadata.status}")

        try:
            # Only run the actual model test if not marked for skip. The record properties
            # function in finally block will always be called and handles the pytest.skip.
            if test_metadata.status != ModelStatus.NOT_SUPPORTED_SKIP:
                tester = DynamicTester(
                    model_info.name,
                    run_mode,
                    loader=loader,
                    model_info=model_info,
                    compiler_config=cc,
                    record_property_handle=record_property,
                    forge_models_test=True,
                    **test_metadata.to_tester_args(),
                )

                # results = tester.test_model()
                # tester.finalize()
                # FIXME - Consider catching exceptions here and using as failed reason

        finally:
            # If we mark tests with xfail at collection time, then this isn't hit.
            # Always record properties and handle skip/xfail cases uniformly
            record_model_test_properties(
                record_property,
                request,
                model_info=model_info,
                test_metadata=test_metadata,
                run_mode=run_mode,
            )

    # Cleanup memory after each test to prevent memory leaks
    gc.collect()
