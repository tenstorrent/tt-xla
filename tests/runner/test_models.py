# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import gc
from tests.runner.test_utils import (
    ModelStatus,
    import_model_loader_and_variant,
    DynamicTorchModelTester,
    setup_test_discovery,
    create_test_id_generator,
)
from tests.runner.requirements import RequirementsManager

# Setup test discovery using utility functions
TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", ".."))
MODELS_ROOT, test_entries = setup_test_discovery(PROJECT_ROOT)


@pytest.mark.parametrize(
    "mode",
    ["eval"],
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
def test_all_models(test_entry, mode, op_by_op, record_property, test_metadata):
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

        # Use the variant from the test_entry parameter
        loader = ModelLoader(variant=variant)

        # Get model name from the ModelLoader's ModelInfo
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"model_name: {model_info.name} status: {test_metadata.status}")

        # FIXME - Add some support for skipping tests.
        # if test_metadata.status == ModelStatus.NOT_SUPPORTED_SKIP:
        #     skip_full_eval_test(
        #         record_property,
        #         model_info.name,
        #         bringup_status=test_metadata.skip_bringup_status,
        #         reason=test_metadata.skip_reason,
        #         model_group=model_info.group,
        #         forge_models_test=True,
        #     )

        tester = DynamicTorchModelTester(
            mode,
            loader=loader,
            **test_metadata.to_tester_args(),
        )

        tester.test()

    # Cleanup memory after each test to prevent memory leaks
    gc.collect()
