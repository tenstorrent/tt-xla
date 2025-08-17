# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import gc
from tests.utils import skip_full_eval_test
from tt_torch.tools.utils import CompilerConfig, CompileDepth
from tests.runner.test_utils import (
    ModelStatus,
    import_model_loader_and_variant,
    DynamicTester,
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
        model_info = ModelLoader.get_model_info(variant=variant)
        print(f"model_name: {model_info.name} status: {test_metadata.status}")

        if test_metadata.status == ModelStatus.NOT_SUPPORTED_SKIP:
            skip_full_eval_test(
                record_property,
                cc,
                model_info.name,
                bringup_status=test_metadata.skip_bringup_status,
                reason=test_metadata.skip_reason,
                model_group=model_info.group,
                forge_models_test=True,
            )

        tester = DynamicTester(
            model_info.name,
            mode,
            loader=loader,
            model_info=model_info,
            compiler_config=cc,
            record_property_handle=record_property,
            forge_models_test=True,
            **test_metadata.to_tester_args(),
        )

        results = tester.test_model()
        tester.finalize()

    # Cleanup memory after each test to prevent memory leaks
    gc.collect()
