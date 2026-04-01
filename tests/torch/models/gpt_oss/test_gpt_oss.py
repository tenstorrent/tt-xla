# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
)

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.gpt_oss.pytorch import ModelVariant

from .tester import GptOssTester

VARIANT_NAME = ModelVariant.GPT_OSS_120B

MODEL_NAME = build_model_name(
    Framework.TORCH,
    "gpt_oss",
    "120B",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Helpers -----


def _create_bfp4_tester() -> GptOssTester:
    """Create inference tester with full BFP4 weight quantization and custom MoE."""
    compiler_config = CompilerConfig(
        experimental_weight_dtype="bfp_bf4",
    )
    return GptOssTester(
        VARIANT_NAME,
        compiler_config=compiler_config,
        inject_custom_moe=True,
    )


def _create_mixed_bfp_tester() -> GptOssTester:
    """Create inference tester for mixed precision: experts BFP4, rest BFP8.

    The current compiler infrastructure only exposes a single global
    ``experimental_weight_dtype`` — per-module overrides are not yet
    supported.  This test compiles the full model at BFP8 as a
    conservative baseline until per-module weight dtype lands.
    """
    compiler_config = CompilerConfig(
        experimental_weight_dtype="bfp_bf8",
    )
    return GptOssTester(
        VARIANT_NAME,
        compiler_config=compiler_config,
        inject_custom_moe=True,
    )


# ----- Tests -----


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.NOT_STARTED,
)
def test_gpt_oss_120b_full_bfp4():
    """GPT-OSS 120B inference with all weights quantized to BFP4."""
    tester = _create_bfp4_tester()
    tester.test()


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.NOT_STARTED,
)
def test_gpt_oss_120b_experts_bfp4_rest_bfp8():
    """GPT-OSS 120B inference — experts at BFP4, everything else at BFP8.

    NOTE: Per-module ``experimental_weight_dtype`` is not yet supported
    in the compiler.  This test currently runs the full model at BFP8 as
    a conservative baseline.  Update ``_create_mixed_bfp_tester`` once
    per-module weight dtype overrides are available.
    """
    tester = _create_mixed_bfp_tester()
    tester.test()
