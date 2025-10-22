# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    failed_ttmlir_compilation,
    incorrect_result,
)

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.vision_text_dual_encoder.mm_image_ttt.jax import (
    ModelLoader,
    ModelVariant,
)

from .tester import VisionTextDualEncoderTester

VARIANT = ModelVariant.BASE

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> VisionTextDualEncoderTester:
    return VisionTextDualEncoderTester(VARIANT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: pcc=nan. Required: pcc=0.99. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_vision_text_dual_encoder_training(
    training_tester: VisionTextDualEncoderTester,
):
    training_tester.test()
