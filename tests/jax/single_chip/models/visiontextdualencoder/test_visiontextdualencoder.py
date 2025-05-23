# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from .tester import VisionTextDualEncoderTester

IMAGE_MODEL_PATH = "google/vit-base-patch16-224"
TEXT_MODEL_PATH = "google-bert/bert-base-uncased"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "vision_text_dual_encoder",
    "vit_base_patch16_224_bert_base",
    ModelTask.MM_IMAGE_TTT,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> VisionTextDualEncoderTester:
    return VisionTextDualEncoderTester(IMAGE_MODEL_PATH, TEXT_MODEL_PATH)


@pytest.fixture
def training_tester() -> VisionTextDualEncoderTester:
    return VisionTextDualEncoderTester(
        IMAGE_MODEL_PATH, TEXT_MODEL_PATH, RunMode.TRAINING
    )


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "Atol comparison failed. Calculated: atol=6632.052734375. Required: atol=0.16. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_vision_text_dual_encoder_inference(
    inference_tester: VisionTextDualEncoderTester,
):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_vision_text_dual_encoder_training(
    training_tester: VisionTextDualEncoderTester,
):
    training_tester.test()
