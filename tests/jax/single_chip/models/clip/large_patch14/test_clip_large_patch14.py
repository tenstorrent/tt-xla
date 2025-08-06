# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
    failed_runtime,
)

from ..tester import FlaxCLIPTester

MODEL_PATH = "openai/clip-vit-large-patch14"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "clip_vit_patch14",
    "large",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> FlaxCLIPTester:
    return FlaxCLIPTester(MODEL_PATH, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "BinaryOpType cannot be mapped to BcastOpMath "
        "https://github.com/tenstorrent/tt-xla/issues/288"
    )
)
def test_clip_large_patch14_inference(inference_tester: FlaxCLIPTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_clip_large_patch14_training(training_tester: FlaxCLIPTester):
    training_tester.test()
