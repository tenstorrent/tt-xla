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
    failed_runtime,
)

from ..tester import Dinov2Tester

MODEL_PATH = "facebook/dinov2-base"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "dinov2",
    "base",
    ModelTask.CV_IMAGE_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> Dinov2Tester:
    return Dinov2Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> Dinov2Tester:
    return Dinov2Tester(MODEL_PATH, RunMode.TRAINING)


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
        "input_tensor_a.get_padded_shape().rank() == this->slice_start.rank() && this->slice_start.rank() == this->slice_end.rank() "
        "(https://github.com/tenstorrent/tt-xla/issues/535)"
    )
)
def test_dinov2_base_inference(inference_tester: Dinov2Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_dinov2_base_training(training_tester: Dinov2Tester):
    training_tester.test()
