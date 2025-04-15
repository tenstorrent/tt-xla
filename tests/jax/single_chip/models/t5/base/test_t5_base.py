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

from ..tester import T5Tester

MODEL_PATH = "google-t5/t5-base"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "t5",
    "base",
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> T5Tester:
    return T5Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> T5Tester:
    return T5Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.skip(
    reason=failed_runtime(
        "ttnn::operations::binary::BinaryDeviceOperation: unsupported broadcast "
        "(https://github.com/tenstorrent/tt-xla/issues/505)"
    )
)
def test_t5_base_inference(inference_tester: T5Tester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_t5_base_training(training_tester: T5Tester):
    training_tester.test()
