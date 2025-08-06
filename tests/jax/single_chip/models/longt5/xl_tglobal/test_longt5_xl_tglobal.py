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
    failed_fe_compilation,
)

from ..tester import LongT5Tester

MODEL_PATH = "google/long-t5-tglobal-xl"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "longt5",
    "xl_tglobal",
    ModelTask.NLP_TEXT_CLS,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LongT5Tester:
    return LongT5Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> LongT5Tester:
    return LongT5Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "Fatal Python error: Floating point exception "
        "https://github.com/tenstorrent/tt-xla/issues/463"
    )
)
def test_longt5_xl_tglobal_inference(inference_tester: LongT5Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_longt5_xl_tglobal_training(training_tester: LongT5Tester):
    training_tester.test()
