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
    failed_ttmlir_compilation,
)

from ..tester import MarianTester

MODEL_PATH = "Helsinki-NLP/opus-mt-en-de"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "marian",
    "opus_mt_en_de",
    ModelTask.NLP_TEXT_CLS,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MarianTester:
    return MarianTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> MarianTester:
    return MarianTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386"
    )
)
def test_marian_opus_mt_en_de_inference(inference_tester: MarianTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_marian_opus_mt_en_de_training(training_tester: MarianTester):
    training_tester.test()
