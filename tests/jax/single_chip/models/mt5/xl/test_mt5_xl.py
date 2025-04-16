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

from ..tester import MT5Tester

MODEL_PATH = "google/mt5-xl"
MODEL_NAME = build_model_name(
    Framework.JAX, "mt5", "xl", ModelTask.NLP_SUMMARIZATION, ModelSource.HUGGING_FACE
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MT5Tester:
    return MT5Tester(MODEL_PATH)


@pytest.fixture
def training_tester() -> MT5Tester:
    return MT5Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


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
        "Out of Memory: Not enough space to allocate 2048917504 B DRAM buffer "
        "across 12 banks, where each bank needs to store 170745856 B "
        "(https://github.com/tenstorrent/tt-xla/issues/187)"
    )
)
def test_mt5_xl_inference(inference_tester: MT5Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mt5_xl_training(training_tester: MT5Tester):
    training_tester.test()
