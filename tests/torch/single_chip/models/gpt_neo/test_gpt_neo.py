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
    failed_ttmlir_compilation,
)

from .tester import GPTNeoTester

VARIANT_NAME = "gpt_neo"


MODEL_NAME = build_model_name(
    Framework.TORCH,
    "gpt_neo",
    "base",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTNeoTester:
    return GPTNeoTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> GPTNeoTester:
    return GPTNeoTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
        "error: Failed to legalize operation 'stablehlo.reduce_window' "
        "https://github.com/tenstorrent/tt-xla/issues/783"
    )
)
def test_torch_gpt_neo_inference(inference_tester: GPTNeoTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_torch_gpt_neo_training(training_tester: GPTNeoTester):
    training_tester.test()
