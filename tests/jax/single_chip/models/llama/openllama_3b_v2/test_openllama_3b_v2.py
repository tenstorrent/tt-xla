# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    incorrect_result,
)
from third_party.tt_forge_models.config import Parallelism

from third_party.tt_forge_models.llama.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import LLamaTester

VARIANT_NAME = ModelVariant._3B_V2
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> LLamaTester:
    return LLamaTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> LLamaTester:
    return LLamaTester(VARIANT_NAME, RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=incorrect_result(
        "AssertionError: PCC comparison failed. Calculated: pcc=0.9683969616889954. Required: pcc=0.99. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_openllama3b_inference(inference_tester: LLamaTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.large
@pytest.mark.skip(reason="Support for training not implemented")
def test_openllama3b_training(training_tester: LLamaTester):
    training_tester.test()
