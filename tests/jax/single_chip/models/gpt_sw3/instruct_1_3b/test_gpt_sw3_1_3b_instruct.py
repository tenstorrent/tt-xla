# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.gpt_sw3.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import GPTSw3Tester

VARIANT_NAME = ModelVariant.INSTRUCT_1_3B

MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTSw3Tester:
    return GPTSw3Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> GPTSw3Tester:
    return GPTSw3Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.large
def test_gpt_sw3_1_3b_instruct_inference(inference_tester: GPTSw3Tester):
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
def test_gpt_sw3_1_3b_instruct_training(training_tester: GPTSw3Tester):
    training_tester.test()
