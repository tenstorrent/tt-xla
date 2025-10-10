# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.gpt2.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import GPT2Tester

VARIANT_NAME = ModelVariant.XL
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


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
def test_gpt2_xl_inference(inference_tester: GPT2Tester):
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
def test_gpt2_xl_training(training_tester: GPT2Tester):
    training_tester.test()
