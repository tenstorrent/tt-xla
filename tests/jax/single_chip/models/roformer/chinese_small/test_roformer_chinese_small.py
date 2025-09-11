# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
)
from third_party.tt_forge_models.config import ModelInfo, Parallelism
from third_party.tt_forge_models.roformer.masked_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import RoFormerTester

MODEL_PATH = "junnyu/roformer_chinese_small"
VARIANT_NAME = ModelVariant.CHINESE_SMALL
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)
from third_party.tt_forge_models.roformer.masked_lm.jax import (
    ModelVariant,
    ModelLoader,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> RoFormerTester:
    return RoFormerTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> RoFormerTester:
    return RoFormerTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.PASSED,
)
def test_roformer_chinese_small_inference(inference_tester: RoFormerTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_roformer_chinese_small_training(training_tester: RoFormerTester):
    training_tester.test()
