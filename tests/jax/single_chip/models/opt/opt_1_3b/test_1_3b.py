# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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

from third_party.tt_forge_models.opt.causal_lm.jax import (
    ModelVariant,
    ModelLoader,
)
from ..tester import OPTTester

MODEL_PATH = "facebook/opt-1.3b"
VARIANT_NAME = ModelVariant._1_3B
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OPTTester:
    return OPTTester(MODEL_PATH)


@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    parallelism=Parallelism.SINGLE_DEVICE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "AssertionError: PCC comparison failed. Calculated: pcc=0.3758324682712555. Required: pcc=0.99. "
        "https://github.com/tenstorrent/tt-xla/issues/379"
    )
)
def test_opt_1_3b_inference(inference_tester: OPTTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_opt_1_3b_training(training_tester: OPTTester):
    training_tester.test()
