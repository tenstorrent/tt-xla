# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import (
    BringupStatus,
    Category,
    failed_runtime,
)

from ..tester import OPTTester
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.opt.causal_lm.jax import ModelVariant, ModelLoader

VARIANT_NAME = ModelVariant._2_7B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> OPTTester:
    return OPTTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "OOM on device issues due to consteval - https://github.com/tenstorrent/tt-xla/issues/1447"
    )
)
def test_opt_2_7b_inference(inference_tester: OPTTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "OOM on device issues due to consteval - https://github.com/tenstorrent/tt-xla/issues/1447"
    )
)
def test_opt_2_7b_training(training_tester: OPTTester):
    training_tester.test()
