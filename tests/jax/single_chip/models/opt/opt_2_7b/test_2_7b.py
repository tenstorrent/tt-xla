# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.opt.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import OPTTester

VARIANT_NAME = ModelVariant._2_7B
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----

@pytest.fixture
def training_tester() -> OPTTester:
    return OPTTester(VARIANT_NAME, run_mode=RunMode.TRAINING)

# ----- Tests -----

@pytest.mark.xfail(
    reason=failed_runtime(
        "OOM on device issues due to consteval - https://github.com/tenstorrent/tt-xla/issues/1447"
    )
)

@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.SINGLE_DEVICE,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.FORWARD,
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
