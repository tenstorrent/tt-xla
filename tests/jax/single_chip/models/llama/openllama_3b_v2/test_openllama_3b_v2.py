# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.llama.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import LLamaTester

VARIANT_NAME = ModelVariant._3B_V2
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----

@pytest.fixture
def training_tester() -> LLamaTester:
    return LLamaTester(VARIANT_NAME, RunMode.TRAINING)

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
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.large
@pytest.mark.xfail(
    reason=failed_runtime(
        "OOM on device issues due to consteval - https://github.com/tenstorrent/tt-xla/issues/1447"
    )
)
def test_openllama3b_training(training_tester: LLamaTester):
    training_tester.test()
