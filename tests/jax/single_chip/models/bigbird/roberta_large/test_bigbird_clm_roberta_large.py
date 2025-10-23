# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_ttmlir_compilation

from third_party.tt_forge_models.bigbird.causal_lm.jax.loader import (
    ModelLoader,
    ModelVariant,
)
from third_party.tt_forge_models.config import Parallelism

from ..tester import BigBirdCLMTester

VARIANT_NAME = ModelVariant.LARGE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> BigBirdCLMTester:
    return BigBirdCLMTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.gather' that was explicitly marked illegal "
        "https://github.com/tenstorrent/tt-mlir/issues/4795"
    )
)
def test_bigbird_roberta_large_training(training_tester: BigBirdCLMTester):
    training_tester.test()
