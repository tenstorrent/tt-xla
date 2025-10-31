# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ExecutionPass, failed_runtime

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.xglm.causal_lm.jax import ModelLoader, ModelVariant

from ..tester import XGLMTester

VARIANT_NAME = ModelVariant._564M
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)

# ----- Fixtures -----


@pytest.fixture
def training_tester() -> XGLMTester:
    return XGLMTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.training
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    run_mode=RunMode.TRAINING,
    parallelism=Parallelism.SINGLE_DEVICE,
    execution_pass=ExecutionPass.FORWARD,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Invalid data size. numElements * elementSize == data->size(). "
        "Issue: https://github.com/tenstorrent/tt-xla/issues/1313"
    )
)
def test_xglm_564m_training(training_tester: XGLMTester):
    training_tester.test()
