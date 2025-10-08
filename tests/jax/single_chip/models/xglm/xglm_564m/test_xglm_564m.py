# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ExecutionPass,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_runtime,
)
from third_party.tt_forge_models.xglm.causal_lm.jax.loader import ModelVariant
from ..tester import XGLMTester

VARIANT_NAME = ModelVariant._564M
MODEL_NAME = build_model_name(
    Framework.JAX,
    "xglm",
    str(VARIANT_NAME),
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> XGLMTester:
    return XGLMTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> XGLMTester:
    return XGLMTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
@pytest.mark.xfail(
    reason=failed_runtime(
        "Invalid data size. numElements * elementSize == data->size(). "
        "Issue: https://github.com/tenstorrent/tt-xla/issues/1313"
    )
)
def test_xglm_564m_inference(inference_tester: XGLMTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
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
