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
    failed_ttmlir_compilation,
)

from third_party.tt_forge_models.gpt_neo.causal_lm.jax import ModelVariant

from ..tester import GPTNeoTester

VARIANT_NAME = ModelVariant.GPT_NEO_125M
MODEL_NAME = build_model_name(
    Framework.JAX,
    "gpt_neo",
    "125m",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPTNeoTester:
    return GPTNeoTester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> GPTNeoTester:
    return GPTNeoTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


# This test specifically is somewhat flaky, it failed and then returned to passing
# without any apparent reason for that.
@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_gpt_neo_125m_inference(inference_tester: GPTNeoTester):
    inference_tester.test()


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    execution_pass=ExecutionPass.BACKWARD,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "error: failed to legalize operation 'ttir.scatter'"
        "https://github.com/tenstorrent/tt-mlir/issues/4792"
    )
)
def test_gpt_neo_125m_training(training_tester: GPTNeoTester):
    training_tester.test()
