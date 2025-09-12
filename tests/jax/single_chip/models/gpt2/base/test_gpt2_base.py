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
)
from third_party.tt_forge_models.gpt2.causal_lm.jax import ModelVariant
from ..tester import GPT2Tester

MODEL_VARIANT = ModelVariant.BASE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "gpt2",
    "base",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_VARIANT)


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(MODEL_VARIANT, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_gpt2_base_inference(inference_tester: GPT2Tester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
    execution_pass=ExecutionPass.BACKWARD,
)
@pytest.mark.xfail(reason="error: failed to legalize operation 'ttir.scatter'")
def test_gpt2_base_training(training_tester: GPT2Tester):
    training_tester.test()
