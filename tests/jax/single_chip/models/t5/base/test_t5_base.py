# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

from third_party.tt_forge_models.t5.summarization.jax.loader import ModelVariant

from ..tester import T5Tester

VARIANT_NAME = ModelVariant.BASE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "t5",
    str(VARIANT_NAME),
    ModelTask.NLP_SUMMARIZATION,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> T5Tester:
    return T5Tester(VARIANT_NAME)


@pytest.fixture
def training_tester() -> T5Tester:
    return T5Tester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.INCORRECT_RESULT,
)
@pytest.mark.xfail(
    reason=incorrect_result(
        "PCC comparison failed on Blackhole. Calculated: pcc=0.7276687622070312. Required: pcc=0.99."
        "https://github.com/tenstorrent/tt-xla/issues/1038"
    )
)
def test_t5_base_inference(inference_tester: T5Tester):
    inference_tester.test()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_t5_base_training(training_tester: T5Tester):
    training_tester.test()
