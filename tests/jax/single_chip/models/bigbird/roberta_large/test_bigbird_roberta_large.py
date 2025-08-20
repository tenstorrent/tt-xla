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
    failed_ttmlir_compilation,
)

from ..tester import BigBirdTester
from third_party.tt_forge_models.bigbird.question_answering.jax.loader import (
    ModelVariant,
)

VARIANT_NAME = ModelVariant.LARGE
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bigbird",
    "roberta_large",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BigBirdTester:
    return BigBirdTester(VARIANT_NAME)


def training_tester() -> BigBirdTester:
    return BigBirdTester(VARIANT_NAME, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "Failed to legalize operation 'ttir.scatter' "
        "https://github.com/tenstorrent/tt-xla/issues/911"
    )
)
def test_bigbird_roberta_large_inference(inference_tester: BigBirdTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_bigbird_roberta_large_training(training_tester: BigBirdTester):
    training_tester.test()
