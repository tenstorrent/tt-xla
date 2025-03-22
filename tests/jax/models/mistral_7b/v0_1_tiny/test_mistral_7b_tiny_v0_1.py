# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from ..tester import MistralTester

MODEL_PATH = "ksmcg/Mistral-tiny"
MODEL_GROUP = ModelGroup.GENERALITY
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.1_tiny",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)

# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MistralTester:
    return MistralTester(MODEL_PATH)


def training_tester() -> MistralTester:
    return MistralTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'ttir.gather' "
        "https://github.com/tenstorrent/tt-xla/issues/318"
    )
)
def test_mistral_7b_tiny_v0_1_inference(inference_tester: MistralTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mistral_7b_tiny_v0_1_training(training_tester: MistralTester):
    training_tester.test()
