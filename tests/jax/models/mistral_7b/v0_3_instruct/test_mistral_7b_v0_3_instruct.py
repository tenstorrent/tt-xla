# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from transformers import FlaxMistralForCausalLM, FlaxPreTrainedModel, MistralConfig

from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_fe_compilation,
)

from ..tester import MistralTester

MODEL_PATH = "unsloth/mistral-7b-instruct-v0.3"
MODEL_GROUP = ModelGroup.RED
MODEL_NAME = build_model_name(
    Framework.JAX,
    "mistral-7b",
    "v0.3_instruct",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


# ----- Tester -----


class Mistral7BV03Tester(MistralTester):

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        # Initializing model with random weights because the official weights are
        # gated. From v0.2 version of Mistral-7B sliding window attention was removed,
        # but Transformers Flax implementation of Mistral-7B wasn't updated to take
        # that into account so it expects to have a sliding_window set in the config.
        # Using custom config in order to change the sliding_window from Null to
        # full context window length, effectively achieving same result since attention
        # mask will not be masked.
        config = MistralConfig.from_pretrained(self._model_name)
        config.sliding_window = 32768
        return FlaxMistralForCausalLM(config)


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> MistralTester:
    return Mistral7BV03Tester(MODEL_PATH)


def training_tester() -> MistralTester:
    return Mistral7BV03Tester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.skip(
    reason=failed_fe_compilation(
        "OOMs in CI (https://github.com/tenstorrent/tt-xla/issues/186)"
    )
)
def test_mistral_7b_v0_3_instruct_inference(inference_tester: Mistral7BV03Tester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=MODEL_GROUP,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_mistral_7b_v0_3_instruct_training(training_tester: Mistral7BV03Tester):
    training_tester.test()
