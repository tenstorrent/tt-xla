# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from infra import ComparisonConfig, Framework, RunMode
from transformers import (
    AutoTokenizer,
    FlaxBigBirdForQuestionAnswering,
    FlaxPreTrainedModel,
)
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

MODEL_PATH = "google/bigbird-base-trivia-itc"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bigbird",
    "roberta_base",
    ModelTask.NLP_QA,
    ModelSource.HUGGING_FACE,
)


class BigBirdBaseTester(BigBirdTester):
    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(model_path, comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxBigBirdForQuestionAnswering.from_pretrained(
            self._model_path, attention_type="original_full"
        )

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        inputs = tokenizer(question, text, return_tensors="jax")
        return inputs


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BigBirdBaseTester:
    return BigBirdBaseTester(MODEL_PATH)


def training_tester() -> BigBirdBaseTester:
    return BigBirdBaseTester(MODEL_PATH, run_mode=RunMode.TRAINING)


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
        "'ttir.scatter' op Dimension size to slice into must be 1 "
        "https://github.com/tenstorrent/tt-xla/issues/386"
    )
)
def test_bigbird_roberta_base_inference(inference_tester: BigBirdBaseTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_bigbird_roberta_base_training(inference_tester: BigBirdBaseTester):
    training_tester.test()
