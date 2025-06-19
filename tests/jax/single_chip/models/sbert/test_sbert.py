# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from flax import linen as nn
from transformers import AutoTokenizer, FlaxBertModel, BertConfig

from infra import ComparisonConfig, Framework, ModelTester, RunMode
from tests.utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_fe_compilation,
)

from .model_implementation import FlaxSentenceTransformerBERT

MODEL_PATH = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "sentence-transformer",
    "bert",
    ModelTask.NLP_TEXT_CLS,
    ModelSource.HUGGING_FACE,
)

# ----- Tester -----


class FlaxSBERTTester(ModelTester):
    """Tester for a SentenceTransformer BERT model"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        config = BertConfig.from_pretrained(self._model_path)
        return FlaxSentenceTransformerBERT(config=config)

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        # The model weights used in this test were trained on Turkish sentence pairs.
        # These sentences below match the training data distribution.
        inputs = tokenizer(
            ["Bu örnek bir cümle", "Her cümle vektöre çevriliyor"],
            padding=True,
            truncation="longest_first",
            max_length=512,
            return_tensors="jax",
        )
        return inputs

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        pretrained_model = FlaxBertModel.from_pretrained(self._model_path, from_pt=True)
        params = pretrained_model.params

        inputs = self._get_input_activations()
        return {
            "variables": params,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids"),
        }


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxSBERTTester:
    return FlaxSBERTTester()


@pytest.fixture
def training_tester() -> FlaxSBERTTester:
    return FlaxSBERTTester(run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.push
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_FE_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_fe_compilation(
        "Failed to legalize operation 'stablehlo.shift_right_logical'. "
        "https://github.com/tenstorrent/tt-xla/issues/417"
    )
)
def test_flax_sbert_inference(inference_tester: FlaxSBERTTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.RED,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_sbert_training(training_tester: FlaxSBERTTester):
    training_tester.test()
