# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import AutoTokenizer, FlaxDistilBertForMaskedLM

MODEL = "distilbert/distilbert-base-uncased"

# ----- Tester -----


class FlaxDistilBertForMaskedLMTester(ModelTester):
    """Tester for DistilBert model with a `language modeling` head on top."""

    # @override
    @staticmethod
    def _get_model() -> nn.Module:
        return FlaxDistilBertForMaskedLM.from_pretrained(MODEL)

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "__call__"

    # @override
    @staticmethod
    def _get_input_activations() -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer("Hello [MASK].", return_tensors="np")
        return [inputs["input_ids"]]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        input_activations = self._get_input_activations()

        assert len(input_activations) == 1
        assert hasattr(self._model, "params")

        return {"input_ids": input_activations[0], "params": self._model.params}


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester()


@pytest.fixture
def training_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.skip(reason="failed to legalize operation 'stablehlo.dot_general'")
def test_flax_distil_bert_for_masked_lm_inference(
    inference_tester: FlaxDistilBertForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_distil_bert_for_masked_lm_training(
    training_tester: FlaxDistilBertForMaskedLMTester,
):
    training_tester.test()
