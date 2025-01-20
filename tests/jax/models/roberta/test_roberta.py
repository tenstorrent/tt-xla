# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ModelTester, RunMode
from transformers import AutoTokenizer, FlaxRobertaForMaskedLM

MODEL_PATH = "FacebookAI/roberta-base"

# ----- Tester -----


class FlaxRobertaForMaskedLMTester(ModelTester):
    """Tester for Roberta model on a masked language modeling task."""

    # @override
    def _get_model(self) -> nn.Module:
        return FlaxRobertaForMaskedLM.from_pretrained(MODEL_PATH)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        inputs = tokenizer("Hello <mask>.", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }

    # @ override
    def _get_static_argnames(self):
        return ["train"]


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester()


@pytest.fixture
def training_tester() -> FlaxRobertaForMaskedLMTester:
    return FlaxRobertaForMaskedLMTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(reason="failed to legalize operation 'stablehlo.reduce_window'")
def test_roberta_inference(
    inference_tester: FlaxRobertaForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_roberta_training(
    training_tester: FlaxRobertaForMaskedLMTester,
):
    training_tester.test()
