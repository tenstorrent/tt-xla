# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ModelTester, RunMode
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel

MODEL_PATH = "openai-community/gpt2"


class GPT2Tester(ModelTester):
    """Tester for GPT2 for autoregressive text generation."""

    # @override
    def _get_model(self) -> nn.Module:
        return FlaxGPT2LMHeadModel.from_pretrained(MODEL_PATH)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        inputs = tokenizer("Hello", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> GPT2Tester:
    return GPT2Tester()


@pytest.fixture
def training_tester() -> GPT2Tester:
    return GPT2Tester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(reason="failed to legalize operation 'stablehlo.reduce'")
def test_gpt2_inference(
    inference_tester: GPT2Tester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_gpt2_training(
    training_tester: GPT2Tester,
):
    training_tester.test()
