# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ModelTester, RunMode
from transformers import AutoTokenizer, FlaxDistilBertForMaskedLM
from utils import record_model_test_properties, runtime_fail

MODEL_PATH = "distilbert/distilbert-base-uncased"
MODEL_NAME = MODEL_PATH.split("/")[1]

# ----- Tester -----


class FlaxDistilBertForMaskedLMTester(ModelTester):
    """Tester for DistilBert model on a masked language modeling task"""

    # @override
    def _get_model(self) -> nn.Module:
        return FlaxDistilBertForMaskedLM.from_pretrained(MODEL_PATH)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        inputs = tokenizer("Hello [MASK].", return_tensors="np")
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
def inference_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester()


@pytest.fixture
def training_tester() -> FlaxDistilBertForMaskedLMTester:
    return FlaxDistilBertForMaskedLMTester(RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.xfail(
    reason=runtime_fail(
        "Cannot get the device from a tensor with host storage "
        "(https://github.com/tenstorrent/tt-xla/issues/171)"
    )
)
def test_flax_distilbert_inference(
    inference_tester: FlaxDistilBertForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_distilbert_training(
    training_tester: FlaxDistilBertForMaskedLMTester,
    record_tt_xla_property: Callable,
):
    record_model_test_properties(record_tt_xla_property, MODEL_NAME)

    training_tester.test()
