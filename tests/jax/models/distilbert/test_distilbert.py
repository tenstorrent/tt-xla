# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from infra import ModelTester, RunMode
from transformers import AutoTokenizer, FlaxDistilBertForMaskedLM, FlaxPreTrainedModel
from utils import runtime_fail

MODEL_PATH = "distilbert/distilbert-base-uncased"
MODEL_NAME = "distilbert"

# ----- Tester -----


class FlaxDistilBertForMaskedLMTester(ModelTester):
    """Tester for DistilBert model on a masked language modeling task"""

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
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


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.xfail(
    reason=runtime_fail(
        "Host data with total size 20B does not match expected size 10B of device buffer! "
        "(https://github.com/tenstorrent/tt-xla/issues/182)"
    )
)
def test_flax_distilbert_inference(inference_tester: FlaxDistilBertForMaskedLMTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category="model_test",
    model_name=MODEL_NAME,
    run_mode=RunMode.INFERENCE.value,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_distilbert_training(training_tester: FlaxDistilBertForMaskedLMTester):
    training_tester.test()
