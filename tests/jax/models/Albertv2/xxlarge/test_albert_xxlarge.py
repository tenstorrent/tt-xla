# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import pytest
from flax import linen as nn
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import AutoTokenizer, FlaxAlbertForMaskedLM

MODEL = "albert/albert-xxlarge-v2"


class FlaxAlbertForMaskedLMTester(ModelTester):
    """Tester for Albert model with a `language modeling` head on top."""

    # @override
    @staticmethod
    def _get_model() -> nn.Module:
        return FlaxAlbertForMaskedLM.from_pretrained(MODEL)

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "__call__"

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
    def _get_forward_method_args(self):
        return self._get_input_activations()

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {"params": self._model.params}


# ----- Fixtures -----


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    return config


@pytest.fixture
def inference_tester(
    comparison_config: ComparisonConfig,
) -> FlaxAlbertForMaskedLMTester:
    return FlaxAlbertForMaskedLMTester(comparison_config)


@pytest.fixture
def training_tester(
    comparison_config: ComparisonConfig,
) -> FlaxAlbertForMaskedLMTester:
    return FlaxAlbertForMaskedLMTester(comparison_config, TestType.TRAINING)



# ----- Tests -----


@pytest.mark.skip(reason="failed to legalize operation 'stablehlo.dot_general'")
def test_flax_albert_for_masked_lm_inference(
    inference_tester: FlaxAlbertForMaskedLMTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_albert_for_masked_lm_training(
    training_tester: FlaxAlbertForMaskedLMTester,
):
    training_tester.test()
