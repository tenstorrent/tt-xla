# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Dict

import jax
import pytest
from flax import linen as nn
from infra import ModelTester, ComparisonConfig, RunMode
from transformers import AutoTokenizer, FlaxGPT2LMHeadModel

MODEL = "openai-community/gpt2"


class GPT2Tester(ModelTester):
    """Tester for GPT2."""

    @staticmethod
    def _get_model() -> nn.Module:
        return FlaxGPT2LMHeadModel.from_pretrained(MODEL)

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "__call__"

    # @override
    @staticmethod
    def _get_input_activations() -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        inputs = tokenizer("Hello", return_tensors="np")
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
) -> GPT2Tester:
    return GPT2Tester(comparison_config)


@pytest.fixture
def training_tester(
    comparison_config: ComparisonConfig,
) -> GPT2Tester:
    return GPT2Tester(comparison_config, RunMode.TRAINING)



# ----- Tests -----


# @pytest.mark.skip(reason="failed to legalize operation 'stablehlo.dot_general'")
def test_flax_gpt2_with_lm_headinference(
    inference_tester: GPT2Tester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_gp2_with_lm_head_training(
    training_tester: GPT2Tester,
):
    training_tester.test()
