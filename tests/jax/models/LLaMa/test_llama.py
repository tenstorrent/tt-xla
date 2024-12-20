# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Dict

import jax
import pytest
from flax import linen as nn
from infra import ModelTester, ComparisonConfig, RunMode
from transformers import LlamaTokenizer, FlaxLlamaForCausalLM

MODEL = "openlm-research/open_llama_3b_v2"


class LLamaTester(ModelTester):
    """Tester for a Llama model."""

    @staticmethod
    def _get_model() -> nn.Module:
        return FlaxLlamaForCausalLM.from_pretrained(MODEL, from_pt=True)

    # @override
    @staticmethod
    def _get_forward_method_name() -> str:
        return "__call__"

    # @override
    @staticmethod
    def _get_input_activations() -> Sequence[jax.Array]:
        tokenizer = LlamaTokenizer.from_pretrained(MODEL)
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
) -> LLamaTester:
    return LLamaTester(comparison_config)


@pytest.fixture
def training_tester(
    comparison_config: ComparisonConfig,
) -> LLamaTester:
    return LLamaTester(comparison_config, RunMode.TRAINING)



# ----- Tests -----


@pytest.mark.skip(reason="failed to legalize operation 'stablehlo.dot_general'")
def test_flax_gpt2_with_lm_headinference(
    inference_tester: LLamaTester,
):
    inference_tester.test()


@pytest.mark.skip(reason="Support for training not implemented")
def test_flax_gp2_with_lm_head_training(
    training_tester: LLamaTester,
):
    training_tester.test()
