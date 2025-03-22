# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxMistralForCausalLM,
    FlaxPreTrainedModel,
    MistralConfig,
)


class Mistral7BTester(ModelTester):
    """Tester for Mistral-7B model variants with a language modeling head on top."""

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxMistralForCausalLM.from_pretrained(self._model_name)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        inputs = tokenizer("Hello ", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }


class Mistral7BV02Tester(Mistral7BTester):
    """Tester for Mistral-7B model v0.2 and later variants, with a language modeling
    head on top."""

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        # Initializing model with random weights because the official weights are
        # gated. From v0.2 version of Mistral-7B sliding window attention was removed,
        # but Transformers Flax implementation of Mistral-7B wasn't updated to take
        # that into account so it expects to have a sliding_window set in the config.
        # Using custom config in order to change the sliding_window from Null to
        # full context window length, effectively achieving the same result as if there
        # was no sliding window mask.
        config = MistralConfig.from_pretrained(self._model_name)
        config.sliding_window = config.max_position_embeddings
        return FlaxMistralForCausalLM(config)
