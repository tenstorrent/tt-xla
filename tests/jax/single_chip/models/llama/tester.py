# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import FlaxLlamaForCausalLM, FlaxPreTrainedModel, LlamaTokenizer


class LLamaTester(ModelTester):
    """Tester for Llama models."""

    # Note: Llama variants are not commonly used for anything else but auto-regressive text generation.

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
        return FlaxLlamaForCausalLM.from_pretrained(self._model_name, from_pt=True)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = LlamaTokenizer.from_pretrained(self._model_name)
        inputs = tokenizer("Hello", return_tensors="np")
        return inputs["input_ids"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }
