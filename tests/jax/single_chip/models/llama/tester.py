# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import FlaxLlamaForCausalLM, FlaxPreTrainedModel, LlamaTokenizer


class LLamaTester(JaxModelTester):
    """Tester for Llama models."""

    # Note: Llama variants are not commonly used for anything else but auto-regressive text generation.

    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxLlamaForCausalLM.from_pretrained(self._model_path, from_pt=True)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = LlamaTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello there fellow traveler", return_tensors="jax")
        return inputs
