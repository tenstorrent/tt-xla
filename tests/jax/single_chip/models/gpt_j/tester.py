# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxPreTrainedModel,
    FlaxGPTJForCausalLM,
)


class GPTJTester(ModelTester):
    """Tester for GPT-J models."""

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
        return FlaxGPTJForCausalLM.from_pretrained(self._model_name)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
        return inputs["input_ids"]

    # @overridde
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "input_ids": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
