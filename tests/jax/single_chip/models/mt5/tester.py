# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxPreTrainedModel,
    FlaxMT5ForConditionalGeneration,
)
from jaxtyping import PyTree


class MT5Tester(ModelTester):
    """Tester for mT5 models."""

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
        return FlaxMT5ForConditionalGeneration.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer(
            "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien.",
            return_tensors="jax",
        )
        return inputs

    # @overridde
    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        assert hasattr(self._model, "params")
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        decoder_input_ids = tokenizer(
            text_target="Weiter Verhandlung in Syrien.", return_tensors="jax"
        ).input_ids
        return {
            "params": self._model.params,
            "decoder_input_ids": decoder_input_ids,
            **self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
