# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Mapping

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxLongT5ForConditionalGeneration,
    FlaxPreTrainedModel,
)


class LongT5Tester(JaxModelTester):
    """Tester for Long T5 model variants."""

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
        return FlaxLongT5ForConditionalGeneration.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer(
            "My friends are cool but they eat too many carbs.", return_tensors="jax"
        )
        return inputs

    # @override
    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        decoder_input_ids = tokenizer(
            text_target="I eat less carbs.", return_tensors="jax"
        ).input_ids
        return {
            "params": self._input_parameters,
            "decoder_input_ids": decoder_input_ids,
            **self._input_activations,
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
