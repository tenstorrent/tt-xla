# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from jaxtyping import PyTree
from transformers import (
    AutoTokenizer,
    FlaxBlenderbotForConditionalGeneration,
    FlaxPreTrainedModel,
)


class BlenderBotTester(JaxModelTester):
    """Tester for BlenderBot models."""

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
        model = FlaxBlenderbotForConditionalGeneration.from_pretrained(
            self._model_path, from_pt=True
        )
        model.params = model.to_bf16(model.params)
        return model

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        return tokenizer(
            "My friends are cool but they eat too many carbs.",
            truncation=True,
            return_tensors="jax",
        )

    # @override
    def _get_static_argnames(self):
        return ["train"]
