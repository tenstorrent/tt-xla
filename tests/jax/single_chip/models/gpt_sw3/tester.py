# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import FlaxGPT2LMHeadModel, FlaxPreTrainedModel, GPTSw3Tokenizer


class GPTSw3Tester(JaxModelTester):
    """Tester for GPT-SW3 model."""

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
        return FlaxGPT2LMHeadModel.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Sequence[jax.Array]:
        tokenizer = GPTSw3Tokenizer.from_pretrained(self._model_path)
        inputs = tokenizer(
            "Träd är fina för att", return_tensors="jax"
        )  # input is a swedish statement
        return inputs

    # @override
    def _get_static_argnames(self):
        return ["train"]
