# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from gemma import gm
from transformers import AutoTokenizer, FlaxPreTrainedModel
from jaxtyping import PyTree


class GemmaTester(ModelTester):
    """Tester for Gemma models."""

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
        return self._model_path

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = gm.text.Gemma3Tokenizer()
        sampler = gm.text.Sampler(
            model=self._get_model, params=params, tokenizer=tokenizer
        )
        prompt = "Explain the concept of gravity in simple terms."
        output = sampler.sample(prompt, max_new_tokens=100)
        return output

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        self._model.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
        return {
            "params": self._model.params,
            **self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
