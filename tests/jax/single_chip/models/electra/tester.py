# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import AutoTokenizer, FlaxElectraForCausalLM, FlaxPreTrainedModel


class ElectraTester(JaxModelTester):
    """Tester for Electra models for Causal LM task"""

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
        return FlaxElectraForCausalLM.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        return tokenizer(
            "Hello, my dog is cute",
            return_tensors="jax",
        )

    # @override
    def _get_static_argnames(self):
        return ["train"]
