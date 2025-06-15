#SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import AutoTokenizer, FlaxPreTrainedModel,FlaxAutoModel


class FlaxBertSentenceEmbeddingTester(ModelTester):
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
        a = FlaxAutoModel.from_pretrained(self._model_path,from_pt=True)
        print(a.config)
        exit()
        return FlaxAutoModel.from_pretrained(self._model_path,from_pt=True)


    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello [MASK]", return_tensors="jax")
        return inputs