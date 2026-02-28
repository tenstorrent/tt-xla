# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence, Any, Mapping

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import AutoTokenizer, FlaxPreTrainedModel
from deepseekr1.utils.utils import load_model


class DeepseekTester(ModelTester):
    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        self._model = None
        self._params = None
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        self._model, self._params = load_model(SHARD_MODEL=True)
        return self._model

    # @override
    def _get_input_parameters(self) -> Dict[str, jax.Array]:
        return self._params

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello there fellow traveler", return_tensors="jax")
        return {k: v for k, v in inputs.data.items()}

    # @override
    def _get_forward_method_name(self) -> str:
        return "apply"

    # @override
    def _get_forward_method_args(self) -> Sequence[Any]:
        return [self._input_parameters]

    # @override
    def _get_forward_method_kwargs(self) -> Mapping[str, Any]:
        return self._input_activations
