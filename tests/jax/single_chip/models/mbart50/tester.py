# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    AutoTokenizer,
    FlaxMBartForConditionalGeneration,
    FlaxPreTrainedModel,
)
from jaxtyping import PyTree


class MBartTester(ModelTester):
    """Tester for MBart model variants"""

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
        return FlaxMBartForConditionalGeneration.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello, my dog is cute.", return_tensors="jax")
        return inputs

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            **self._get_input_activations(),
        }

    # override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
