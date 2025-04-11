# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import XGLMTokenizer, FlaxPreTrainedModel, FlaxXGLMForCausalLM
from jaxtyping import PyTree


class XGLMTester(ModelTester):
    """Tester for XGLM models for language Modeling task."""

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
        model = FlaxXGLMForCausalLM.from_pretrained(self._model_path)
        model.params = model.to_bf16(model.params)
        return model

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = XGLMTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello, my dog is cute.", return_tensors="jax")
        return inputs

    def _get_forward_method_kwargs(self) -> Dict[str, PyTree]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            **self._get_input_activations(),
        }
