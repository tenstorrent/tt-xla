# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import CLIPProcessor, FlaxCLIPModel, FlaxPreTrainedModel


class FlaxCLIPTester(ModelTester):
    """Tester for CLIP family of models on image classification tasks."""

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        from_pt = self._model_name == "openai/clip-vit-large-patch14-336"

        return FlaxCLIPModel.from_pretrained(self._model_name, from_pt=from_pt)

    # @override
    def _get_input_activations(self) -> Dict:
        image = jax.random.uniform(jax.random.PRNGKey(42), (1, 3, 224, 224))
        preprocessor = CLIPProcessor.from_pretrained(self._model_name, do_rescale=False)
        inputs = preprocessor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="np",
        )
        return inputs

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            **self._get_input_activations(),
        }
