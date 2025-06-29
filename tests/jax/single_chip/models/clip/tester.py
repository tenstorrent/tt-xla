# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import CLIPProcessor, FlaxCLIPModel, FlaxPreTrainedModel


class FlaxCLIPTester(JaxModelTester):
    """Tester for CLIP family of models on image classification tasks."""

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
        from_pt = self._model_path == "openai/clip-vit-large-patch14-336"

        return FlaxCLIPModel.from_pretrained(self._model_path, from_pt=from_pt)

    # @override
    def _get_input_activations(self) -> Dict:
        image = jax.random.uniform(jax.random.PRNGKey(42), (1, 3, 224, 224))
        preprocessor = CLIPProcessor.from_pretrained(self._model_path, do_rescale=False)
        inputs = preprocessor(
            text=["a photo of a cat", "a photo of a dog"],
            images=image,
            return_tensors="jax",
        )
        return inputs
