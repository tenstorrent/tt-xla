# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from infra import ComparisonConfig, ModelTester, RunMode, create_random_input_image
from transformers import (
    AutoImageProcessor,
    FlaxPreTrainedModel,
    FlaxRegNetForImageClassification,
)


class RegNetTester(ModelTester):
    """Tester for RegNet model"""

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
        return FlaxRegNetForImageClassification.from_pretrained(
            self._model_path, from_pt="True"
        )

    # @override
    def _get_input_activations(self) -> jax.Array:
        image_size = 224  # default image size for vision models
        random_image = create_random_input_image(image_size)

        processor = AutoImageProcessor.from_pretrained(self._model_path)
        inputs = processor(images=random_image, return_tensors="jax")
        return inputs["pixel_values"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "pixel_values": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
