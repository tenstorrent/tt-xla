# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import jax.random as random
from infra import ComparisonConfig, ModelTester, RunMode, create_random_input_image
from transformers import (
    AutoImageProcessor,
    Dinov2Config,
    FlaxPreTrainedModel,
    FlaxDinov2ForImageClassification,
)


class Dinov2Tester(ModelTester):
    """Tester for DINOv2 model"""

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
        return FlaxDinov2ForImageClassification.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> jax.Array:
        model_config = Dinov2Config.from_pretrained(self._model_path)
        image_size = model_config.image_size
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
