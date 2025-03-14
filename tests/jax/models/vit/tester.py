# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import ComparisonConfig, ModelTester, RunMode

import jax.numpy as jnp
import jax.random as random
import jax

from typing import Dict, List, Sequence, Tuple, Union
from transformers import (
    FlaxPreTrainedModel,
    FlaxViTForImageClassification,
    ViTConfig,
    ViTImageProcessor,
)


class ViTTester(ModelTester):
    """Tester for ViT family of models."""

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
        return FlaxViTForImageClassification.from_pretrained(self._model_name)

    # @override
    def _get_input_activations(self) -> jax.Array:
        model_config = ViTConfig.from_pretrained(self._model_name)
        image_size = model_config.image_size
        key = random.PRNGKey(0)
        random_image = random.randint(
            key, (image_size, image_size, 3), 0, 256, dtype=jnp.uint8
        )

        processor = ViTImageProcessor.from_pretrained(self._model_name)
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
