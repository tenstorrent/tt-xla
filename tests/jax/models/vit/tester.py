# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Sequence, Tuple, Union

import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import (
    FlaxPreTrainedModel,
    FlaxViTForImageClassification,
    ViTConfig,
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
        return jax.random.uniform(jax.random.PRNGKey(0), (1, 3, image_size, image_size))

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
