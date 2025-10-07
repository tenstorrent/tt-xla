# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import jax
from infra import ComparisonConfig, JaxModelTester, Model, RunMode

from third_party.tt_forge_models.vit.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)


class ViTTester(JaxModelTester):
    """Tester for ViT family of models."""

    def __init__(
        self,
        variant_name: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        model = self._model_loader.load_model()
        return model

    # @override
    def _get_input_activations(self) -> jax.Array:
        return self._model_loader.load_inputs()
