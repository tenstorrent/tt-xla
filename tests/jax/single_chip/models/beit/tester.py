# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import FlaxPreTrainedModel
from third_party.tt_forge_models.beit.image_classification.jax import ModelLoader


class FlaxBeitForImageClassificationTester(JaxModelTester):
    """Tester for Beit family of models on image classification tasks."""

    def __init__(
        self,
        model_variant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_variant = model_variant
        self._model_loader = ModelLoader(variant=model_variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict:
        return self._model_loader.load_inputs(batch_size=1)
