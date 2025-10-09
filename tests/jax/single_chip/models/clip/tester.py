# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Sequence

import jax
from infra import ComparisonConfig, JaxModelTester, Model, RunMode

from third_party.tt_forge_models.clip.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)


class FlaxCLIPTester(JaxModelTester):
    """Tester for CLIP family of models on image classification tasks."""

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
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict:
        return self._model_loader.load_inputs()

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        return ["train"]

    # @override
    def _wrapper_model(self, f):
        def model(args, kwargs):
            # TODO: Check if we need to support both image and text model output
            out = f(*args, **kwargs).text_model_output.pooler_output
            return out

        return model
