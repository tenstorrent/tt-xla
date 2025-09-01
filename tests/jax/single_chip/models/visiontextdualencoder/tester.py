# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

import jax
from flax import linen as nn
from infra import ComparisonConfig, JaxModelTester, RunMode
from third_party.tt_forge_models.vision_text_dual_encoder.mm_image_ttt.jax import ModelLoader, ModelVariant


class VisionTextDualEncoderTester(JaxModelTester):
    """Tester for Vision-Text Dual Encoder (VTDE) model."""

    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BASE,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
