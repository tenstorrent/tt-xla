# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
import jax

from infra import ComparisonConfig, JaxModelTester, RunMode, Model
from third_party.tt_forge_models.resnet.image_classification.jax import (
    ModelLoader,
    ModelVariant,
)
from tests.infra.testers.compiler_config import CompilerConfig


class ResNetTester(JaxModelTester):
    """Tester for ResNet family of models."""

    def __init__(
        self,
        variant_name: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode, compiler_config)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()
