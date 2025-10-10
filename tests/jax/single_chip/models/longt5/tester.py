# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Sequence

import jax
from infra import ComparisonConfig, JaxModelTester, Model, RunMode

from third_party.tt_forge_models.longt5.text_classification.jax import (
    ModelLoader,
    ModelVariant,
)


class LongT5Tester(JaxModelTester):
    """Tester for Long T5 model variants."""

    def __init__(
        self,
        variant: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        return self._model_loader.load_inputs()

    # @override
    def _get_static_argnames(self) -> Optional[Sequence[str]]:
        return ["train"]
