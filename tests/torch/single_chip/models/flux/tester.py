# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


class FluxTester(TorchModelTester):
    """Tester for FLUX model on text-to-image generation tasks."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        # Convert string variant name to ModelVariant enum
        variant = ModelVariant(variant_name)
        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model(dtype_override=torch.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs(dtype_override=torch.bfloat16)
