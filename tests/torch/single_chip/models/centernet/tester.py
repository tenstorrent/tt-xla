# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Any, Dict, Sequence
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from third_party.tt_forge_models.centernet.pytorch import ModelLoader, ModelVariant


class CenterNetWrapper(torch.nn.Module):
    """
    Wraps CenterNet to return only the final stack's outputs.

    CenterNet (Hourglass backbone) produces one prediction per stack.
    This wrapper selects only the last stack's output (most refined),
    matching the official implementation:
    https://github.com/xingyizhou/CenterNet/blob/4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c/src/lib/detectors/ctdet.py#L30
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[-1]


class CenterNetTester(TorchModelTester):
    """Tester for CenterNet model."""

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
        return CenterNetWrapper(self._model_loader.load_model())

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
