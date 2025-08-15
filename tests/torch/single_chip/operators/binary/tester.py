# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import torch
# from tests.infra import ComparisonConfig, Model, RunMode, TorchModelTester
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

# from third_party.tt_forge_models.alexnet.pytorch.loader import ModelLoader


class ModelDirect(torch.nn.Module):

    model_name = "model_op_src_from_host"

    def __init__(self, kwargs):
        super(ModelDirect, self).__init__()
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        output = torch.add(x, y, **self.kwargs)
        return output


class BinaryAddTester(TorchModelTester):
    """Tester for Binary Add model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        # self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        # return self._model_loader.load_model()
        return ModelDirect(kwargs={})

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # return self._model_loader.load_inputs()
        return {
            "x": torch.randn(1, 3, 24, 22),
            "y": torch.randn(1, 3, 24, 22),
        }
