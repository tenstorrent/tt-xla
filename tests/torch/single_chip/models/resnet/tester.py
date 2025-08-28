# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from third_party.tt_forge_models.resnet.pytorch import ModelLoader
from tt_torch.tools.utils import CompilerConfig
from transformers import (
    AutoConfig,
    AutoModel,
)
import torch


class ResnetTester(TorchModelTester):
    """Tester for resnet model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        cc = CompilerConfig()
        cc.enable_consteval = False
        cc.consteval_parameters = False
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode, cc)

    # @override
    def _get_model(self) -> Model:
        config = AutoConfig.from_pretrained("xai-org/grok-2")
        config.num_hidden_layers = 3
        config.vision_config.num_hidden_layers = 0
        model = AutoModel.from_config(config)

        return model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        batch_size = 1
        max_length = 128
        return torch.randint(0, 65535, (batch_size, max_length), dtype=torch.int32)
