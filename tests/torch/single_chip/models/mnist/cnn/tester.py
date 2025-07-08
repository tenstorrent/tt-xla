# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Type

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester


class MNISTCNNTester(TorchModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        model_class: Type[Model],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_class = model_class
        super().__init__(comparison_config, run_mode)
        
        
    def get_model_inputs_and_parameters(self):
        model = self._model_class()
        # Example input: batch size 1, 1 channel, 28x28 image
        parameters = {name: param for name, param in model.named_parameters()}
        print(f"Model parameters:", parameters)

    # @override
    def _get_model(self) -> Model:
        return self._model_class().to(dtype=torch.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # Channels is 1 as MNIST is in grayscale.
        return torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)  # B, C, H, W
