# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Type

import torch
import torch.nn as nn
from infra import ComparisonConfig, ModelTester, RunMode
from utilities.types import Model


class MNISTCNNTester(ModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        model_class: Type[Model],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_class = model_class
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> nn.Module:
        return self._model_class().to(dtype=torch.bfloat16)

    # @override
    def _get_forward_method_name(self) -> str:
        return "forward"

    # @override
    def _get_input_activations(self) -> torch.Tensor:
        # Channels is 1 as MNIST is in grayscale.
        return torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)  # B, C, H, W

    # @override
    def _get_forward_method_args(self) -> Sequence[torch.Tensor]:
        return [self._get_input_activations()]
