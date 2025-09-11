# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Type

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from .model_implementation import MNISTMLPModel


class MNISTMLPTester(TorchModelTester):
    """Tester for MNIST MLP model."""

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        skip_compilation: bool = False,
    ) -> None:
        super().__init__(comparison_config, run_mode, skip_compilation=skip_compilation)

    # @override
    def _get_model(self) -> Model:
        return MNISTMLPModel().to(dtype=torch.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return torch.randn((4, 28 * 28), dtype=torch.bfloat16)  # B, C, H, W
