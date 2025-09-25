# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence, Type

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from infra.testers.compiler_config import CompilerConfig


class MNISTCNNTester(TorchModelTester):
    """Tester for MNIST CNN model."""

    def __init__(
        self,
        model_class: Type[Model],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
    ) -> None:
        self._model_class = model_class
        super().__init__(comparison_config, run_mode, compiler_config)

    # @override
    def _get_model(self) -> Model:
        return self._model_class().to(dtype=torch.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # Channels is 1 as MNIST is in grayscale.
        return torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)  # B, C, H, W
