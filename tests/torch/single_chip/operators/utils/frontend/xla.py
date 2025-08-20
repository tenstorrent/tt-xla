# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Forge frontend


from typing import Any, Dict, Sequence

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester


class SweepsTester(TorchModelTester):
    """Tester for Sweeps model."""

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self.model = model
        self.inputs = inputs
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self.model

    def _get_forward_method_args(self) -> Sequence[Any]:
        return self.inputs

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # return {
        #     "x": torch.randn(1, 3, 24, 22),
        #     "y": torch.randn(1, 3, 24, 22),
        # }
        inputs = {}
        # names = ["x", "y", "z", "a1", "a2", "a3", "a4", "a5", "a6"]
        # for index, input in enumerate(self.inputs):
        #     inputs[names[index]] = input
        return inputs


class XlaFrontend:
    @classmethod
    def verify(cls, model, inputs, compiler_cfg, verify_config):
        sweeps_tester = SweepsTester(model, inputs)
        sweeps_tester.test()
