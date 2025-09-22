# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import math

from enum import Enum

from typing import Any, Dict, Sequence

# from ..utils.frontend.xla.provider import SweepsTester
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


# ---- Force TT device for this test session ----
xr.set_device_type("TT")


class SweepsTester(TorchModelTester):
    """Tester for Sweeps model."""

    def __init__(
        self,
        model: torch.nn.Module,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self.model = model
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self.model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        # Use the model's own _get_input_activations method if it exists
        if hasattr(self.model, "_get_input_activations"):
            return self.model._get_input_activations()
        # Fallback to empty inputs
        return []


class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator, shape_1, shape_2):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_another_op"
        self.operator = operator
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, yy)

    def _get_input_activations(self) -> Sequence[torch.Tensor]:
        x = torch.rand(*self.shape_1)
        y = torch.rand(*self.shape_2)
        return [x, y]


class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape_1, shape_2):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_const_eval_pass"
        self.operator = operator
        self.shape_1 = shape_1
        self.shape_2 = shape_2

        # Initialize constants on CPU first, then register as buffers so they move with the model
        c1 = torch.rand(*shape_1)
        c2 = torch.rand(*shape_2)

        self.register_buffer("constant1", c1)
        self.register_buffer("constant2", c2)

    def forward(self, x, y):
        mm1 = self.operator(self.constant1, self.constant2)
        mm2 = self.operator(x, y)
        aa = torch.add(mm1, mm2)
        return aa

    def _get_input_activations(self) -> Sequence[torch.Tensor]:
        x = torch.rand(*self.shape_1)
        y = torch.rand(*self.shape_2)
        return [x, y]


@pytest.fixture(scope="module")
def device():
    return xm.xla_device()


@pytest.mark.parametrize("shape_1", [(1, 4)])
def test_matmul_model_from_another_op(shape_1):

    shape_2 = shape_1[:-2] + (shape_1[-1], shape_1[-2])

    model = ModelFromAnotherOp(torch.matmul, shape_1, shape_2).eval()

    sweeps_tester = SweepsTester(model)
    sweeps_tester.test()


@pytest.mark.parametrize("shape_1", [(1, 4)])
def test_matmul_model_const_eval_pass(shape_1):

    shape_2 = shape_1[:-2] + (shape_1[-1], shape_1[-2])

    model = ModelConstEvalPass(torch.matmul, shape_1, shape_2).eval()

    sweeps_tester = SweepsTester(model)
    sweeps_tester.test()
