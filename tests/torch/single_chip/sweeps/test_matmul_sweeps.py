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
        inputs = {}
        return inputs

    
class ModelFromAnotherOp(torch.nn.Module):
    def __init__(self, operator):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_from_another_op"
        self.operator = operator

    def forward(self, x, y):
        xx = torch.add(x, x)
        yy = torch.add(y, y)
        return self.operator(xx, yy)
    

class ModelConstEvalPass(torch.nn.Module):
    def __init__(self, operator, shape_1, shape_2, device):
        super().__init__()
        self.testname = "Matmul_operator_test_op_src_const_eval_pass"
        self.operator = operator

        self.c1 = torch.rand(shape_1)  # .to(device) ?
        self.c2 = torch.rand(shape_2)  # .to(device) ?

        self.register_buffer("constant1", self.c1)
        self.register_buffer("constant2", self.c2)

    def forward(self, x, y):
        mm1 = self.operator(self.c1, self.c2)
        mm2 = self.operator(x, y)
        aa = torch.add(mm1, mm2)
        return aa
    

@pytest.fixture(scope="module")
def device():
    return xm.xla_device()


@pytest.mark.parametrize("shape_1", [(1, 4)])
def test_matmul_model_from_another_op(device, shape_1):

    shape_2 = shape_1[:-2] + (shape_1[-1], shape_1[-2])

    model = ModelFromAnotherOp(torch.matmul).eval()
    model.to(device)

    x = torch.rand(shape_1).to(device)
    y = torch.rand(shape_2).to(device)

    sweeps_tester = SweepsTester(model, [x, y])
    sweeps_tester.test()

    out = model(x, y)
    print("Final output:", out)


@pytest.mark.parametrize("shape_1", [(1, 4)])
def test_matmul_model_const_eval_pass(device, shape_1):

    shape_2 = shape_1[:-2] + (shape_1[-1], shape_1[-2])

    model = ModelConstEvalPass(torch.matmul, shape_1, shape_2, device).eval()
    model.to(device)

    x = torch.rand(shape_1).to(device)
    y = torch.rand(shape_2).to(device)

    sweeps_tester = SweepsTester(model, [x, y])
    sweeps_tester.test()

    out = model(x, y)
    print("Final output:", out)
