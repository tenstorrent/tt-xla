# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs
import torch
import pytest
from infra import run_op_test, run_op_test_with_random_inputs, Framework
from utils import Category

eltwise_unary_ops = [
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.angle,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.bitwise_not,
    torch.ceil,
    lambda act: torch.clamp(act, -1, 1),  # needs min and max
    torch.conj_physical,
    torch.cos,
    torch.cosh,
    torch.deg2rad,
    torch.digamma,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.fix,
    torch.floor,
    torch.frac,
    torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logit,
    torch.i0,
    torch.isnan,
    torch.nan_to_num,
    torch.neg,
    torch.negative,
    torch.positive,
    torch.rad2deg,
    torch.reciprocal,
    # torch.round, error: failed to legalize operation 'stablehlo.round_nearest_even'
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sgn,
    torch.signbit,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
]

# @pytest.mark.push
# @pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("op", eltwise_unary_ops)
def test_eltwise_unary_ops(op):
    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    model = Unary()

    if op is torch.bitwise_not:
        input_x = torch.randint(-100, 100, (32, 32))
        run_op_test(model, [input_x], framework=Framework.TORCH)
    else:
        run_op_test_with_random_inputs(model, [(32, 32)], framework=Framework.TORCH)
