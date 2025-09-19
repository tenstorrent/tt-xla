# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs
import torch
import pytest
from infra import run_op_test, run_op_test_with_random_inputs, Framework
from utils import Category


def run_unary_ops(op):
    """
    Runs a unary operation with random inputs for torch.
    Args:
        op: The unary operation to run.
    """
    run_op_test_with_random_inputs(op, [(32, 32)], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_abs():
    class Abs(torch.nn.Module):
        def forward(self, x):
            return torch.abs(x)

    run_unary_ops(Abs())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_acos():
    class Acos(torch.nn.Module):
        def forward(self, x):
            return torch.acos(x)

    run_unary_ops(Acos())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_acosh():
    class Acosh(torch.nn.Module):
        def forward(self, x):
            return torch.acosh(x)

    run_unary_ops(Acosh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_angle():
    class Angle(torch.nn.Module):
        def forward(self, x):
            return torch.angle(x)

    run_unary_ops(Angle())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_asin():
    class Asin(torch.nn.Module):
        def forward(self, x):
            return torch.asin(x)

    run_unary_ops(Asin())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_asinh():
    class Asinh(torch.nn.Module):
        def forward(self, x):
            return torch.asinh(x)

    run_unary_ops(Asinh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_atan():
    class Atan(torch.nn.Module):
        def forward(self, x):
            return torch.atan(x)

    run_unary_ops(Atan())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_atanh():
    class Atanh(torch.nn.Module):
        def forward(self, x):
            return torch.atanh(x)

    run_unary_ops(Atanh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bitwise_not():
    class BitwiseNot(torch.nn.Module):
        def forward(self, x):
            return torch.abs(x)

    model = BitwiseNot()
    input_x = torch.randint(-100, 100, (32, 32))

    run_op_test(model, [input_x], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_ceil():
    class Ceil(torch.nn.Module):
        def forward(self, x):
            return torch.ceil(x)

    run_unary_ops(Ceil())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_clamp():
    class Clamp(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, -1, 1)

    run_unary_ops(Clamp())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_conj_physical():
    class ConjPhysical(torch.nn.Module):
        def forward(self, x):
            return torch.conj_physical(x)

    run_unary_ops(ConjPhysical())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_cos():
    class Cos(torch.nn.Module):
        def forward(self, x):
            return torch.cos(x)

    run_unary_ops(Cos())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_cosh():
    class Cosh(torch.nn.Module):
        def forward(self, x):
            return torch.cosh(x)

    run_unary_ops(Cosh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deg2rad():
    class Deg2rad(torch.nn.Module):
        def forward(self, x):
            return torch.deg2rad(x)

    run_unary_ops(Deg2rad())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_digamma():
    class Digamma(torch.nn.Module):
        def forward(self, x):
            return torch.digamma(x)

    run_unary_ops(Digamma())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_erf():
    class Erf(torch.nn.Module):
        def forward(self, x):
            return torch.erf(x)

    run_unary_ops(Erf())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_erfc():
    class Erfc(torch.nn.Module):
        def forward(self, x):
            return torch.erfc(x)

    run_unary_ops(Erfc())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_erfinv():
    class Erfinv(torch.nn.Module):
        def forward(self, x):
            return torch.erfinv(x)

    run_unary_ops(Erfinv())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_exp():
    class Exp(torch.nn.Module):
        def forward(self, x):
            return torch.exp(x)

    run_unary_ops(Exp())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_exp2():
    class Exp2(torch.nn.Module):
        def forward(self, x):
            return torch.exp2(x)

    run_unary_ops(Exp2())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_expm1():
    class Expm1(torch.nn.Module):
        def forward(self, x):
            return torch.expm1(x)

    run_unary_ops(Expm1())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_fix():
    class Fix(torch.nn.Module):
        def forward(self, x):
            return torch.fix(x)

    run_unary_ops(Fix())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_floor():
    class Floor(torch.nn.Module):
        def forward(self, x):
            return torch.floor(x)

    run_unary_ops(Floor())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_frac():
    class Frac(torch.nn.Module):
        def forward(self, x):
            return torch.frac(x)

    run_unary_ops(Frac())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_lgamma():
    class Lgamma(torch.nn.Module):
        def forward(self, x):
            return torch.lgamma(x)

    run_unary_ops(Lgamma())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_log():
    class Log(torch.nn.Module):
        def forward(self, x):
            return torch.log(x)

    run_unary_ops(Log())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_log10():
    class Log10(torch.nn.Module):
        def forward(self, x):
            return torch.log10(x)

    run_unary_ops(Log10())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_log1p():
    class Log1p(torch.nn.Module):
        def forward(self, x):
            return torch.log1p(x)

    run_unary_ops(Log1p())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_log2():
    class Log2(torch.nn.Module):
        def forward(self, x):
            return torch.log2(x)

    run_unary_ops(Log2())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_logit():
    class Logit(torch.nn.Module):
        def forward(self, x):
            return torch.logit(x)

    run_unary_ops(Logit())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_i0():
    class I0(torch.nn.Module):
        def forward(self, x):
            return torch.i0(x)

    run_unary_ops(I0())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_nan_to_num():
    class NanToNum(torch.nn.Module):
        def forward(self, x):
            return torch.nan_to_num(x)

    run_unary_ops(NanToNum())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_neg():
    class Neg(torch.nn.Module):
        def forward(self, x):
            return torch.neg(x)

    run_unary_ops(Neg())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_negative():
    class Negative(torch.nn.Module):
        def forward(self, x):
            return torch.negative(x)

    run_unary_ops(Negative())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_positive():
    class Positive(torch.nn.Module):
        def forward(self, x):
            return torch.positive(x)

    run_unary_ops(Positive())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_rad2deg():
    class Rad2deg(torch.nn.Module):
        def forward(self, x):
            return torch.rad2deg(x)

    run_unary_ops(Rad2deg())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_reciprocal():
    class Reciprocal(torch.nn.Module):
        def forward(self, x):
            return torch.reciprocal(x)

    run_unary_ops(Reciprocal())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="error: failed to legalize operation 'stablehlo.round_nearest_even' "
    "https://github.com/tenstorrent/tt-xla/issues/1441"
)
def test_round():
    class Round(torch.nn.Module):
        def forward(self, x):
            return torch.round(x)

    run_unary_ops(Round())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_rsqrt():
    class Rsqrt(torch.nn.Module):
        def forward(self, x):
            return torch.rsqrt(x)

    run_unary_ops(Rsqrt())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sigmoid():
    class Sigmoid(torch.nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    run_unary_ops(Sigmoid())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sign():
    class Sign(torch.nn.Module):
        def forward(self, x):
            return torch.sign(x)

    run_unary_ops(Sign())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sgn():
    class Sgn(torch.nn.Module):
        def forward(self, x):
            return torch.sgn(x)

    run_unary_ops(Sgn())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_signbit():
    class Signbit(torch.nn.Module):
        def forward(self, x):
            return torch.signbit(x)

    run_unary_ops(Signbit())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sin():
    class Sin(torch.nn.Module):
        def forward(self, x):
            return torch.sin(x)

    run_unary_ops(Sin())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sinh():
    class Sinh(torch.nn.Module):
        def forward(self, x):
            return torch.sinh(x)

    run_unary_ops(Sinh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sqrt():
    class Sqrt(torch.nn.Module):
        def forward(self, x):
            return torch.sqrt(x)

    run_unary_ops(Sqrt())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_square():
    class Square(torch.nn.Module):
        def forward(self, x):
            return torch.square(x)

    run_unary_ops(Square())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_tan():
    class Tan(torch.nn.Module):
        def forward(self, x):
            return torch.tan(x)

    model = Tan()

    run_op_test_with_random_inputs(model, [(32, 32)], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_tanh():
    class Tanh(torch.nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    run_unary_ops(Tanh())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_trunc():
    class Trunc(torch.nn.Module):
        def forward(self, x):
            return torch.trunc(x)

    run_unary_ops(Trunc())
