# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


def run_binary_ops(op):
    """
    Runs a binary operation with random float inputs for torch.
    """
    input_x = torch.randn(32, 32, dtype=torch.bfloat16)
    input_y = torch.randn(32, 32, dtype=torch.bfloat16)
    run_op_test(op, [input_x, input_y], framework=Framework.TORCH)


def run_bitwise_binary_ops(op):
    """
    Runs a bitwise binary operation with random integer inputs for torch.
    """
    input_x = torch.randint(-100, 100, (32, 32))
    input_y = torch.randint(-100, 100, (32, 32))
    run_op_test(op, [input_x, input_y], framework=Framework.TORCH)


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_add():
    class Add(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    run_binary_ops(Add())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_atan2():
    class Atan2(torch.nn.Module):
        def forward(self, x, y):
            return torch.atan2(x, y)

    run_binary_ops(Atan2())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_arctan2():
    class Arctan2(torch.nn.Module):
        def forward(self, x, y):
            return torch.arctan2(x, y)

    run_binary_ops(Arctan2())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bitwise_and():
    class BitwiseAnd(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_and(x, y)

    run_bitwise_binary_ops(BitwiseAnd())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bitwise_or():
    class BitwiseOr(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_or(x, y)

    run_bitwise_binary_ops(BitwiseOr())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bitwise_xor():
    class BitwiseXor(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_xor(x, y)

    run_bitwise_binary_ops(BitwiseXor())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.skip(
    reason="Bitwise left shift is not supported in tt backend yet. Skipping test."
)
def test_bitwise_left_shift():
    class BitwiseLeftShift(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_left_shift(x, y)

    run_binary_ops(BitwiseLeftShift())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.skip(
    reason="Bitwise right shift is not supported in tt backend yet. Skipping test."
)
def test_bitwise_right_shift():
    class BitwiseRightShift(torch.nn.Module):
        def forward(self, x, y):
            return torch.bitwise_right_shift(x, y)

    run_binary_ops(BitwiseRightShift())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_div():
    class Div(torch.nn.Module):
        def forward(self, x, y):
            return torch.div(x, y)

    run_binary_ops(Div())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_divide():
    class Divide(torch.nn.Module):
        def forward(self, x, y):
            return torch.divide(x, y)

    run_binary_ops(Divide())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_floor_divide():
    class FloorDivide(torch.nn.Module):
        def forward(self, x, y):
            return torch.floor_divide(x, y)

    run_binary_ops(FloorDivide())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_fmod():
    class Fmod(torch.nn.Module):
        def forward(self, x, y):
            return torch.fmod(x, y)

    run_binary_ops(Fmod())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_logaddexp():
    class Logaddexp(torch.nn.Module):
        def forward(self, x, y):
            return torch.logaddexp(x, y)

    run_binary_ops(Logaddexp())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_logaddexp2():
    class Logaddexp2(torch.nn.Module):
        def forward(self, x, y):
            return torch.logaddexp2(x, y)

    run_binary_ops(Logaddexp2())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_mul():
    class Mul(torch.nn.Module):
        def forward(self, x, y):
            return torch.mul(x, y)

    run_binary_ops(Mul())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_multiply():
    class Multiply(torch.nn.Module):
        def forward(self, x, y):
            return torch.multiply(x, y)

    run_binary_ops(Multiply())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_nextafter():
    class Nextafter(torch.nn.Module):
        def forward(self, x, y):
            return torch.nextafter(x, y)

    run_binary_ops(Nextafter())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_remainder():
    class Remainder(torch.nn.Module):
        def forward(self, x, y):
            return torch.remainder(x, y)

    run_binary_ops(Remainder())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_sub():
    class Sub(torch.nn.Module):
        def forward(self, x, y):
            return torch.sub(x, y)

    run_binary_ops(Sub())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_subtract():
    class Subtract(torch.nn.Module):
        def forward(self, x, y):
            return torch.subtract(x, y)

    run_binary_ops(Subtract())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.xfail(
    reason="PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. https://github.com/tenstorrent/tt-xla/issues/379"
)
def test_true_divide():
    class TrueDivide(torch.nn.Module):
        def forward(self, x, y):
            return torch.true_divide(x, y)

    run_binary_ops(TrueDivide())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_equal():
    class Equal(torch.nn.Module):
        def forward(self, x, y):
            return torch.eq(x, y)

    run_binary_ops(Equal())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_not_equal():
    class NotEqual(torch.nn.Module):
        def forward(self, x, y):
            return torch.ne(x, y)

    run_binary_ops(NotEqual())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_less_than_or_equal():
    class LessThanOrEqual(torch.nn.Module):
        def forward(self, x, y):
            return torch.le(x, y)

    run_binary_ops(LessThanOrEqual())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_greater_than_or_equal():
    class GreaterThanOrEqual(torch.nn.Module):
        def forward(self, x, y):
            return torch.ge(x, y)

    run_binary_ops(GreaterThanOrEqual())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_greater_than():
    class GreaterThan(torch.nn.Module):
        def forward(self, x, y):
            return torch.gt(x, y)

    run_binary_ops(GreaterThan())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_less_equal():
    class LessEqual(torch.nn.Module):
        def forward(self, x, y):
            return torch.less_equal(x, y)

    run_binary_ops(LessEqual())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_less():
    class Less(torch.nn.Module):
        def forward(self, x, y):
            return torch.less(x, y)

    run_binary_ops(Less())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_maximum():
    class Maximum(torch.nn.Module):
        def forward(self, x, y):
            return torch.maximum(x, y)

    run_binary_ops(Maximum())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_minimum():
    class Minimum(torch.nn.Module):
        def forward(self, x, y):
            return torch.minimum(x, y)

    run_binary_ops(Minimum())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_fmax():
    class Fmax(torch.nn.Module):
        def forward(self, x, y):
            return torch.fmax(x, y)

    run_binary_ops(Fmax())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_fmin():
    class Fmin(torch.nn.Module):
        def forward(self, x, y):
            return torch.fmin(x, y)

    run_binary_ops(Fmin())


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_not_equal():
    class NotEqual(torch.nn.Module):
        def forward(self, x, y):
            return torch.not_equal(x, y)

    run_binary_ops(NotEqual())
