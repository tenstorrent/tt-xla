# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
import torch
import torch_xla.core.xla_model as xm
from tt_torch.tools.utils import CompilerConfig

import pytest

from infra.comparators.torch_comparator import TorchComparator


@pytest.mark.parametrize("bias", [True, False])
def test_simple_mm(bias):
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 64, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)

    model = MM()
    golden = model(input_x)

    model = torch.compile(model, backend="tt")

    output = model(input_x)
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )

    comparator.compare(output, golden)


@pytest.mark.parametrize("bias", [True, False])
def test_simple_mm_eager(bias):
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)

    model = MM()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.parametrize("in_channels", [3, 64])
@pytest.mark.parametrize("out_channels", [3, 64])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                1,
                bias,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    input_x = torch.randn(1, in_channels, 224, 224, dtype=torch.bfloat16)

    model = Conv()
    golden = model(input_x)

    model = torch.compile(model, backend="tt")

    output = model(input_x)

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.parametrize("in_channels", [3, 64])
@pytest.mark.parametrize("out_channels", [3, 64])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_eager(
    in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                1,
                bias,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    input_x = torch.randn(1, in_channels, 224, 224, dtype=torch.bfloat16)

    model = Conv()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


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


@pytest.mark.parametrize("op", eltwise_unary_ops)
def test_eltwise_unary(op):
    input_x = (
        torch.randn(32, 32, dtype=torch.bfloat16)
        if op is not torch.bitwise_not
        else torch.randint(-100, 100, (32, 32))
    )

    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    model = Unary()
    golden = model(input_x)

    cc = CompilerConfig()
    cc.enable_consteval = True

    model = torch.compile(model, backend="tt", options=cc)

    output = model(input_x)

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.01),
            pcc=PccConfig(enabled=False, required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.parametrize("op", eltwise_unary_ops)
def test_eltwise_unary_eager(op):
    if op is torch.erf:
        pytest.skip(
            "erf not decomposed in eager execution. Becomes `stablehlo.custom_call(@mhlo.erf)` which we do not yet lower to ttir"
        )

    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    input_x = (
        torch.randn(32, 32, dtype=torch.bfloat16)
        if op is not torch.bitwise_not
        else torch.randint(-100, 100, (32, 32))
    )

    model = Unary()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.01),
            pcc=PccConfig(enabled=False, required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


eltwise_binary_ops = [
    torch.add,
    torch.atan2,
    torch.arctan2,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.div,
    torch.divide,
    torch.floor_divide,
    torch.fmod,
    torch.logaddexp,
    torch.logaddexp2,
    torch.mul,
    torch.multiply,
    torch.nextafter,
    torch.remainder,
    torch.sub,
    torch.subtract,
    torch.true_divide,
    torch.eq,
    torch.ne,
    torch.le,
    torch.ge,
    torch.greater,
    torch.greater_equal,
    torch.gt,
    torch.less_equal,
    torch.lt,
    torch.less,
    torch.maximum,
    torch.minimum,
    torch.fmax,
    torch.fmin,
    torch.not_equal,
]


@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary(op):
    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(f"{op} not supported in tt backend yet. Skipping test.")
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)

    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    golden = model(input_x, input_y)

    model = torch.compile(model, backend="tt")

    output = model(input_x, input_y)

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
            pcc=PccConfig(enabled=False, required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary_eager(op):
    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(f"{op} not supported in tt backend yet. Skipping test.")
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)

    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    golden = model(input_x, input_y)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)
    input_y = input_y.to(device)

    output = model(input_x, input_y).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
            pcc=PccConfig(enabled=False, required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)
