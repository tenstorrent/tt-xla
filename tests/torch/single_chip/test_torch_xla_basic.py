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

import pytest

from infra.comparators.torch_comparator import TorchComparator

# TODO(@LPanosTT): https://github.com/tenstorrent/tt-xla/issues/1137
# We would like to use the OpTester/GraphTester infra instead of manually
# calculating and comparing golden vs device results.


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
        )
    )
    comparator.compare(output, golden)
