# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs
from utils import Category
import torch.nn as nn

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OP_TEST)
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

    run_op_test_with_random_inputs(
        Conv(),
        [(1, in_channels, 224, 224)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_conv_randn():
    """Test conv2d operation combined for random inputs"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

        def forward(self, x):
            return self.conv(x)

    model = Conv()
    model =model.to(torch.bfloat16)
    run_op_test_with_random_inputs(
        model,
        [(1,128,256,256)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_conv_real_inputs():
    """Test conv2d operation combined for random inputs"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            )

        def forward(self, x):
            return self.conv(x)

    model = Conv()
    model = model.to(torch.bfloat16)
    w=torch.load("weight_for_problematic_conv.pt")
    model.conv.weight = torch.nn.Parameter(w)
    run_op_test_with_saved_inputs(
        model,
        [torch.load("input_for_problematic_conv.pt")],
        framework=Framework.TORCH,
    )