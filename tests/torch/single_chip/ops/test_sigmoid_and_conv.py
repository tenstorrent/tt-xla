# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test_with_random_inputs
from infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sigmoid_and_conv2d",
)
def test_sigmoid_and_conv_randn():
    """Test sigmoid and conv2d operations combined for random inputs"""

    class SigmoidConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )

        def forward(self, x1, x2):
            y1 = torch.sigmoid(x2)
            y2 = self.conv(x1)
            return (y1, y2)

    model = SigmoidConv()

    run_op_test_with_random_inputs(
        model,
        [(1, 64, 64, 64), (1, 1, 64, 64)],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sigmoid_and_conv2d_saved",
)
def test_sigmoid_and_conv_saved_inputs():
    """Test sigmoid and conv2d operations combined for saved inputs"""

    class SigmoidConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )

        def forward(self, x1, x2):
            y1 = torch.sigmoid(x2)
            y2 = self.conv(x1)
            return (y1, y2)

    model = SigmoidConv()

    run_op_test_with_saved_inputs(
        model,
        [torch.load("input_1.pt"), torch.load("input_2.pt")],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.sigmoid",
)
def test_sigmoid_saved_inputs():
    """Test sigmoid operation with saved inputs."""

    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    model = Sigmoid()

    run_op_test_with_saved_inputs(
        model, [torch.load("input_2.pt")], framework=Framework.TORCH
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_conv2d_saved_inputs():
    """Test conv2d operation with saved inputs."""

    class Conv2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            )

        def forward(self, x):
            return self.conv(x)

    model = Conv2d()

    run_op_test_with_saved_inputs(
        model, [torch.load("input_1.pt")], framework=Framework.TORCH
    )
