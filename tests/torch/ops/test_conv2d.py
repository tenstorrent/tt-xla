# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
import torch.nn as nn
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_saved_inputs





@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_yoloworld_conv_module_randn():
    """Test conv2d,batchnorm2d,silu operations combined for random inputs"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
            self.bn = nn.BatchNorm2d(
                num_features=128,
                eps=0.001,
                momentum=0.03,
                affine=True,
                track_running_stats=True
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.act(x)
            return x

    model = Conv()
    model =model.to(torch.bfloat16)
    model.eval()
    run_op_test_with_random_inputs(
        model,
        [(1,128,40,40)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_yoloworld_conv_module_saved():
    """Test conv2d, batchnorm2d, silu operations combined for saved inputs & weights"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(
                num_features=128,
                eps=0.001,
                momentum=0.03,
                affine=True,
                track_running_stats=True,
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    model = Conv().to(torch.bfloat16)
    model.eval()

    w = torch.load("weight_conv.pt", map_location="cpu").to(torch.bfloat16)

    state = model.state_dict()
    state["conv.weight"] = w
    model.load_state_dict(state, strict=False)

    inp = torch.load("input_conv.pt").to(torch.bfloat16)

    run_op_test_with_saved_inputs(
        model,
        [inp],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_yoloworld_conv_module_randn_weights_real_input():
    """Test conv2d, batchnorm2d, silu with random weights and saved input"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(
                num_features=128,
                eps=0.001,
                momentum=0.03,
                affine=True,
                track_running_stats=True,
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    model = Conv().to(torch.bfloat16)
    model.eval()

    # Real (saved) input
    inp = torch.load("input_conv.pt", map_location="cpu").to(torch.bfloat16)

    run_op_test_with_saved_inputs(
        model,
        [inp],
        framework=Framework.TORCH,
    )


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.conv2d",
)
def test_yoloworld_conv_module_real_weights_randn_input():
    """Test conv2d, batchnorm2d, silu with saved weights and random input"""

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(
                num_features=128,
                eps=0.001,
                momentum=0.03,
                affine=True,
                track_running_stats=True,
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    model = Conv().to(torch.bfloat16)
    model.eval()

    # Load real (saved) weights
    w = torch.load("weight_conv.pt", map_location="cpu").to(torch.bfloat16)
    state = model.state_dict()
    state["conv.weight"] = w
    model.load_state_dict(state, strict=False)

    # Random input
    inp = torch.randn(1, 128, 40, 40, dtype=torch.bfloat16)

    run_op_test_with_saved_inputs(
        model,
        [inp],
        framework=Framework.TORCH,
    )


# @pytest.mark.push
# @pytest.mark.nightly
# @pytest.mark.single_device
# @pytest.mark.record_test_properties(category=Category.OP_TEST)
# @pytest.mark.parametrize("in_channels", [3, 64])
# @pytest.mark.parametrize("out_channels", [3, 64])
# @pytest.mark.parametrize("kernel_size", [2, 3])
# @pytest.mark.parametrize("stride", [1, 2])
# @pytest.mark.parametrize("padding", [0, 1])
# @pytest.mark.parametrize("dilation", [1, 2])
# @pytest.mark.parametrize("bias", [True, False])
# def test_conv2d(
#     in_channels, out_channels, kernel_size, stride, padding, dilation, bias
# ):
#     class Conv(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.conv = torch.nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 dilation,
#                 1,
#                 bias,
#                 dtype=torch.bfloat16,
#             )

#         def forward(self, x):
#             return self.conv(x)

#     run_op_test_with_random_inputs(
#         Conv(),
#         [(1, in_channels, 224, 224)],
#         dtype=torch.bfloat16,
#         framework=Framework.TORCH,
#     )
