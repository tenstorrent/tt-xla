# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
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


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("math_fidelity", ["hifi2", "hifi4", "ttnn_default"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False])
def test_conv2d_mf_fp32_acc(math_fidelity, fp32_dest_acc_en):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                64,
                64,
                3,
                1,
                1,
                1,
                1,
                True,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    compiler_config = CompilerConfig(
        math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en
    )
    run_op_test_with_random_inputs(
        Conv(),
        [(1, 64, 224, 224)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
    )
