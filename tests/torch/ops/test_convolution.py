# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.conv2d",
)
@pytest.mark.parametrize(
    [
        "batch_size",
        "output_channels",
        "input_channels",
        "input_height",
        "input_width",
        "filter_height",
        "filter_width",
    ],
    [
        # Representative shape from ResNet - single test case as requested
        (1, 64, 64, 56, 56, 3, 3),
    ],
    ids=lambda val: f"{val}",
)
@pytest.mark.parametrize("format", ["float32", "bfloat16", "bfp8"])
@pytest.mark.parametrize(
    "optimization_level", [0, 1], ids=["opt_level_0", "opt_level_1"]
)
def test_conv2d(
    request,
    batch_size: int,
    output_channels: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    filter_height: int,
    filter_width: int,
    format: str,
    optimization_level: int,
):
    device = xm.xla_device()
    device_kind = xm.xla_device_kind(device)
    device_str = str(device_kind)
    if device_str == "Blackhole" and format == "bfp8" and optimization_level > 0:
        pytest.xfail(f"conv2d BFP8 with optimizer not supported on {device_str}")

    def conv2d(img: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(img, weight, stride=1, padding=1)

    img_shape = (batch_size, input_channels, input_height, input_width)
    kernel_shape = (output_channels, input_channels, filter_height, filter_width)

    if format == "float32":
        dtype = torch.float32
        compiler_config = CompilerConfig(optimization_level=optimization_level)
    elif format == "bfloat16":
        dtype = torch.bfloat16
        compiler_config = CompilerConfig(optimization_level=optimization_level)
    elif format == "bfp8":
        dtype = torch.bfloat16
        compiler_config = CompilerConfig(
            optimization_level=optimization_level, enable_bfp8_conversion=True
        )

    run_op_test_with_random_inputs(
        conv2d,
        [img_shape, kernel_shape],
        dtype=dtype,
        compiler_config=compiler_config,
        framework=Framework.TORCH,
    )
