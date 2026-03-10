# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["batch_norm_decomposition.ttnn.mlir"])
def test_batch_norm_decomposition_conv2d(request):
    """BatchNorm following Conv2d is decomposed into alpha*x + beta (mul + add)."""

    def conv2d_then_batch_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_mean: torch.Tensor,
        bn_var: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.nn.functional.conv2d(x, weight, padding=1)
        return torch.nn.functional.batch_norm(
            x, bn_mean, bn_var, bn_weight, bn_bias, training=False
        )

    num_channels = 16
    run_graph_test_with_random_inputs(
        conv2d_then_batch_norm,
        [
            (1, 16, 8, 8),  # x: NCHW
            (num_channels, 16, 3, 3),  # conv weight: out_ch x in_ch x kH x kW
            (num_channels,),  # bn weight (scale)
            (num_channels,),  # bn bias (offset)
            (num_channels,),  # bn running mean
            (num_channels,),  # bn running var
        ],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )
