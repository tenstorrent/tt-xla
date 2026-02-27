# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs



@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.ops.aten.max_pool3d_with_indices",
)
@pytest.mark.parametrize(
    ["input_shape", "kernel_size", "stride", "padding", "dilation"],
    [
        # Exact config from DenseUNet3d model that triggers reduce_window failure
        [(1, 96, 16, 128, 128), [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1]],
    ],
    ids=[
        "dense_unet_3d_config",
    ],
)
def test_maxpool3d(input_shape, kernel_size, stride, padding, dilation):
    """Sanity test for 3D max pooling via aten.max_pool3d_with_indices.

    The dense_unet_3d_config case reproduces the 'failed to legalize
    stablehlo.reduce_window' error seen during SHLO -> TTIR conversion.
    """

    def maxpool3d(x: torch.Tensor) -> torch.Tensor:
        out, _ = torch.ops.aten.max_pool3d_with_indices.default(
            x, kernel_size, stride, padding, dilation, False
        )
        return out

    run_op_test_with_random_inputs(
        maxpool3d,
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.MaxPool3d",
)
@pytest.mark.parametrize(
    ["input_shape", "kernel_size", "stride", "padding"],
    [
        # Exact config from DenseUNet3d model
        [(1, 96, 16, 128, 128), (3, 3, 3), 2, (1, 1, 1)],
    ],
    ids=[
        "dense_unet_3d_config",
    ],
)
def test_nn_maxpool3d(input_shape, kernel_size, stride, padding):
    """Sanity test for nn.MaxPool3d module."""

    class MaxPool3dModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(
                kernel_size=kernel_size, stride=stride, padding=padding
            )

        def forward(self, x):
            return self.pool(x)

    run_op_test_with_random_inputs(
        MaxPool3dModule(),
        [input_shape],
        dtype=torch.float32,
        framework=Framework.TORCH,
    )
