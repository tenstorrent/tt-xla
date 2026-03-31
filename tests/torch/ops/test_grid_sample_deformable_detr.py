# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from infra.workloads import TorchWorkload
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.grid_sample",
)
def test_grid_sample_bilinear():
    """Test grid_sample with real Deformable DETR inputs saved from the model."""

    class GridSample(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor, grid):
            return torch.nn.functional.grid_sample(
                input_tensor,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

    model = GridSample().to(torch.bfloat16)

    # Load real inputs saved from Deformable DETR model
    value_l = torch.load("/home/tt-xla/s_check/grid_sample_value_l.pt").to(torch.bfloat16)
    grid_l = torch.load("/home/tt-xla/s_check/grid_sample_grid_l.pt").to(torch.bfloat16)

    print(f"value_l shape: {value_l.shape}")
    print(f"grid_l shape: {grid_l.shape}")

    run_op_test(
        model, [value_l, grid_l], framework=Framework.TORCH
    )