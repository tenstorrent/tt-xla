# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.nn.functional.grid_sample",
)
def test_grid_sample_bilinear():
    """Test grid_sample operation with bilinear interpolation."""

    class GridSample(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor, grid):
            
            # https://github.com/huggingface/transformers/blob/51f94ea06d19a6308c61bbb4dc97c40aabd12bad/src/transformers/models/deformable_detr/modeling_deformable_detr.py#L87C33-L87C59
            return torch.nn.functional.grid_sample(
                input_tensor,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

    model = GridSample().to(torch.bfloat16)

    input_shape = (8, 32, 100, 134)
    grid_shape = (8, 17821, 4, 2)

    run_op_test_with_random_inputs(
        model, [input_shape, grid_shape], dtype=torch.bfloat16, framework=Framework.TORCH
    )