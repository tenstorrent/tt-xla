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
def test_grid_sample():

    class GridSample(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_tensor, grid):
            return torch.nn.functional.grid_sample(
                input_tensor,
                grid,
            )

    model = GridSample()

    input_shape = (1, 256, 28, 28)
    grid_shape = (1, 7, 25281, 2)

    run_op_test_with_random_inputs(
        model, [input_shape, grid_shape], framework=Framework.TORCH
    )