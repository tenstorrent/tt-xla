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
    torch_op_name="torch.histc",
)
@pytest.mark.parametrize(
    ["input_shape", "bins"],
    [
        ((128,), 32),
        ((512,), 32),
        ((256,), 16),
    ],
)
def test_histc_float(input_shape: tuple, bins: int):
    """Test torch.histc with float input. Currently passes via CPU fallback."""

    class Histc(torch.nn.Module):
        def __init__(self, bins):
            super().__init__()
            self.bins = bins

        def forward(self, x):
            return torch.histc(x, bins=self.bins, min=0, max=self.bins - 1)

    run_op_test_with_random_inputs(
        Histc(bins),
        [input_shape],
        minval=0.0,
        maxval=float(bins),
        dtype=torch.float32,
        framework=Framework.TORCH,
    )


