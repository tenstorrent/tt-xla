# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.topk",
)
@pytest.mark.parametrize(
    ["input_shape", "k"],
    [
        pytest.param(
            (1, 10),
            5,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed. Calculated: pcc=0.0042618513107299805. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1714",
            ),
        ),
        pytest.param(
            (1, 20),
            5,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed. Calculated: pcc=-0.3680523633956909. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1714",
            ),
        ),
        pytest.param(
            (1, 30),
            5,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed. Calculated: pcc=0.45831882953643. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1714",
            ),
        ),
        pytest.param(
            (1, 40),
            5,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed. Calculated: pcc=0.22071652114391327. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1714",
            ),
        ),
        pytest.param(
            (1, 8400),
            300,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed. Calculated: pcc=0.054578136652708054. Required: pcc=0.99. - https://github.com/tenstorrent/tt-xla/issues/1714",
            ),
        ),
    ],
)
def test_topk_indices(input_shape: tuple, k: int):
    """Test topk operation returning indices."""

    class TopKIndices(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[1]

    model = TopKIndices(k)

    run_op_test_with_random_inputs(
        model, [input_shape], dtype=torch.float32, framework=Framework.TORCH
    )
