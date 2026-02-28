# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "in_channels,out_channels",
    [
        pytest.param(32, 32, id="both_div32"),
        pytest.param(12, 32, id="in_not_div32"),
        pytest.param(32, 12, id="out_not_div32"),
        pytest.param(12, 34, id="neither_div32"),
    ],
)
def test_conv3d(in_channels, out_channels):
    """
    Test torch.nn.Conv3d with channels divisible by 32 or not.

    TT-Metal conv3d requires channel alignment to 32. These cases test
    that padding/alignment is handled correctly when in_channels,
    out_channels, or both are not multiples of 32.
    """
    model = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    run_graph_test_with_random_inputs(
        model,
        [(1, in_channels, 8, 32, 32)],
        framework=Framework.TORCH,
    )
