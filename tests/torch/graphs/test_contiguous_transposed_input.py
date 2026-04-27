# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test
from utils import incorrect_result


@pytest.mark.nightly
def test_contiguous_transposed_input():
    """
    Runs identity test with transposed input tensor (contiguous in memory).
    """

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x + 0

    tensor = torch.arange(32 * 2, dtype=torch.int16).reshape(32, 2).transpose(0, 1)
    run_graph_test(Identity(), [tensor], framework=Framework.TORCH)
