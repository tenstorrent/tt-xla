# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from einops import rearrange
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["concatenate_heads.ttnn.mlir"])
def test_concatenate_heads_3d(request):
    """Permute [0,2,1,3] + reshape to 3D: [batch, num_heads, seq, head_size] -> [batch, seq, hidden]."""

    def concatenate_heads_3d(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b h s d -> b s (h d)")

    run_graph_test_with_random_inputs(
        concatenate_heads_3d,
        [(1, 8, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["concatenate_heads.ttnn.mlir"])
def test_concatenate_heads_2d(request):
    """Permute [0,2,1,3] + reshape to 2D: [batch, num_heads, seq, head_size] -> [batch*seq, hidden]."""

    def concatenate_heads_2d(x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b h s d -> (b s) (h d)")

    run_graph_test_with_random_inputs(
        concatenate_heads_2d,
        [(1, 8, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
