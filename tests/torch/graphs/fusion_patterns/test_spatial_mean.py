# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["spatial_mean.ttnn.mlir"])
def test_spatial_mean_keep_dim(request):
    """Mean over spatial dims [1, 2] of a 4D tensor with keep_dim=True."""

    def spatial_mean_keep_dim(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=[1, 2], keepdim=True)

    run_graph_test_with_random_inputs(
        spatial_mean_keep_dim,
        [(1, 8, 8, 16)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["spatial_mean.ttnn.mlir"])
def test_spatial_mean_no_keep_dim(request):
    """Mean over spatial dims [1, 2] of a 4D tensor with keep_dim=False."""

    def spatial_mean_no_keep_dim(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=[1, 2], keepdim=False)

    run_graph_test_with_random_inputs(
        spatial_mean_no_keep_dim,
        [(1, 8, 8, 16)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
