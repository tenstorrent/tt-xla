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
@pytest.mark.filecheck(["scaled_sum_to_mean.ttnn.mlir"])
def test_scaled_sum_to_mean(request):
    def scaled_sum_to_mean(x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=[2, 3]) * (1.0 / (x.shape[2] * x.shape[3]))

    run_graph_test_with_random_inputs(
        scaled_sum_to_mean,
        [(1, 16, 8, 8)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
