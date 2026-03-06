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
@pytest.mark.filecheck(["gelu.ttnn.mlir"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(approximate, request):
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)

    run_graph_test_with_random_inputs(
        gelu,
        [(64, 256)],
        dtype=torch.float32,
        framework=Framework.TORCH,
        request=request,
    )
