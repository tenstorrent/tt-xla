# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["hardsigmoid.ttnn.mlir"])
def test_hardsigmoid(request):
    def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardsigmoid(x)

    run_graph_test_with_random_inputs(
        hardsigmoid,
        [(64, 256)],
        dtype=torch.float32,
        framework=Framework.TORCH,
        request=request,
    )
