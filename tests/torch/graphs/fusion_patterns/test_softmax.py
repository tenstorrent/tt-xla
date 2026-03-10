# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


# pulls double duty to test SoftmaxFusionPattern and NumericallyStableSoftmaxFusionPattern
@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["softmax.ttnn.mlir"])
def test_softmax(request):
    def softmax(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=1)

    run_graph_test_with_random_inputs(
        softmax,
        [(64, 256)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
