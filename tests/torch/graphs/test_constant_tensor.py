# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_constant_tensor():
    """
    Test that triggers CONSTANT_TENSOR input kind in FX graph.
    """

    class ModelWithConstant(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            constant = torch.tensor(
                [[2.5, 3.5], [4.5, 5.5]], dtype=torch.bfloat16, device=x.device
            )
            return x + constant

    model = ModelWithConstant()
    run_graph_test_with_random_inputs(
        model, [(2, 2)], dtype=torch.bfloat16, framework=Framework.TORCH
    )
