# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.core.xla_model as xm
from utils import Category


class SimpleLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.linear(x).relu()


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_compile_backward_faults_without_aot_autograd():
    torch.manual_seed(0)
    device = xm.xla_device()
    model = SimpleLinear().to(device)
    model.compile(backend="tt")

    x = torch.randn(4, 8, device=device, requires_grad=True)
    output = model(x).sum()

    output.backward()
