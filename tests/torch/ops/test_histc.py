# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Single-chip torch.histc op tests (histc is not sharded in DiffusionGemma)."""

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester

torch.manual_seed(0)


class Histc(torch.nn.Module):
    def __init__(self, bins, min, max):
        super().__init__()
        self.bins, self.min, self.max = bins, min, max

    def forward(self, x):
        return torch.histc(x, bins=self.bins, min=self.min, max=self.max)


# (input, bins, min, max)
CASES = [
    (torch.randint(0, 128, (152,)).float(), 128, 0, 127),  # diffusion gemma case
    (torch.randint(0, 128, (2048,)).float(), 128, 0, 127),  # diffusion gemma case
    (torch.tensor([1.0, 2.0, 1.0]), 4, 0, 3),
    (torch.linspace(0.0, 1.0, 50), 10, 0, 1),
    (torch.tensor([-5.0, -0.1, 0.0, 1.5, 3.0, 4.0, 9.0]), 3, 0, 3),
    (torch.tensor([0.0, 1.5, 3.0, 3.0]), 3, 0, 3),
    (torch.tensor([0.0, float("nan"), 1.5, 3.0, 2.0]), 3, 0, 3),
    (torch.linspace(-3.0, 3.0, 25), 6, -3, 3),
    (torch.tensor([2.0]), 4, 0, 3),
    (torch.rand(500) * 10.0, 256, 0, 10),
    (torch.tensor([1.0, 5.0, 3.0, 9.0, 2.0, 8.0]), 5, 0, 0),
    (torch.tensor([1.0, 5.0, 3.0, 9.0, 2.0, 8.0]), 5, 5, 5),
    (torch.tensor([3.0, 3.0, 3.0]), 4, 0, 0),
]


@pytest.mark.parametrize("x, bins, min, max", CASES)
def test_histc(x, bins, min, max):
    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    tester.test(
        Workload(framework=Framework.TORCH, model=Histc(bins, min, max), args=[x])
    )
