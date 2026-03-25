# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity reproducer for DenseNet121 PCC drop.

Root cause: AvgPool2d(kernel_size=2, stride=2) in transition1 produces NaN/Inf
on TT hardware.

This sanity test isolates only the ONE responsible op — AvgPool2d(k=2, s=2) —
with a random input tensor matching the exact shape it receives in the real model
([1, 128, 56, 56]). No model download required.
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig
from tests.infra.evaluators.evaluation_config import PccConfig


class AvgPool2dSanity(nn.Module):
    """Single-op sanity: only the broken AvgPool2d, nothing else."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)  # [1, 128, 56, 56] → [1, 128, 28, 28] — NaN on TT


@pytest.mark.single_device
def test_densenet121_transition1_avgpool_nan(request):
    """
    Sanity: AvgPool2d(k=2, s=2) alone on a random tensor must not produce NaN.

    Expected: TT output matches CPU (PCC >= 0.99) — no NaN or Inf.
    Actual:   TT produces NaN/Inf — confirms bug is in AvgPool2d itself, not upstream.
    """
    inputs = [torch.randn(1, 128, 56, 56)]

    run_op_test(
        AvgPool2dSanity(),
        inputs,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
        framework=Framework.TORCH,
        request=request,
    )
