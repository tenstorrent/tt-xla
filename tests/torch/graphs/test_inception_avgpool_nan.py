# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity reproducer for Inception v4 PCC drop (pcc=0.014 in full model).

Root cause: AvgPool2d(kernel_size=3, stride=1, padding=1) in InceptionA.branch3
produces NaN/Inf on TT hardware.

This sanity test isolates only the ONE responsible op — AvgPool2d(k=3, s=1, p=1) —
with a random input tensor matching the exact shape it receives in the real model
([1, 384, 35, 35]). No model download required.

Same class of bug as DenseNet121 (test_densenet_avgpool_nan.py):
  AvgPool2d with kernel_size > 1 produces NaN/Inf on TT, regardless of
  input shape, stride, or padding.
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from infra.evaluators import ComparisonConfig
from tests.infra.evaluators.evaluation_config import PccConfig


class AvgPool2dSanity(nn.Module):
    """Single-op sanity: only the broken AvgPool2d from InceptionA.branch3."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.pool(x)  # [1, 384, 35, 35] → [1, 384, 35, 35] — NaN/Inf on TT


@pytest.mark.single_device
def test_inception_v4_inceptiona_branch3_avgpool_nan(request):
    """
    Sanity: AvgPool2d(k=3, s=1, p=1) alone on a random tensor must not produce NaN/Inf.

    Expected: TT output matches CPU (PCC >= 0.99) — no NaN or Inf.
    Actual:   TT produces Inf (and downstream NaN) — confirms bug is in AvgPool2d itself.

    Op location in model:
      timm inception_v4
      └── features[6]  InceptionA
           └── branch3[0]  AvgPool2d(kernel_size=3, stride=1, padding=1)

    This same AvgPool2d appears in every InceptionA (×4), InceptionB (×7),
    and InceptionC (×3) block — causing all downstream values to become inf/NaN.
    """
    torch.manual_seed(0)
    # Non-negative input (post-ReLU statistics from actual block input)
    inputs = [torch.abs(torch.randn(1, 384, 35, 35)) * 1.21]

    run_op_test(
        AvgPool2dSanity(),
        inputs,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
        framework=Framework.TORCH,
        request=request,
    )
