# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity reproducer for RT-DETR R18vd PCC drop (pcc=-0.040 in full model).

Root cause: AvgPool2d(kernel_size=2, stride=2, padding=0) in the ResNet backbone
shortcut path produces NaN/Inf on TT hardware.

This sanity test isolates only the ONE responsible op — AvgPool2d(k=2, s=2) —
with a random input tensor matching the exact shape it receives in the real model
([1, 64, 160, 160]). No model download required.

Same class of bug as DenseNet121 (test_densenet_avgpool_nan.py) and
Inception v4 (test_inception_avgpool_nan.py):
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
    """Single-op sanity: only the broken AvgPool2d from RTDetrResNetBasicLayer shortcut."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.pool(x)  # [1, 64, 160, 160] → [1, 64, 80, 80] — NaN/Inf on TT


@pytest.mark.single_device
def test_rt_detr_r18vd_backbone_shortcut_avgpool_nan(request):
    """
    Sanity: AvgPool2d(k=2, s=2) alone on a random tensor must not produce NaN/Inf.

    Expected: TT output matches CPU (PCC >= 0.99) — no NaN or Inf.
    Actual:   TT produces NaN and Inf — confirms bug is in AvgPool2d itself.

    Op location in model:
      PekingU/rtdetr_r18vd
      └── model.backbone.model.encoder.stages[1]
           └── layers[0].shortcut[0]  AvgPool2d(kernel_size=2, stride=2, padding=0)

    The same AvgPool2d(k=2, s=2) shortcut appears in stages[1], stages[2], and
    stages[3] — all three fail once the first one corrupts the activations.
    """
    torch.manual_seed(0)
    # Non-negative input (post-ReLU statistics from actual stage input)
    inputs = [torch.abs(torch.randn(1, 64, 160, 160)) * 0.44]

    run_op_test(
        AvgPool2dSanity(),
        inputs,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
        framework=Framework.TORCH,
        request=request,
    )
