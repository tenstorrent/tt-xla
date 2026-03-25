# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal reproducer for AvgPool2d NaN on TT hardware.

Root cause investigation for DenseNet121 bug:
  - `transition1.pool` AvgPool2d(kernel_size=2, stride=2) in DenseNet121
    produces NaN/Inf on TT hardware.

Key finding (2026-03-25):
  ANY AvgPool2d with kernel_size > 1 produces NaN on TT hardware,
  regardless of input shape, dtype, or stride.
  MaxPool2d(k=2, s=2) and AvgPool2d(k=1, s=1) pass correctly.

  The bug is in TT-MLIR's lowering of AvgPool2d (stablehlo.reduce_window
  with 'add' reducer + divisor), not in the input data or shape.

Minimal repro:
  AvgPool2d(kernel_size=2, stride=1) on ANY shape → NaN
  AvgPool2d(kernel_size=2, stride=2) on ANY shape → NaN
  AvgPool2d(kernel_size=3, stride=2) on ANY shape → NaN
  AvgPool2d(kernel_size=1, stride=1) on ANY shape → PASSES (trivial)
  MaxPool2d(kernel_size=2, stride=2) on ANY shape → PASSES
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test_with_random_inputs
from infra.evaluators import ComparisonConfig
from tests.infra.evaluators.evaluation_config import PccConfig


_PCC_99 = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
_TORCH = Framework.TORCH
_SHAPE = (1, 4, 8, 8)  # smallest shape that reproduces the bug


class AvgPool2dWrapper(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)


# ---------------------------------------------------------------------------
# Minimal repro — confirms bug with tiny tensor (fast, < 5s)
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_avgpool2d_minimal_k2s2(request):
    """AvgPool2d(k=2, s=2) on shape [1,4,8,8] → NaN on TT. Minimal repro."""
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=2, stride=2),
        [_SHAPE],
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


@pytest.mark.single_device
def test_avgpool2d_minimal_k2s1(request):
    """AvgPool2d(k=2, s=1) on shape [1,4,8,8] → NaN on TT. stride doesn't matter."""
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=2, stride=1),
        [_SHAPE],
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


@pytest.mark.single_device
def test_avgpool2d_minimal_k3s2(request):
    """AvgPool2d(k=3, s=2) on shape [1,4,8,8] → NaN on TT. kernel_size=3 also fails."""
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=3, stride=2),
        [_SHAPE],
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


@pytest.mark.single_device
def test_avgpool2d_k1s1_passes(request):
    """AvgPool2d(k=1, s=1) — trivial case, should pass (no actual averaging)."""
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=1, stride=1),
        [_SHAPE],
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


@pytest.mark.single_device
def test_maxpool2d_k2s2_passes(request):
    """MaxPool2d(k=2, s=2) — same shape/stride as failing AvgPool2d, should pass."""

    class MaxPool2dWrapper(nn.Module):
        def forward(self, x):
            return nn.MaxPool2d(2, stride=2)(x)

    run_op_test_with_random_inputs(
        MaxPool2dWrapper(),
        [_SHAPE],
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


# ---------------------------------------------------------------------------
# DenseNet121 exact shape (context: original bug report)
# ---------------------------------------------------------------------------


@pytest.mark.single_device
def test_avgpool2d_densenet_transition1_shape(request):
    """
    Standalone AvgPool2d(k=2, s=2) with exact DenseNet121 transition1 shape.
    Shape: [1, 128, 56, 56] — transition1.conv output.
    """
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=2, stride=2),
        [(1, 128, 56, 56)],
        minval=-3.5,
        maxval=2.5,
        dtype="float32",
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )


# ---------------------------------------------------------------------------
# dtype parametric (both float32 and bfloat16 fail)
# ---------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_avgpool2d_dtype(dtype, request):
    """Bug reproduces in both float32 and bfloat16 — not dtype dependent."""
    run_op_test_with_random_inputs(
        AvgPool2dWrapper(kernel_size=2, stride=2),
        [_SHAPE],
        dtype=dtype,
        comparison_config=_PCC_99,
        framework=_TORCH,
        request=request,
    )
