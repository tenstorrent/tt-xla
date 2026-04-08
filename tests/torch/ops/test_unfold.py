# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Functional-equivalence tests for the ``torch.nn.functional.unfold`` override.

The override in ``tt_torch.torch_overrides`` replaces ``F.unfold`` with a
gather-based im2col (``_unfold_via_gather``) because the native op lowers to
HLO ops unsupported on the TT backend. These tests close the testing gap called
out in review by asserting the rewrite is element-for-element identical to
``torch.nn.functional.unfold`` on CPU across a wide sweep of kernel / stride /
padding / dilation combinations (including non-square parameters).
"""

import itertools

import pytest
import torch
from tt_torch.torch_overrides import _unfold_via_gather

KERNELS = [1, 2, 3, (2, 3)]
STRIDES = [1, 2, (1, 2)]
PADDINGS = [0, 1, (1, 0)]
DILATIONS = [1, 2, (1, 2)]


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation",
    list(itertools.product(KERNELS, STRIDES, PADDINGS, DILATIONS)),
)
@pytest.mark.parametrize("shape", [(2, 3, 16, 14)])
def test_unfold_matches_native(shape, kernel_size, stride, padding, dilation):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32)

    expected = torch.nn.functional.unfold(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    got = _unfold_via_gather(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.push
@pytest.mark.single_device
def test_unfold_defaults_match_native():
    """The override is also exercised through the positional/keyword default
    path used by callers that only pass ``kernel_size``."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 8, 8, dtype=torch.float32)

    expected = torch.nn.functional.unfold(x, kernel_size=2)
    got = _unfold_via_gather(x, kernel_size=2)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)
