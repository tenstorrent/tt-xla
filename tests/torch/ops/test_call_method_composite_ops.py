# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import operator

import torch
from tt_torch.backend.passes import handle_composite_ops
from tt_torch.composite_ops import (
    composite_topk,
    composite_topk_indices,
    composite_topk_values,
)

from tests.utils import capture_gm_via_compile, get_call_function_targets

# This file tests that handle_composite_ops correctly handles call_method nodes.
# Specifically, when the composite ops are called as `x.topk(k, ...)` instead of `torch.topk(x, k, ...)`.


class _TopKBothMethod(torch.nn.Module):
    def forward(self, x):
        v, i = x.topk(3)
        return v, i


class _TopKValuesMethod(torch.nn.Module):
    def forward(self, x):
        return x.topk(3)[0]


class _TopKIndicesMethod(torch.nn.Module):
    def forward(self, x):
        return x.topk(3)[1]


def test_handle_composite_ops_method_selects_both():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKBothMethod(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    assert composite_topk in targets
    assert composite_topk_values not in targets
    assert composite_topk_indices not in targets
    assert operator.getitem in targets


def test_handle_composite_ops_method_selects_values():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKValuesMethod(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    assert composite_topk_values in targets
    assert composite_topk not in targets
    assert composite_topk_indices not in targets
    assert operator.getitem not in targets


def test_handle_composite_ops_method_selects_indices():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKIndicesMethod(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    assert composite_topk_indices in targets
    assert composite_topk not in targets
    assert composite_topk_values not in targets
    assert operator.getitem not in targets
