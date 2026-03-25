# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import operator
from unittest.mock import patch

import pytest
import torch
from tt_torch.backend.passes import handle_composite_ops
from tt_torch.composite_ops import (
    composite_gelu,
    composite_topk,
    composite_topk_indices,
    composite_topk_values,
)

from tests.utils import capture_gm_via_compile, get_call_function_targets


class _TopKIndices(torch.nn.Module):
    def forward(self, x):
        return torch.topk(x, 3)[1]


class _TopKValues(torch.nn.Module):
    def forward(self, x):
        return torch.topk(x, 3)[0]


class _TopKBoth(torch.nn.Module):
    def forward(self, x):
        v, i = torch.topk(x, 3)
        return v, i


class _GeluModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")


@pytest.mark.push
def test_handle_composite_ops_gelu_no_multi_output():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_GeluModel(), x)
    with patch("tt_torch.backend.passes._replace_multi_output_op") as mock_replace:
        handle_composite_ops(gm)
        mock_replace.assert_not_called()
    targets = get_call_function_targets(gm)
    assert composite_gelu in targets
    assert torch.nn.functional.gelu not in targets
    assert operator.getitem not in targets


@pytest.mark.push
def test_handle_composite_ops_selects_indices():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKIndices(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    assert composite_topk_indices in targets
    assert composite_topk not in targets
    assert composite_topk_values not in targets
    # getitem node should have been erased
    assert operator.getitem not in targets


@pytest.mark.push
def test_handle_composite_ops_selects_values():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKValues(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    assert composite_topk_values in targets
    assert composite_topk not in targets
    assert composite_topk_indices not in targets
    assert operator.getitem not in targets


@pytest.mark.push
def test_handle_composite_ops_selects_both():
    x = torch.randn(1, 10)
    gm = capture_gm_via_compile(_TopKBoth(), x)
    handle_composite_ops(gm)
    targets = get_call_function_targets(gm)
    print(targets)
    assert composite_topk in targets
    assert composite_topk_values not in targets
    assert composite_topk_indices not in targets
    # getitem nodes remain since composite_topk still returns a tuple
    assert operator.getitem in targets
