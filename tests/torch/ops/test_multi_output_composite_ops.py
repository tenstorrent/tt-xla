# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import operator
from unittest.mock import patch

import torch
from tt_torch.backend.passes import handle_composite_ops
from tt_torch.composite_ops import (
    composite_gelu,
    composite_topk,
    composite_topk_indices,
    composite_topk_values,
)


def _capture_via_compile(model, *args):
    """
    Capture the FX GraphModule as seen by a torch.compile backend.

    torch.compile (dynamo) preserves high-level torch ops like torch.topk,
    which is required for handle_composite_ops to match on node.target.
    torch.export.export / make_fx would lower to aten.topk.default instead.
    """
    captured = {}

    def _backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    torch.compile(model, backend=_backend)(*args)
    return captured["gm"]


def _call_function_targets(gm):
    return {n.target for n in gm.graph.nodes if n.op == "call_function"}


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


def test_handle_composite_ops_gelu_no_multi_output():
    x = torch.randn(1, 10)
    gm = _capture_via_compile(_GeluModel(), x)
    with patch("tt_torch.backend.passes._replace_multi_output_op") as mock_replace:
        handle_composite_ops(gm)
        mock_replace.assert_not_called()
    targets = _call_function_targets(gm)
    assert composite_gelu in targets
    assert torch.nn.functional.gelu not in targets
    assert operator.getitem not in targets


def test_handle_composite_ops_selects_indices():
    x = torch.randn(1, 10)
    gm = _capture_via_compile(_TopKIndices(), x)
    handle_composite_ops(gm)
    targets = _call_function_targets(gm)
    assert composite_topk_indices in targets
    assert composite_topk not in targets
    assert composite_topk_values not in targets
    # getitem node should have been erased
    assert operator.getitem not in targets


def test_handle_composite_ops_selects_values():
    x = torch.randn(1, 10)
    gm = _capture_via_compile(_TopKValues(), x)
    handle_composite_ops(gm)
    targets = _call_function_targets(gm)
    assert composite_topk_values in targets
    assert composite_topk not in targets
    assert composite_topk_indices not in targets
    assert operator.getitem not in targets


def test_handle_composite_ops_selects_both():
    x = torch.randn(1, 10)
    gm = _capture_via_compile(_TopKBoth(), x)
    handle_composite_ops(gm)
    targets = _call_function_targets(gm)
    print(targets)
    assert composite_topk in targets
    assert composite_topk_values not in targets
    assert composite_topk_indices not in targets
    # getitem nodes remain since composite_topk still returns a tuple
    assert operator.getitem in targets
