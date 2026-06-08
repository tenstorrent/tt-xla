# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python proof of the chisel softmax-golden bug — no device, no model, no MLIR.

The DeepSeek-V3.1 chisel report shows every attention `ttnn.softmax` (dimension=3) failing
numerics (PCC ~= 0). A standalone tt-mlir silicon repro proved the *device* softmax is correct
(PCC 0.9999), so the failure must be in chisel's golden.

The golden harness (third_party/tt-mlir/.../tools/golden/mapping.py):

    def chisel_ttnn_softmax(op, inputs):                       # ~line 8448
        return softmax_golden(input_tensor=inputs["input"],
                              dimension=unpack_mlir_attr(op.attributes["dimension"]))

    def softmax_golden(input_tensor, **kwargs):                # ~line 3053
        dimension = kwargs.get("dim", 1)                       # reads "dim", caller sent "dimension"
        return torch.nn.functional.softmax(input_tensor, dim=dimension)

The caller passes the axis as ``dimension=`` but the golden reads ``kwargs.get("dim", 1)`` — so the
op's real axis is dropped and the golden ALWAYS softmaxes over dim 1. This test exercises the golden
directly (pure torch) to prove that, in milliseconds.

Run (in the tt-xla docker container, with the venv that has the chisel/golden packages):
    pytest -svv chisel_analysis/repros/test_chisel_softmax_golden_bug.py
"""
import inspect

import pytest
import torch
from golden.mapping import chisel_ttnn_softmax, softmax_golden


def test_softmax_golden_ignores_dimension_kwarg():
    """softmax_golden(dimension=3) should reduce over axis 3 — but it uses dim 1 instead."""
    torch.manual_seed(0)
    x = torch.randn(
        2, 4, 8, 16
    )  # 4-D, like attention scores (dimension=3 is the last axis)

    got = softmax_golden(x, dimension=3)

    # BUG: the passed dimension is dropped; the golden softmaxes over dim 1 (its hardcoded default).
    assert torch.allclose(
        got, torch.softmax(x, dim=1)
    ), "expected the buggy dim-1 behavior"
    assert not torch.allclose(
        got, torch.softmax(x, dim=3)
    ), "if this fails, the bug is fixed — golden now honors dimension=3"


def test_softmax_golden_only_honors_the_dim_key():
    """Control: the golden DOES work, but only when the axis is passed under the key it reads."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 16)
    assert torch.allclose(softmax_golden(x, dim=3), torch.softmax(x, dim=3))


def test_caller_still_passes_the_wrong_key():
    """Guards against code drift: confirm chisel_ttnn_softmax really sends `dimension=`.

    If tt-mlir is fixed (caller switches to `dim=`, or the golden reads `dimension`), this
    assertion flips and tells us the report's softmax failures are no longer spurious.
    """
    src = inspect.getsource(chisel_ttnn_softmax)
    assert (
        "dimension=" in src
    ), "caller no longer passes dimension= — re-verify the golden"
    assert 'kwargs.get("dim"' in inspect.getsource(
        softmax_golden
    ), "softmax_golden no longer reads the 'dim' key — the bug may be fixed"
