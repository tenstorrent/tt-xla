# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Out-of-range slice bounds. Origin: DiffusionGemma's sliding-window KV-cache update
(transformers cache_utils) does ``x[:, :, -sliding_window + 1 :, :]`` -> start -1023
on a dim of size 19. CPU's ``__getitem__`` clamps any out-of-range bound; XLA's
strict ``aten.slice.Tensor`` rejects the negative ones ("Value out of range"), which
``clamp_neg_slice_bounds`` fixes by clamping them to ``-dim_size``.

Covers the four out-of-range corners (start/end x too-negative/too-positive):
non-empty results (neg start, pos end) compare via PCC; empty results
(neg end, pos start) assert shape. (refs: tt-xla #5199)
"""

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from infra.utilities.types import Framework
from utils import Category

from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs

SHAPE = (1, 8, 19, 256)  # [batch, kv_heads, seq, head_dim]; dim 2 (seq) = 19


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.ops.aten.slice.Tensor",
)
@pytest.mark.parametrize(
    ["start", "end"],
    [
        (-1023, None),  # neg_start: x[:, :, -1023:, :]  (DiffusionGemma cache slice)
        (None, 1023),  # pos_end:   x[:, :, :1023, :]
    ],
    ids=["neg_start", "pos_end"],
)
def test_slice_oob_full(start, end, request):
    """Out-of-range bound with a non-empty result -> compared against CPU via PCC."""

    def slice_op(x: torch.Tensor) -> torch.Tensor:
        return x[:, :, start:end, :]

    run_op_test_with_random_inputs(
        slice_op,
        [SHAPE],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.ops.aten.slice.Tensor",
)
@pytest.mark.parametrize(
    ["start", "end"],
    [
        (None, -1023),  # neg_end:   x[:, :, :-1023, :]
        (1023, None),  # pos_start: x[:, :, 1023:, :]
    ],
    ids=["neg_end", "pos_start"],
)
def test_slice_oob_empty(start, end):
    """Out-of-range bound with an empty result -> assert shape (PCC can't compare 0 elements)."""

    def slice_op(x):
        return x[:, :, start:end, :]

    expected = torch.empty(SHAPE)[
        :, :, start:end, :
    ].shape  # CPU clamp -> (1, 8, 0, 256)

    xr.set_device_type("TT")
    model = torch.compile(slice_op, backend="tt")
    x = torch.randn(SHAPE, dtype=torch.bfloat16).to(xm.xla_device())
    with torch.no_grad():
        out = model(x).to("cpu")

    assert out.shape == expected
