# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Advanced-index assignment with mixed int32/int64 indices. torch_xla
concatenates index tensors when lowering; mixed dtypes fail XLA's type check.
Hit by FlexAttention's create_block_mask (Krea Realtime builds it on device,
eagerly), hence both the eager and compiled paths are covered."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra.utilities.types import Framework

from tests.infra.testers.single_chip.op.op_tester import run_op_test


def _index_put(dense, rows, cols):
    dense = dense.clone()
    dense[rows, cols] = 1.0
    return dense


def _make_inputs(device=None):
    dense = torch.zeros(4, 6, device=device)
    rows = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(-1)
    cols = torch.arange(4, dtype=torch.int64, device=device).unsqueeze(-1)
    return dense, rows, cols


@pytest.mark.nightly
@pytest.mark.single_device
def test_index_put_mixed_dtype_compiled():
    run_op_test(_index_put, list(_make_inputs()), framework=Framework.TORCH)


@pytest.mark.nightly
@pytest.mark.single_device
def test_index_put_mixed_dtype_eager():
    """Eager lazy-tensor path — the one the Krea model takes."""
    xr.set_device_type("TT")
    device_out = _index_put(*_make_inputs(torch_xla.device()))
    torch_xla.sync()
    assert torch.equal(device_out.cpu(), _index_put(*_make_inputs()))
