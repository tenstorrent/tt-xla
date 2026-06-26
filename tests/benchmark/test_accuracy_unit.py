# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the accuracy helpers (PCC / rel-L2 / assert_pcc)."""

import pytest
import torch

from accuracy import assert_pcc, compute_pcc, compute_rel_l2


def test_compute_pcc_identical_is_one():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert compute_pcc(t, t.clone()) == pytest.approx(1.0)


def test_compute_rel_l2_identical_is_zero():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert compute_rel_l2(t, t.clone()) == pytest.approx(0.0)


def test_assert_pcc_passes_on_identical():
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert assert_pcc(t, t.clone(), 0.99) == pytest.approx(1.0)


def test_assert_pcc_raises_below_threshold():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.randn(3, 4)
    with pytest.raises(AssertionError, match="PCC comparison failed"):
        assert_pcc(a, b, 0.999999)
