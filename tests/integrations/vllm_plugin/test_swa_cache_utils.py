# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sliding-window ring helpers.

These are pure functions (no device, no vLLM engine), so they run anywhere and
pin down the invariants the worker's byte reservation and the model runner's
allocation both depend on.
"""

import numpy as np
import pytest
from vllm.utils.math_utils import cdiv
from vllm_tt.swa_cache_utils import (
    PAGE_TABLE_STICK_BLOCK_ALIGNMENT,
    assign_ring_slots,
    build_sliding_ring_page_table,
    sliding_page_table_width,
    sliding_ring_phys_blocks,
    sliding_ring_reserve_bytes,
    sliding_window_blocks,
)

pytestmark = [pytest.mark.push, pytest.mark.cpu]


@pytest.mark.parametrize(
    "window,block_size",
    [(1024, 32), (4096, 32), (512, 64), (1, 32), (8192, 128), (1023, 32)],
)
def test_sliding_window_blocks_alignment_and_slack(window, block_size):
    wb = sliding_window_blocks(window, block_size)
    # stick alignment: window_blocks int32 ids must be a 32-byte multiple
    assert wb % PAGE_TABLE_STICK_BLOCK_ALIGNMENT == 0
    assert wb * 4 % 32 == 0
    # covers the window plus one slack block for a straddling request...
    assert wb >= cdiv(window, block_size) + 1
    # ...without over-provisioning by more than the alignment step
    assert wb - (cdiv(window, block_size) + 1) < PAGE_TABLE_STICK_BLOCK_ALIGNMENT


def test_sliding_window_blocks_gemma4_case():
    # gemma-4: 1024-token window, 32-token blocks -> 33 needed -> 40 aligned
    assert sliding_window_blocks(1024, 32) == 40


def test_phys_blocks_reserves_one_subring_per_slot_plus_null():
    assert sliding_ring_phys_blocks(40, 32) == 40 * 32 + 1
    # slot s occupies [1 + s*wb, 1 + (s+1)*wb); the last slot must fit exactly
    wb, n = 40, 32
    assert 1 + (n - 1) * wb + wb == sliding_ring_phys_blocks(wb, n)


def test_reserve_bytes_uses_unsharded_page_size():
    # vLLM sizes the pool with the full page_size_bytes and the worker inflates
    # the budget by the shard factor, so the reservation must NOT be divided.
    wb, n, page = 40, 32, 256 * 1024
    assert sliding_ring_reserve_bytes(wb, n, page) == (wb * n + 1) * page


def test_page_table_width_is_capped_by_the_buffer():
    assert sliding_page_table_width(40, 4096) == 40  # ring is narrower
    assert sliding_page_table_width(40, 8) == 8  # buffer is narrower


def _slots(rows, batch, mapping, free):
    return list(assign_ring_slots(rows, batch, mapping, free))


def test_assign_ring_slots_is_stable_across_condense():
    mapping, free = {}, list(range(4))
    assert _slots(["a", "b", "c"], ["a", "b", "c"], mapping, free) == [0, 1, 2]
    # "a" finishes; InputBatch condenses and reorders the remaining rows
    assert _slots(["c", "b"], ["c", "b"], mapping, free) == [2, 1]
    # a request must never change slot while it is alive
    assert mapping["b"] == 1 and mapping["c"] == 2


def test_assign_ring_slots_reclaims_and_reuses_smallest():
    mapping, free = {}, list(range(3))
    _slots(["a", "b", "c"], ["a", "b", "c"], mapping, free)
    # "a" and "b" leave -> their slots come back, smallest handed out first
    assert _slots(["c", "d"], ["c", "d"], mapping, free) == [2, 0]
    assert _slots(["c", "d", "e"], ["c", "d", "e"], mapping, free) == [2, 0, 1]


def test_assign_ring_slots_does_not_free_live_requests_on_a_partial_pass():
    # rows may be a subset of the batch (multi-pass prepare); reclaiming against
    # the full batch keeps the un-listed request's slot reserved.
    mapping, free = {}, list(range(2))
    _slots(["a", "b"], ["a", "b"], mapping, free)
    assert _slots(["a"], ["a", "b"], mapping, free) == [0]
    assert mapping["b"] == 1 and 1 not in free


def test_assign_ring_slots_exhausted_pool_raises():
    mapping, free = {}, [0]
    _slots(["a"], ["a"], mapping, free)
    with pytest.raises(IndexError):
        _slots(["a", "b"], ["a", "b"], mapping, free)


def test_page_table_stays_inside_each_users_subring():
    wb, width = 8, 8
    start_block = np.array([0, 3], dtype=np.int64)
    slots = np.array([0, 2], dtype=np.int64)
    pt = build_sliding_ring_page_table(start_block, slots, wb, width)
    assert pt.shape == (2, width)
    for row, slot in enumerate(slots):
        lo, hi = 1 + slot * wb, 1 + (slot + 1) * wb
        assert pt[row].min() >= lo and pt[row].max() < hi
        # every block of the sub-ring is used exactly once
        assert sorted(pt[row].tolist()) == list(range(lo, hi))
    # block 0 is reserved as the null sink and must never be handed out
    assert (pt != 0).all()


def test_page_table_below_window_is_identity_within_the_subring():
    wb, width = 8, 8
    pt = build_sliding_ring_page_table(
        np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), wb, width
    )
    assert pt[0].tolist() == [1 + wb + i for i in range(width)]


def test_page_table_wraps_once_past_the_window():
    wb, width = 8, 8
    # start_block=3 -> logical order 3,4,..,7,0,1,2 inside the sub-ring
    pt = build_sliding_ring_page_table(
        np.array([3], dtype=np.int64), np.array([0], dtype=np.int64), wb, width
    )
    assert pt[0].tolist() == [1 + ((3 + i) % wb) for i in range(width)]
