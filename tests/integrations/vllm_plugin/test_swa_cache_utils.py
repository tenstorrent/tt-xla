# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for hybrid KV sliding-window ring helpers.

Device-free coverage for window sizing, stable slot assignment, and page-table
rotation math (follow-ups to the hybrid KV ring buffer work).
"""

import numpy as np
import pytest
from vllm_tt.swa_cache_utils import (
    assign_ring_slots,
    build_sliding_ring_page_table,
    sliding_ring_phys_blocks,
    sliding_ring_reserve_bytes,
    sliding_window_blocks,
)

# Device-free, but the vLLM push matrix only runs device-marked jobs; tag so the
# single_device job collects these (they need no device and run in milliseconds).
pytestmark = [pytest.mark.push, pytest.mark.single_device]


def test_sliding_window_blocks_alignment_and_slack():
    # cdiv(4096, 32) + 1 = 129, round up to multiple of 8 -> 136.
    assert sliding_window_blocks(4096, 32) == 136
    assert sliding_window_blocks(4096, 32) % 8 == 0
    # Exact multiple of block_size still gets +1 slack before rounding.
    assert sliding_window_blocks(128, 32) == 8  # cdiv=4, +1=5 -> 8
    assert sliding_window_blocks(256, 32) == 16  # cdiv=8, +1=9 -> 16


def test_sliding_ring_phys_and_reserve_bytes():
    wb = 8
    assert sliding_ring_phys_blocks(wb, max_num_reqs=4) == 8 * 4 + 1
    assert (
        sliding_ring_reserve_bytes(
            wb, max_num_reqs=4, page_size_bytes=100, kv_shard_size=1
        )
        == 33 * 100
    )
    assert (
        sliding_ring_reserve_bytes(
            wb, max_num_reqs=4, page_size_bytes=100, kv_shard_size=2
        )
        == (33 * 100) // 2
    )


def test_assign_ring_slots_stable_reclaim_and_smallest_free():
    req_ring_slot: dict[str, int] = {}
    free = list(range(4))

    slots = assign_ring_slots(["a", "b"], ["a", "b"], req_ring_slot, free)
    assert list(slots) == [0, 1]
    assert req_ring_slot == {"a": 0, "b": 1}

    # Same ids keep their slots even if row order changes.
    slots = assign_ring_slots(["b", "a"], ["b", "a"], req_ring_slot, free)
    assert list(slots) == [1, 0]

    # Departed "a" frees slot 0; new "c" takes the smallest free (0).
    slots = assign_ring_slots(["b", "c"], ["b", "c"], req_ring_slot, free)
    assert list(slots) == [1, 0]
    assert "a" not in req_ring_slot
    assert req_ring_slot["c"] == 0

    # Partial prepare pass: reclaim against the full batch, not the row subset,
    # so a still-live request not in this pass keeps its slot.
    req_ring_slot = {"x": 0, "y": 1}
    free = [2, 3]
    slots = assign_ring_slots(["x"], ["x", "y"], req_ring_slot, free)
    assert list(slots) == [0]
    assert req_ring_slot == {"x": 0, "y": 1}


def test_assign_ring_slots_empty_free_list_raises():
    req_ring_slot = {"a": 0, "b": 1}
    free: list[int] = []
    with pytest.raises(AssertionError, match="no free sliding-ring slots"):
        assign_ring_slots(["a", "b", "c"], ["a", "b", "c"], req_ring_slot, free)


def test_build_sliding_ring_page_table_below_window_identity():
    wb, bs = 8, 16
    seq_lens = np.array([50], dtype=np.int64)  # cur_block=3 < wb
    slots = np.array([2], dtype=np.int64)
    pt, cp_rel, start_block = build_sliding_ring_page_table(
        seq_lens, bs, wb, slots, target_num_reqs=2
    )
    assert pt.shape == (2, wb)
    assert list(start_block) == [0]
    # Identity rotation into slot-2 sub-ring starting at physical 1+2*8=17.
    assert list(pt[0]) == list(range(17, 17 + wb))
    assert pt[1].sum() == 0
    assert cp_rel[0] == 49
    assert cp_rel[1] == -1


def test_build_sliding_ring_page_table_wrap_example():
    # Analysis example: wb=8, bs=16, slot=2, seq_len=200 ->
    # cur_block=12, start_block=5, rot=[5,6,7,0,1,2,3,4], cp_rel=119.
    wb, bs = 8, 16
    seq_lens = np.array([200], dtype=np.int64)
    slots = np.array([2], dtype=np.int64)
    pt, cp_rel, start_block = build_sliding_ring_page_table(
        seq_lens, bs, wb, slots, target_num_reqs=1
    )
    assert list(start_block) == [5]
    assert list(pt[0]) == [22, 23, 24, 17, 18, 19, 20, 21]
    assert cp_rel[0] == 119


def test_build_sliding_ring_page_table_width_always_wb():
    # Width must stay wb even when a full-attention path width would be smaller
    # (the old min(wb, width) bug).
    wb, bs = 16, 32
    assert wb > 8  # pretend full-path width would have been 8
    seq_lens = np.array([wb * bs], dtype=np.int64)
    slots = np.array([0], dtype=np.int64)
    pt, _, _ = build_sliding_ring_page_table(seq_lens, bs, wb, slots, target_num_reqs=1)
    assert pt.shape[1] == wb
