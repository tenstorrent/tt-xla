# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for hybrid KV sliding-window ring buffers.

Shared by the worker (byte reservation) and model runner (physical sizing /
page-table rotation) so the window_blocks formula cannot drift.
"""

from __future__ import annotations

import numpy as np
from vllm.utils.math_utils import cdiv


def sliding_window_blocks(sliding_window: int, block_size: int) -> int:
    """Blocks per user in a sliding ring, 32-byte page-table stick aligned.

    ``cdiv(window, block_size) + 1`` covers a window that straddles a block
    boundary; rounding up to a multiple of 8 makes ``window_blocks * sizeof(int32)
    % 32 == 0`` for the paged-op page-table stick.
    """
    n = cdiv(sliding_window, block_size) + 1
    return ((n + 7) // 8) * 8


def sliding_ring_phys_blocks(window_blocks: int, max_num_reqs: int) -> int:
    """Physical block count: one sub-ring per batch slot plus a leading null."""
    return window_blocks * max_num_reqs + 1


def sliding_ring_reserve_bytes(
    window_blocks: int,
    max_num_reqs: int,
    page_size_bytes: int,
    kv_shard_size: int,
) -> int:
    """Per-device byte cost of one sliding layer's ring (after KV-head sharding)."""
    return (
        sliding_ring_phys_blocks(window_blocks, max_num_reqs) * page_size_bytes
    ) // kv_shard_size


def assign_ring_slots(
    row_req_ids: list[str],
    batch_req_ids: list[str],
    req_ring_slot: dict[str, int],
    free_ring_slots: list[int],
) -> np.ndarray:
    """Assign each request a stable per-user sliding-ring slot.

    The physical sliding ring is split into one sub-ring per slot; a request
    must keep the SAME slot for its lifetime, otherwise a subsequent decode step
    would read/write a different sub-ring. Keying the slot by req_id (not batch
    row) makes it survive InputBatch row-condensing, which reorders rows when
    requests finish.

    ``row_req_ids`` are the requests to return slots for (the current prepare
    pass's rows). ``batch_req_ids`` is the full current batch, used to reclaim
    the slots of departed requests -- reclaiming against the full batch (not the
    per-pass subset) means a partial pass never frees a still-live request's
    slot. Freed slots are reused by later requests, whose fresh prefill
    overwrites the sub-ring. Mutates ``req_ring_slot`` and ``free_ring_slots`` in
    place and returns an int64 array of slots aligned with ``row_req_ids``.
    """
    active = set(batch_req_ids)
    for rid in [r for r in req_ring_slot if r not in active]:
        free_ring_slots.append(req_ring_slot.pop(rid))
    # Hand out the smallest free slot first (deterministic across steps).
    free_ring_slots.sort(reverse=True)
    slots = []
    for rid in row_req_ids:
        slot = req_ring_slot.get(rid)
        if slot is None:
            assert free_ring_slots, (
                "no free sliding-ring slots; concurrency exceeds max_num_reqs "
                f"({len(req_ring_slot)} live mappings for "
                f"{len(row_req_ids)} row reqs)"
            )
            slot = free_ring_slots.pop()
            req_ring_slot[rid] = slot
        slots.append(slot)
    return np.array(slots, dtype=np.int64)


def build_sliding_ring_page_table(
    seq_lens: np.ndarray,
    block_size: int,
    window_blocks: int,
    slots: np.ndarray,
    target_num_reqs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a rotated, window-width ring page table and relative cache positions.

    Page-table width is always ``window_blocks`` (never clamped to the
    full-attention path width). Physical IDs land in each user's sub-ring
    ``[1 + slot*wb, 1 + (slot+1)*wb)``.

    Returns:
        page_table: int64 ``[target_num_reqs, window_blocks]``
        cache_position_rel: int64 ``[target_num_reqs]`` (-1 for unused rows)
        start_block: int64 ``[actual_num_reqs]`` logical start block per row
    """
    wb = window_blocks
    actual_num_reqs = int(seq_lens.shape[0])
    pt = np.zeros((target_num_reqs, wb), dtype=np.int64)
    cp_rel = np.full((target_num_reqs,), -1, dtype=np.int64)
    if actual_num_reqs == 0:
        return pt, cp_rel, np.zeros((0,), dtype=np.int64)

    cur_pos_abs = seq_lens.astype(np.int64) - 1
    cur_block = cur_pos_abs // block_size
    num_win = np.minimum(cur_block + 1, wb)
    start_block = cur_block - num_win + 1
    jr = np.arange(wb)
    rot = (start_block[:, None] + jr[None, :]) % wb
    user_base = (1 + slots.astype(np.int64) * wb)[:, None]
    pt[:actual_num_reqs] = rot + user_base
    cp_rel[:actual_num_reqs] = cur_pos_abs - start_block * block_size
    return pt, cp_rel, start_block
