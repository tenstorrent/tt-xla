# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the hybrid KV sliding-window ring buffers.

Shared by the worker (byte reservation) and the model runner (physical sizing,
slot assignment, page-table rotation) so the geometry cannot drift between the
two: the worker reserves exactly what the runner later allocates.

Everything here is pure (numpy / ints only, no device or vLLM state) so it can
be unit tested without hardware.
"""

from __future__ import annotations

import numpy as np
from vllm.utils.math_utils import cdiv

# A paged-op page-table stick is window_blocks int32 block ids and must be
# 32-byte aligned: window_blocks * 4 % 32 == 0 -> window_blocks % 8 == 0.
PAGE_TABLE_STICK_BLOCK_ALIGNMENT = 8


def sliding_window_blocks(
    sliding_window: int, block_size: int, max_model_len: int
) -> int:
    """Blocks per user in a sliding ring, page-table-stick aligned.

    ``cdiv(window, block_size) + 1`` covers a window straddling a block
    boundary; the round-up satisfies the stick alignment above. The ring is
    over-provisioned to that width -- the sliding_window mask still limits
    attention to the real window.

    Capped by ``max_model_len``: no request can ever hold more context than
    that, so a layer's static sliding_window (e.g. 1024) must not size the
    ring past the model_len a shorter-context run actually needs.
    """
    window = min(sliding_window, max_model_len)
    n = cdiv(window, block_size) + 1
    align = PAGE_TABLE_STICK_BLOCK_ALIGNMENT
    return ((n + align - 1) // align) * align


def sliding_ring_is_profitable(
    sliding_window: int, block_size: int, max_model_len: int
) -> bool:
    """Whether a sliding ring is cheaper than leaving the layer in the pool.

    A ring costs ``sliding_window_blocks()`` blocks per user per layer; plain
    full attention costs ``cdiv(max_model_len, block_size)``. The ring only wins
    once the window actually clips the context -- below that the window never
    slides, yet the ring still pays its ``+1`` straddle block and the stick
    alignment round-up, making it strictly more expensive.
    """
    ring_blocks = sliding_window_blocks(sliding_window, block_size, max_model_len)
    return ring_blocks < cdiv(max_model_len, block_size)


def sliding_ring_phys_blocks(window_blocks: int, max_num_reqs: int) -> int:
    """Physical blocks for one sliding layer: one sub-ring per batch slot, plus
    a leading null block (index 0) that padded / inactive rows write to."""
    return window_blocks * max_num_reqs + 1


def sliding_ring_reserve_bytes(
    window_blocks: int, max_num_reqs: int, page_size_bytes: int
) -> int:
    """Bytes one sliding layer's ring costs, in vLLM's KV accounting units.

    NOTE the units: vLLM sizes the block pool with the full, un-sharded
    ``page_size_bytes``, and the worker inflates the budget by the KV shard
    factor to match. The reservation is therefore counted the same way -- do NOT
    divide by the shard count here, or the rings get under-reserved by that
    factor.
    """
    return sliding_ring_phys_blocks(window_blocks, max_num_reqs) * page_size_bytes


def sliding_page_table_width(window_blocks: int, base_width: int) -> int:
    """Page-table width for a sliding group: the ring never needs more columns
    than its window, but stays within the full-attention buffer width."""
    return min(window_blocks, base_width)


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
            slot = free_ring_slots.pop()
            req_ring_slot[rid] = slot
        slots.append(slot)
    return np.array(slots, dtype=np.int64)


def build_sliding_ring_page_table(
    start_block: np.ndarray,
    slots: np.ndarray,
    window_blocks: int,
    width: int,
) -> np.ndarray:
    """Page table (rows x width) pointing into each request's own sub-ring.

    Row i owns physical blocks ``[1 + slots[i]*wb, 1 + (slots[i]+1)*wb)``. The
    window is rotated into logical order (``% wb``) so the fill lands on the
    correct ring position and the read's causal + sliding_window mask stays
    correct. Block 0 is left for the shared null sink.
    """
    jr = np.arange(width)
    rot = (start_block[:, None] + jr[None, :]) % window_blocks
    return rot + (1 + slots * window_blocks)[:, None]
