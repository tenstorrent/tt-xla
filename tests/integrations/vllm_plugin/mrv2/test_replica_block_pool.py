# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the per-replica KV block allocator.

``ReplicaBlockPool`` (see vllm_tt/replica_block_pool.py) rebases vLLM's global
block ids onto the slots a DP replica physically holds. The property that
matters is injectivity per replica: two rows on one replica must never be handed
the same slot. Plain ``id % slots_per_replica`` violates exactly that, so the
first test pins it.
"""

import pytest
from vllm_tt.replica_block_pool import ReplicaBlockPool

pytestmark = [pytest.mark.push, pytest.mark.cpu]

SLOTS = 8
ROWS_PER_REPLICA = 2
REPLICAS = 2


def make_pool(slots=SLOTS):
    return ReplicaBlockPool(REPLICAS, slots, ROWS_PER_REPLICA)


def test_stride_apart_ids_on_one_replica_do_not_collide():
    # Rows 0 and 1 are both replica 0. Ids a full stride apart are what `% slots`
    # would fold onto each other.
    pool = make_pool()
    first = pool.add_row(0, [1, 2, 3])
    second = pool.add_row(1, [1 + SLOTS, 2 + SLOTS, 3 + SLOTS])
    assert not set(first) & set(second)


def test_row_to_replica_mapping():
    pool = make_pool()
    assert [pool.replica_of_row(r) for r in range(4)] == [0, 0, 1, 1]


def test_slots_stay_within_the_replica_slice():
    pool = make_pool()
    for row in range(4):
        for slot in pool.add_row(row, [10 * row + 1, 10 * row + 2]):
            assert 0 < slot < SLOTS


def test_replicas_allocate_independently():
    # Same global id on two replicas is two distinct physical copies.
    pool = make_pool()
    assert pool.add_row(0, [5]) == pool.add_row(2, [5])


def test_clear_row_returns_slots():
    pool = make_pool()
    before = pool.free_slots(0)
    pool.add_row(0, [1, 2, 3])
    assert pool.free_slots(0) == before - 3
    pool.clear_row(0)
    assert pool.free_slots(0) == before


def test_slots_are_reused_after_clear():
    pool = make_pool()
    first = pool.add_row(0, [1, 2])
    pool.clear_row(0)
    assert sorted(pool.add_row(0, [90, 91])) == sorted(first)


def test_append_extends_without_releasing():
    pool = make_pool()
    head = pool.add_row(0, [1, 2])
    tail = pool.append_row(0, [3])
    assert not set(head) & set(tail)
    assert pool.free_slots(0) == SLOTS - 1 - 3


def test_add_row_releases_the_previous_occupant():
    pool = make_pool()
    pool.add_row(0, [1, 2, 3])
    before = pool.free_slots(0)
    pool.add_row(0, [4])
    assert pool.free_slots(0) == before + 2


def test_shared_id_within_a_replica_shares_one_slot():
    # A cached prefix block hit by two rows of the same replica.
    pool = make_pool()
    assert pool.add_row(0, [7]) == pool.add_row(1, [7])
    assert pool.free_slots(0) == SLOTS - 1 - 1


def test_shared_id_is_freed_only_by_its_last_user():
    pool = make_pool()
    pool.add_row(0, [7])
    pool.add_row(1, [7])
    held = pool.free_slots(0)
    pool.clear_row(0)
    assert pool.free_slots(0) == held
    pool.clear_row(1)
    assert pool.free_slots(0) == held + 1


def test_null_block_is_never_allocated():
    pool = make_pool()
    assert pool.add_row(0, [0, 0]) == [0, 0]
    # Reserved, so it costs no slot and cannot be handed to a real block.
    assert pool.free_slots(0) == SLOTS - 1
    assert 0 not in pool.add_row(1, list(range(1, SLOTS)))


def test_exhaustion_raises_instead_of_reusing_a_live_slot():
    pool = make_pool()
    pool.add_row(0, list(range(1, SLOTS)))
    with pytest.raises(RuntimeError, match="ran out of KV cache slots"):
        pool.add_row(1, [1000])


def test_clear_row_is_idempotent():
    pool = make_pool()
    pool.add_row(0, [1])
    pool.clear_row(0)
    pool.clear_row(0)
    assert pool.free_slots(0) == SLOTS - 1
