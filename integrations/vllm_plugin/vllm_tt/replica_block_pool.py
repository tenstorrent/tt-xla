# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-replica physical slot allocation for a DP-sharded KV block pool."""

from .logger import tt_init_logger

logger = tt_init_logger(__name__)


class ReplicaBlockPool:
    """Maps vLLM's global block ids onto per-replica physical cache slots.

    Under DP+TP each replica's KV tensor holds only ``slots_per_replica`` blocks
    while vLLM allocates ids from one global pool with no notion of replicas. A
    request's row fixes its replica for life, so ids are translated here against
    that replica's own free list. Plain ``id % slots_per_replica`` is not enough:
    two co-resident rows on one replica can draw ids a whole stride apart and
    would land on the same slot.

    Slot 0 is reserved, mirroring vLLM's null block — padding and unscheduled
    rows point their write page table there.

    Ids are refcounted per replica, so rows sharing a cached prefix block within
    a replica share one physical slot.
    """

    def __init__(
        self, num_replicas: int, slots_per_replica: int, rows_per_replica: int
    ):
        assert slots_per_replica > 1, "need at least one slot besides the null block"
        assert rows_per_replica > 0
        self.slots_per_replica = slots_per_replica
        self.rows_per_replica = rows_per_replica
        self._free = [
            list(range(slots_per_replica - 1, 0, -1)) for _ in range(num_replicas)
        ]
        self._slot_of: list[dict[int, int]] = [{} for _ in range(num_replicas)]
        self._refs: list[dict[int, int]] = [{} for _ in range(num_replicas)]
        self._ids_of_row: dict[int, list[int]] = {}

    def replica_of_row(self, row: int) -> int:
        return row // self.rows_per_replica

    def add_row(self, row: int, block_ids) -> list[int]:
        """Claim slots for a row that is being (re)filled from scratch."""
        self.clear_row(row)
        return self.append_row(row, block_ids)

    def append_row(self, row: int, block_ids) -> list[int]:
        """Claim slots for ids appended to a row, returning them in order."""
        replica = self.replica_of_row(row)
        held = self._ids_of_row.setdefault(row, [])
        slot_of = self._slot_of[replica]
        refs = self._refs[replica]
        free = self._free[replica]

        slots = []
        for block_id in block_ids:
            block_id = int(block_id)
            if block_id == 0:
                # vLLM's null block; unrefcounted there, so unrefcounted here.
                slots.append(0)
                continue
            slot = slot_of.get(block_id)
            if slot is None:
                if not free:
                    raise RuntimeError(
                        f"DP replica {replica} ran out of KV cache slots "
                        f"({self.slots_per_replica} per replica). Lower "
                        "max_num_seqs or max_model_len, or raise "
                        "gpu_memory_utilization."
                    )
                slot = free.pop()
                slot_of[block_id] = slot
                refs[block_id] = 0
            refs[block_id] += 1
            held.append(block_id)
            slots.append(slot)
        return slots

    def clear_row(self, row: int) -> None:
        """Release every slot the row held."""
        held = self._ids_of_row.pop(row, None)
        if not held:
            return
        replica = self.replica_of_row(row)
        slot_of = self._slot_of[replica]
        refs = self._refs[replica]
        free = self._free[replica]
        for block_id in held:
            remaining = refs[block_id] - 1
            if remaining:
                refs[block_id] = remaining
                continue
            del refs[block_id]
            free.append(slot_of.pop(block_id))

    def free_slots(self, replica: int) -> int:
        """Unclaimed slots on a replica. For tests and logging."""
        return len(self._free[replica])
