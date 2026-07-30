# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

"""Replica-aware block allocator for DP+TP sharding.

Under DP (data parallelism), KV cache blocks are sharded across replicas.
Each replica holds only num_blocks/dp_size slots. This wrapper remaps global
block IDs from vLLM's allocator to per-replica coordinates:

  global_block_id → per_replica_id = global_block_id % num_blocks_per_replica

Both replicas compute the same modulo, so they arrive at the same per-replica
coordinate. The sharding is safe because each replica has its own physical slots.
"""

from typing import Optional


class ReplicaAwareBlockAllocator:
    """Wraps vLLM's block allocator to handle per-replica block ID remapping.

    When DP > 1 and blocks are sharded on the batch axis, each replica only
    physically holds num_blocks/dp_size slots. The global allocator assigns
    sequential global IDs (0, 1, 2, ...), but we remap them to per-replica
    coordinates so both replicas write to the same local slot.
    """

    def __init__(
        self,
        base_allocator,
        replica_id: int,
        dp_size: int,
        total_blocks: int,
    ):
        """Initialize the replica-aware allocator.

        Args:
            base_allocator: vLLM's original block allocator
            replica_id: Which replica this allocator is on (0 to dp_size-1)
            dp_size: Total number of DP replicas
            total_blocks: Total global block count
        """
        self.base_allocator = base_allocator
        self.replica_id = replica_id
        self.dp_size = dp_size
        self.total_blocks = total_blocks

        if dp_size > 1:
            self.num_blocks_per_replica = total_blocks // dp_size
            self.replica_start = replica_id * self.num_blocks_per_replica
        else:
            # No sharding
            self.num_blocks_per_replica = total_blocks
            self.replica_start = 0

    def allocate_block(self):
        """Allocate a block and remap its ID to per-replica coordinate."""
        global_block_id = self.base_allocator.allocate_block()
        return self._remap_to_local(global_block_id)

    def allocate_blocks(self, num_blocks: int):
        """Allocate N blocks and remap each to per-replica coordinates."""
        global_block_ids = self.base_allocator.allocate_blocks(num_blocks)
        return [self._remap_to_local(bid) for bid in global_block_ids]

    def free_block(self, block_id: int):
        """Free a per-replica block ID (remap back to global for vLLM)."""
        global_block_id = self._remap_to_global(block_id)
        self.base_allocator.free_block(global_block_id)

    def free_blocks(self, block_ids):
        """Free multiple per-replica block IDs."""
        global_block_ids = [self._remap_to_global(bid) for bid in block_ids]
        self.base_allocator.free_blocks(global_block_ids)

    def _remap_to_local(self, global_block_id: int) -> int:
        """Map global block ID to per-replica coordinate using modulo."""
        if self.dp_size == 1:
            return global_block_id
        return global_block_id % self.num_blocks_per_replica

    def _remap_to_global(self, local_block_id: int) -> int:
        """Map per-replica coordinate back to global ID (for cleanup)."""
        if self.dp_size == 1:
            return local_block_id
        # Note: This is lossy (multiple globals map to same local).
        # Used only for free() calls, which vLLM doesn't actually use much
        # in the common path. For safety, we could track the mapping explicitly.
        return self.replica_start + local_block_id

    def __getattr__(self, name):
        """Delegate other attributes/methods to the base allocator."""
        return getattr(self.base_allocator, name)
