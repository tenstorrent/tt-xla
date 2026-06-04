# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC

"""Per-device BlockPool for DP-only execution.

vLLM's stock BlockPool hands out globally-unique block IDs from a single free
queue. When the KV cache is sharded along ``num_blocks`` for DP-only mode
(annotation ``("batch", None, None, None)``), each device d physically owns
the slice ``[d * N/dp, (d+1) * N/dp)`` — but vLLM may give a request on
device d a block ID outside that range, breaking the contract between
"block ID in page_table" and "physical shelf location".

TTBlockPool fixes this by maintaining dp_size independent free queues:
each request on device d only draws block IDs from device d's queue.

The dp_rank for the current allocation is communicated via the
``_current_dp_rank`` attribute, set by the scheduler before each call to
``kv_cache_manager.allocate_slots(request, ...)``.
"""

from collections.abc import Iterable
from typing import Optional

import vllm.v1.core.kv_cache_coordinator as _kv_cache_coordinator
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

from .logger import tt_init_logger

logger = tt_init_logger(__name__)

# Module-level dp_size, set by ``install_tt_block_pool()`` and read by
# TTBlockPool.__init__. We use a module global because BlockPool is
# constructed inside vLLM's KVCacheCoordinator with a fixed positional
# signature; we can't thread an extra argument through without vendoring
# the coordinator. The platform calls install_tt_block_pool() exactly once
# during ``check_and_update_config`` before any BlockPool is constructed.
_dp_size: int = 1
_installed: bool = False


def install_tt_block_pool(dp_size: int) -> None:
    """Monkey-patch ``vllm.v1.core.kv_cache_coordinator.BlockPool`` with
    TTBlockPool so the KVCacheCoordinator constructs a per-DP-rank pool.

    Idempotent: subsequent calls update ``_dp_size`` but don't re-patch.
    """
    global _dp_size, _installed
    assert dp_size >= 1
    _dp_size = dp_size
    if not _installed:
        _kv_cache_coordinator.BlockPool = TTBlockPool
        _installed = True
        logger.info("Installed TTBlockPool monkey-patch (dp_size=%d).", dp_size)
    else:
        logger.info("TTBlockPool already installed; updated dp_size=%d.", dp_size)


class TTBlockPool(BlockPool):
    """BlockPool partitioned into dp_size contiguous block-id ranges.

    Each range has its own ``FreeKVCacheBlockQueue``. Allocations are routed
    to the queue at index ``self._current_dp_rank`` (set by the scheduler).
    Frees route blocks back to their owning queue based on block_id range.

    Each device's first block in its range is reserved as a "null" block
    (vLLM's null-block sentinel — used as a placeholder for empty page_table
    entries). vLLM canonically references ``self.null_block`` (device 0's),
    but devices 1..dp-1 also need their range's first block reserved so that
    after host-side page_table translation (``page_table -= dp_rank * N/dp``)
    the translated block_id 0 lands on each device's own null block.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: Optional[KVCacheMetricsCollector] = None,
    ):
        # dp_size is read from the module-level global set by
        # install_tt_block_pool() — see module docstring above.
        dp_size = _dp_size
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        assert dp_size >= 1
        assert num_gpu_blocks % dp_size == 0, (
            f"num_gpu_blocks ({num_gpu_blocks}) must be a multiple of "
            f"dp_size ({dp_size}); pad num_blocks at KV-cache allocation."
        )
        # Prefix caching across DP ranks is unsafe: a cached block on
        # device 0 cannot be reused by a request on device 1 (the KV data
        # lives only in device 0's shard). The plumbing here intentionally
        # does not partition the prefix-cache hash table, so callers must
        # disable prefix caching when dp_size > 1.
        if dp_size > 1 and enable_caching:
            raise ValueError(
                "TTBlockPool: prefix caching is unsupported with dp_size>1. "
                "Disable via VllmConfig.cache_config.enable_prefix_caching=False."
            )

        self.num_gpu_blocks = num_gpu_blocks
        self.dp_size = dp_size
        self.blocks_per_dp = num_gpu_blocks // dp_size
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size

        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # Per-device free queues over contiguous block-id ranges.
        self._free_queues: list[FreeKVCacheBlockQueue] = []
        for d in range(dp_size):
            start = d * self.blocks_per_dp
            end = start + self.blocks_per_dp
            self._free_queues.append(FreeKVCacheBlockQueue(self.blocks[start:end]))

        # Reserve each range's first block as a null block (so local block_id 0
        # after translation is always a valid null on every device).
        self._null_blocks: list[KVCacheBlock] = []
        for queue in self._free_queues:
            null = queue.popleft()
            null.is_null = True
            self._null_blocks.append(null)
        # vLLM references ``self.null_block`` directly; expose device 0's.
        self.null_block = self._null_blocks[0]

        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []
        self.metrics_collector = metrics_collector

        # Set by the scheduler before each allocate_slots() call.
        self._current_dp_rank: int = 0

        logger.info(
            "TTBlockPool initialized: num_gpu_blocks=%d, dp_size=%d, blocks_per_dp=%d",
            num_gpu_blocks,
            dp_size,
            self.blocks_per_dp,
        )

    @property
    def free_block_queue(self) -> FreeKVCacheBlockQueue:
        """Current dp_rank's queue. Vendored vLLM code that touches
        ``self.free_block_queue`` directly transparently hits the per-rank
        queue (e.g. ``SlidingWindowManager`` sink-block reservation will
        come from the current rank — acceptable as it's per-request)."""
        return self._free_queues[self._current_dp_rank]

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        queue = self._free_queues[self._current_dp_rank]
        if num_blocks > queue.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from dp_rank "
                f"{self._current_dp_rank} pool (has {queue.num_free_blocks})"
            )
        ret = queue.popleft_n(num_blocks)
        for block in ret:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    def touch(self, blocks) -> None:
        """Bump ref_cnt; remove from the block's owning queue (not the
        current rank's queue, since a cached block may belong to any rank
        when caching is enabled). Prefix caching is disabled for dp>1
        today, so in practice this only matters for dp_size==1."""
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                owning_rank = block.block_id // self.blocks_per_dp
                self._free_queues[owning_rank].remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Route each block back to its owning queue based on block_id range."""
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        # Bucket by owning rank to amortize append_n.
        by_rank: list[list[KVCacheBlock]] = [[] for _ in range(self.dp_size)]
        for block in blocks_list:
            if block.ref_cnt == 0 and not block.is_null:
                rank = block.block_id // self.blocks_per_dp
                by_rank[rank].append(block)
        for d, blocks_d in enumerate(by_rank):
            if blocks_d:
                self._free_queues[d].append_n(blocks_d)

    def get_num_free_blocks(self) -> int:
        """Return the current dp_rank's free count so scheduler watermark
        checks reflect the lane the next request will be drawing from."""
        return self._free_queues[self._current_dp_rank].num_free_blocks

    def get_total_num_free_blocks(self) -> int:
        return sum(q.num_free_blocks for q in self._free_queues)

    def get_usage(self) -> float:
        # Subtract dp_size to account for one null block per device.
        total = self.num_gpu_blocks - self.dp_size
        if total <= 0:
            return 0.0
        return 1.0 - (self.get_total_num_free_blocks() / total)

    def reset_prefix_cache(self) -> bool:
        # Prefix caching disabled for dp>1; fall back to parent for dp==1.
        if self.dp_size > 1:
            return True
        return super().reset_prefix_cache()
