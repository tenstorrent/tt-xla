# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: AscendScheduler must survive an aborted request.

When an in-flight request is aborted (client disconnect, read-timeout, explicit
``engine.abort()``), the scheduler must drop it and keep serving everyone else.

The original bug: AscendScheduler keeps a mid-prefill chunked request (status
``RUNNING``) in ``self.skipped_waiting`` rather than ``self.running`` until its
prefill completes (tt-xla #4986). The base ``finish_requests`` removes
``RUNNING`` requests only from ``self.running``, so an aborted mid-prefill
request lingered in ``skipped_waiting`` and the next ``schedule()`` raised
``RuntimeError: Invalid request status: FINISHED_ABORTED`` -- killing the whole
EngineCore for every in-flight and future request.

These tests drive the scheduler directly on CPU (no TT device / model
execution): the bug is purely scheduler queue bookkeeping.
"""

import pytest
import torch
from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm_tt.scheduler.ascend_scheduler import AscendScheduler

_BLOCK_SIZE = 16
_NUM_BLOCKS = 1000
_MODEL = "facebook/opt-125m"  # tiny config, cached; no weights are loaded


def _make_scheduler(max_num_batched_tokens: int, max_model_len: int = 512):
    model_config = ModelConfig(model=_MODEL, dtype="float16", seed=42)
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    # The TT platform sets this custom (non-vLLM) attribute to gate the
    # AscendScheduler chunked-prefill path; mirror platform.py here.
    scheduler_config.chunked_prefill_enabled = True

    cache_config = CacheConfig(
        block_size=_BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    cache_config.num_gpu_blocks = _NUM_BLOCKS

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=_NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return AscendScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=_BLOCK_SIZE,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _make_request(rid: str, num_tokens: int) -> Request:
    init_none_hash(sha256)
    return Request(
        request_id=rid,
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=16),
        pooling_params=None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )


def _in_skipped_waiting(sched: AscendScheduler, rid: str) -> bool:
    return rid in {r.request_id for r in sched.skipped_waiting}


@pytest.mark.push
@pytest.mark.cpu
def test_abort_mid_prefill_chunk_does_not_crash_scheduler():
    """Aborting a mid-prefill chunked request must not raise on next schedule()."""
    # Budget << prompt -> the prompt is split into multiple prefill chunks.
    sched = _make_scheduler(max_num_batched_tokens=32)
    req = _make_request("r0", num_tokens=128)
    sched.add_request(req)

    # Step 1: one prefill chunk scheduled. The request is parked in
    # skipped_waiting with status RUNNING (NOT in self.running) -- this is the
    # state that triggered the bug.
    sched.schedule()
    assert req.status == RequestStatus.RUNNING
    assert req not in sched.running
    assert _in_skipped_waiting(sched, "r0")

    # Client disconnects mid-prefill -> abort.
    sched.finish_requests("r0", RequestStatus.FINISHED_ABORTED)
    assert req.status == RequestStatus.FINISHED_ABORTED
    # The aborted request must be purged from the waiting-side queue.
    assert not _in_skipped_waiting(sched, "r0")

    # The next scheduler step must not raise (regression: it raised
    # "RuntimeError: Invalid request status: FINISHED_ABORTED" here).
    sched.schedule()


@pytest.mark.push
@pytest.mark.cpu
def test_abort_one_request_keeps_serving_another():
    """A single aborted request must not take down other in-flight requests."""
    sched = _make_scheduler(max_num_batched_tokens=32)
    victim = _make_request("victim", num_tokens=128)  # long -> chunked prefill
    survivor = _make_request("survivor", num_tokens=8)  # short -> fits one chunk
    sched.add_request(victim)
    sched.add_request(survivor)

    sched.schedule()
    assert victim.status == RequestStatus.RUNNING
    assert _in_skipped_waiting(sched, "victim")

    sched.finish_requests("victim", RequestStatus.FINISHED_ABORTED)

    # Engine keeps stepping; the survivor is still tracked and schedulable.
    for _ in range(3):
        sched.schedule()
    assert "survivor" in sched.requests
    assert survivor.status != RequestStatus.FINISHED_ABORTED
