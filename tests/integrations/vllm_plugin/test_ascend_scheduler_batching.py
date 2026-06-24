# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test: chunked prefill must batch users, not serialize them.

The TT prefill graph is batched ``[max_num_seqs, chunk]`` -- a single prefill
step prefills up to ``max_num_seqs`` users at once. The per-step token budget
(``max_num_batched_tokens``) is therefore sized as ``chunk x max_num_seqs`` so
all same-stage waiting users batch into one step.

The bug (tt-xla #4986 / tt-inference-server #4326): sizing the budget at the
per-sequence ``chunk`` alone let only ONE user's chunk through per step, so 32
waiting users of ISL == chunk prefilled one-per-step -- serialized, ~32x slower
TTFT. The fix decouples two quantities that had been conflated:

  * ``tt_prefill_chunk_size`` -- the PER-SEQUENCE cap (bounds the prefill bucket
    / activation), and
  * ``max_num_batched_tokens`` -- the per-STEP, batch-wide budget
    (= chunk x max_num_seqs).

These tests drive the scheduler directly on CPU (no TT device / model
execution): the batching decision is pure scheduler bookkeeping.
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
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm_tt.scheduler.ascend_scheduler import AscendScheduler

_BLOCK_SIZE = 16
_NUM_BLOCKS = 4000
_MODEL = "facebook/opt-125m"  # tiny config, cached; no weights are loaded


def _make_scheduler(chunk: int, max_num_seqs: int, max_model_len: int = 2048):
    """Build an AscendScheduler configured exactly as platform.py does for
    chunked prefill: budget = chunk x max_num_seqs, with tt_prefill_chunk_size
    set to the per-sequence cap."""
    model_config = ModelConfig(model=_MODEL, dtype="float16", seed=42)
    budget = chunk * max_num_seqs
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=budget,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    # TT-internal attributes set by platform.py for the chunked-prefill path.
    scheduler_config.chunked_prefill_enabled = True
    scheduler_config.tt_prefill_chunk_size = chunk

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


@pytest.mark.push
@pytest.mark.cpu
def test_same_stage_prefills_batch_in_one_step():
    """N fresh users of ISL == chunk must all prefill in a SINGLE step.

    Regression guard: with the budget mis-sized at ``chunk`` (not
    ``chunk x max_num_seqs``) only one user fit per step and prefills
    serialized. With the fix all N users batch into the first step.
    """
    chunk = 2 * _BLOCK_SIZE  # 32, block-aligned
    n_users = 8
    sched = _make_scheduler(chunk=chunk, max_num_seqs=n_users)
    for i in range(n_users):
        sched.add_request(_make_request(f"r{i}", num_tokens=chunk))

    out = sched.schedule()

    # All N users scheduled in this one step, each taking a full chunk.
    assert len(out.num_scheduled_tokens) == n_users, (
        f"expected all {n_users} same-stage prefills to batch into one step, "
        f"got {len(out.num_scheduled_tokens)} (prefills serialized -- the "
        f"budget regression). scheduled={out.num_scheduled_tokens}"
    )
    assert all(v == chunk for v in out.num_scheduled_tokens.values())
    assert out.total_num_scheduled_tokens == chunk * n_users


@pytest.mark.push
@pytest.mark.cpu
def test_long_prompt_capped_at_per_seq_chunk():
    """A prompt longer than the chunk must take only one chunk per step even
    though the batch-wide budget is much larger.

    Guards the per-sequence cap: without it a single long prompt would consume
    the whole ``chunk x max_num_seqs`` budget in one step (no chunking), blowing
    the prefill bucket / DRAM that chunking exists to bound.
    """
    chunk = 2 * _BLOCK_SIZE  # 32
    n_users = 8  # budget = 256, far larger than the 96-token prompt below
    sched = _make_scheduler(chunk=chunk, max_num_seqs=n_users)
    sched.add_request(_make_request("r0", num_tokens=3 * chunk))  # 96 tokens

    out = sched.schedule()

    assert out.num_scheduled_tokens["r0"] == chunk, (
        "long prompt must be capped at the per-seq chunk, not the batch-wide "
        f"budget; scheduled {out.num_scheduled_tokens['r0']} tokens this step"
    )


@pytest.mark.push
@pytest.mark.cpu
def test_chunk_boundary_avoids_single_token_remainder():
    """ISL = chunk + 1 must NOT produce a 1-token final chunk.

    A 1-token "prefill" chunk would be misrouted to the decode path
    (is_prefill := query_len > 1). The scheduler's _block_aligned_chunk backs
    off by one block so the final chunk is > 1 token. With ISL=33, chunk=32,
    block_size=16: step 1 takes 16 (not 32), step 2 takes the remaining 17.
    """
    chunk = 2 * _BLOCK_SIZE  # 32
    n_users = 4
    isl = chunk + 1  # 33 tokens: one more than chunk
    sched = _make_scheduler(chunk=chunk, max_num_seqs=n_users)
    sched.add_request(_make_request("r0", num_tokens=isl))

    # Step 1: backs off by one block to avoid leaving a 1-token final chunk.
    out1 = sched.schedule()
    expected_step1 = chunk - _BLOCK_SIZE  # 16
    assert out1.num_scheduled_tokens["r0"] == expected_step1, (
        f"step 1: expected {expected_step1} tokens (backed off to avoid "
        f"1-token remainder), got {out1.num_scheduled_tokens['r0']}"
    )

    # Step 2: takes the rest (17 tokens) as the final chunk.
    out2 = sched.schedule()
    expected_step2 = isl - expected_step1  # 17
    assert out2.num_scheduled_tokens["r0"] == expected_step2, (
        f"step 2: expected {expected_step2} tokens (final chunk), "
        f"got {out2.num_scheduled_tokens['r0']}"
    )
