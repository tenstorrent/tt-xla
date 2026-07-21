# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Regression test for tt-xla #5664: a partial prefill chunk must not deadlock
decode. A partial is kept out of ``self.running`` but its id was added to
``scheduled_req_ids``; ``update_from_output`` only cleared ids found in
``self.running``, so the partial's id lingered and the decode path (gated on
``scheduled_req_ids`` being empty) starved every other running request. Drives
the scheduler directly on CPU -- no TT device or model execution.
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
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm_tt.scheduler.ascend_scheduler import AscendScheduler

_BLOCK_SIZE = 16
_NUM_BLOCKS = 4000
_MODEL = "facebook/opt-125m"  # tiny config, cached; no weights are loaded
_MAX_NUM_SEQS = 4


def _make_scheduler(max_model_len: int = 2048) -> AscendScheduler:
    model_config = ModelConfig(model=_MODEL, dtype="float16", seed=42)
    scheduler_config = SchedulerConfig(
        max_num_seqs=_MAX_NUM_SEQS,
        max_num_batched_tokens=100_000,  # overridden per-step in the test
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    scheduler_config.tt_chunked_prefill_enabled = True

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


def _make_request(rid: str, num_tokens: int, max_tokens: int = 20) -> Request:
    init_none_hash(sha256)
    return Request(
        request_id=rid,
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )


def _fake_model_output(sched_out) -> ModelRunnerOutput:
    """Stand-in output: every scheduled request produced one token."""
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[7] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={r: None for r in req_ids},
    )


@pytest.mark.push
@pytest.mark.cpu
def test_partial_prefill_chunk_does_not_deadlock_decode():
    sched = _make_scheduler()

    # Step 1: fill max_num_seqs - 1 running slots with requests that fully
    # prefill in one shot.
    sched.max_num_scheduled_tokens = 100_000
    for i in range(_MAX_NUM_SEQS - 1):
        sched.add_request(_make_request(f"filler-{i}", num_tokens=_BLOCK_SIZE))
    out1 = sched.schedule()
    sched.update_from_output(out1, _fake_model_output(out1))
    assert len(sched.running) == _MAX_NUM_SEQS - 1
    assert not sched.scheduled_req_ids

    # Step 2: one more request with a budget too small for its whole prompt, so
    # it gets a partial chunk: kept out of self.running but its id lands in
    # scheduled_req_ids.
    sched.max_num_scheduled_tokens = 3 * _BLOCK_SIZE
    sched.add_request(_make_request("partial-victim", num_tokens=16 * _BLOCK_SIZE))
    out2 = sched.schedule()
    assert out2.num_scheduled_tokens.get("partial-victim", 0) > 0
    assert "partial-victim" not in {r.request_id for r in sched.running}
    assert "partial-victim" in sched.scheduled_req_ids
    sched.update_from_output(out2, _fake_model_output(out2))

    # The regression check: the partial's id must be cleared after its step.
    assert not sched.scheduled_req_ids, (
        "scheduled_req_ids still has stuck entries after the partial's own "
        f"step: {sched.scheduled_req_ids} (this is the #5664 deadlock)"
    )

    # Step 3: budget below one block defers the partial's continuation; decode
    # must still proceed for the other running requests.
    sched.max_num_scheduled_tokens = _BLOCK_SIZE - 1
    out3 = sched.schedule()
    assert out3.total_num_scheduled_tokens > 0, (
        "decode did not proceed for the other running requests (the #5664 " "deadlock)"
    )
