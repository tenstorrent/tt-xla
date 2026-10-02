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
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
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
    scheduler_config.tt_chunked_prefill_enabled = True
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


def _fake_model_output(sched_out) -> ModelRunnerOutput:
    """Stand-in model output: each scheduled request sampled one token."""
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


@pytest.mark.push
@pytest.mark.cpu
def test_new_prefill_preempts_decode_then_decode_combines_old_and_new():
    """When new prompts arrive during decode, prefill is scheduled first;
    the next decode-only step includes both old and newly-prefilled requests.
    """
    chunk = 2 * _BLOCK_SIZE  # 32
    sched = _make_scheduler(chunk=chunk, max_num_seqs=4)

    # Step 1: two initial requests fully prefill and enter running.
    sched.add_request(_make_request("old-0", num_tokens=chunk))
    sched.add_request(_make_request("old-1", num_tokens=chunk))
    out1 = sched.schedule()
    sched.update_from_output(out1, _fake_model_output(out1))
    assert {"old-0", "old-1"}.issubset({r.request_id for r in sched.running})

    # Step 2: no waiting requests -> decode runs for existing running requests.
    out2 = sched.schedule()
    assert {"old-0", "old-1"}.issubset(set(out2.num_scheduled_tokens.keys()))
    sched.update_from_output(out2, _fake_model_output(out2))

    # New prompt arrives while decode is active and there is running capacity.
    assert len(sched.running) < sched.max_num_running_reqs
    sched.add_request(_make_request("new-0", num_tokens=chunk))

    # Step 3: prefill-first policy should schedule only the new prefill,
    # not decode old requests in the same step.
    out3 = sched.schedule()
    step3_ids = set(out3.num_scheduled_tokens.keys())
    assert "new-0" in step3_ids
    assert "old-0" not in step3_ids and "old-1" not in step3_ids
    sched.update_from_output(out3, _fake_model_output(out3))

    # Step 4: with no pending prefills, decode resumes and should include
    # both old and newly-prefilled running requests.
    out4 = sched.schedule()
    step4_ids = set(out4.num_scheduled_tokens.keys())
    assert {"old-0", "old-1", "new-0"}.issubset(step4_ids)


# --------------------------------------------------------------------------- #
# Chunked multi-modal inputs (tt-xla #5824)
# --------------------------------------------------------------------------- #


def _make_mm_request(rid: str, num_tokens: int, offset: int, length: int) -> Request:
    """A request whose prompt carries one image placeholder at ``offset``.

    ``data=None`` is the "already cached upstream" form; the scheduler never
    reads it, only ``mm_position``.
    """
    init_none_hash(sha256)
    return Request(
        request_id=rid,
        prompt_token_ids=[1] * num_tokens,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=16),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=f"{rid}-img0",
                mm_position=PlaceholderRange(offset=offset, length=length),
            )
        ],
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )


def _fake_prefill_only_output(sched_out) -> ModelRunnerOutput:
    """Model output for a step that sampled nothing.

    The real runner discards the sampled token of a request whose prefill is
    still partial (``discard_sampled_tokens_req_indices``). Emitting a token here
    instead would grow ``request.num_tokens`` and shift every later chunk.
    """
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={r: None for r in req_ids},
    )


def _prefill_chunk_boundaries(sched, req_id: str, isl: int) -> list[int]:
    """Cumulative prompt position after each prefill chunk of ``req_id``."""
    boundaries: list[int] = []
    consumed = 0
    for _ in range(32):
        out = sched.schedule()
        n = out.num_scheduled_tokens.get(req_id, 0)
        if n == 0:
            break
        consumed += n
        boundaries.append(consumed)
        sched.update_from_output(out, _fake_prefill_only_output(out))
        if consumed >= isl:
            break
    return boundaries


def _enable_encoder_cache(sched, cache_size: int, compute_budget: int) -> None:
    """Give the scheduler a real encoder budget.

    ``_make_scheduler`` uses a text-only model, so upstream sizes the encoder
    cache at 0 (``supports_mm_inputs`` is False). Install a real one so the
    multimodal scheduling path is exercised rather than short-circuited.
    """
    sched.encoder_cache_manager = EncoderCacheManager(cache_size=cache_size)
    sched.max_num_encoder_input_tokens = compute_budget
    sched.scheduler_config.disable_chunked_mm_input = False


@pytest.mark.push
@pytest.mark.cpu
def test_chunked_mm_item_splits_across_chunks_and_encodes_once():
    """An image larger than the prefill chunk must span chunks, encoded once.

    This is the whole point of tt-xla #5824: the vision encoder is atomic (it
    runs on the whole item and the output is parked in the encoder cache), while
    the decoder placeholder tokens are chunkable. So the item must appear in
    ``scheduled_encoder_inputs`` on exactly one step -- the first chunk that
    touches it -- and later chunks must re-slice the cache instead of
    re-encoding.
    """
    chunk = 2 * _BLOCK_SIZE  # 32
    img_len = 3 * chunk  # 96: spans several chunks
    isl = _BLOCK_SIZE + img_len + _BLOCK_SIZE  # text, image, text
    sched = _make_scheduler(chunk=chunk, max_num_seqs=1)
    _enable_encoder_cache(sched, cache_size=img_len, compute_budget=img_len)
    sched.add_request(_make_mm_request("r0", isl, offset=_BLOCK_SIZE, length=img_len))

    steps: list[int] = []
    encode_steps: list[int] = []
    for i in range(32):
        out = sched.schedule()
        n = out.num_scheduled_tokens.get("r0", 0)
        if n == 0:
            break
        steps.append(n)
        if out.scheduled_encoder_inputs.get("r0"):
            encode_steps.append(i)
        sched.update_from_output(out, _fake_prefill_only_output(out))
        if sum(steps) >= isl:
            break

    assert sum(steps) == isl, f"chunks {steps} must cover the {isl}-token prompt"
    assert len(steps) > 1, f"prompt should have been split, got a single chunk {steps}"
    # Every non-final chunk must be block-aligned (see _block_aligned_chunk).
    for n in steps[:-1]:
        assert (
            n % _BLOCK_SIZE == 0
        ), f"non-final chunk {n} is not block-aligned: {steps}"
    assert len(encode_steps) == 1, (
        "the encoder must run exactly once for one image; it was scheduled on "
        f"steps {encode_steps} (chunks {steps})"
    )


@pytest.mark.push
@pytest.mark.cpu
def test_disable_chunked_mm_input_keeps_item_whole():
    """With chunking disabled, no chunk boundary may fall inside the image.

    Guards the other half of the tt-xla #5824 platform decision: models with
    multimodal-bidirectional attention (``is_mm_prefix_lm`` -- Gemma-3/Gemma-4)
    keep ``disable_chunked_mm_input=True`` because the placeholder span attends
    to itself.

    Note what upstream actually does (``Scheduler._try_schedule_encoder_inputs``):
    it only rolls a chunk back to the item's *offset*, and the guard is skipped
    once ``num_computed_tokens == start_pos``. So keeping an item whole also
    relies on the item fitting in one chunk -- which is exactly the invariant
    ``compute_mm_encoder_budget`` enforces by rejecting
    ``max_tokens_per_mm_item > max_num_batched_tokens``. This test uses such a
    legal config (image == one chunk) and a deliberately non-chunk-aligned
    offset, so the rollback is what has to keep the image intact.
    """
    chunk = 4 * _BLOCK_SIZE  # 64
    offset = 3 * _BLOCK_SIZE  # 48: block-aligned but NOT chunk-aligned
    img_len = chunk  # fits exactly one chunk (the legal-config invariant)
    isl = offset + img_len + _BLOCK_SIZE
    sched = _make_scheduler(chunk=chunk, max_num_seqs=1)
    _enable_encoder_cache(sched, cache_size=img_len, compute_budget=img_len)
    # The Gemma-4 path: never partially schedule an mm item.
    sched.scheduler_config.disable_chunked_mm_input = True
    sched.add_request(_make_mm_request("r0", isl, offset=offset, length=img_len))

    boundaries = _prefill_chunk_boundaries(sched, "r0", isl)

    assert (
        boundaries[-1] == isl
    ), f"boundaries {boundaries} must cover exactly {isl} tokens"
    assert offset in boundaries, (
        f"expected a chunk boundary exactly at the image offset {offset} (the "
        f"upstream roll-back), got boundaries={boundaries}"
    )
    for b in boundaries[:-1]:
        assert not (offset < b < offset + img_len), (
            f"chunk boundary at {b} falls strictly inside the image "
            f"[{offset}, {offset + img_len}) even though chunked mm input is "
            f"disabled; boundaries={boundaries}"
        )


@pytest.mark.push
@pytest.mark.cpu
def test_encoder_truncated_chunk_stays_block_aligned():
    """A chunk the encoder path shortens must still be block-aligned.

    ``_try_schedule_encoder_inputs`` truncates ``num_new_tokens`` to just before
    an mm item it cannot fit in the encoder cache/budget. That boundary is the
    image's ``offset``, which is not block-aligned in general. Leaving it
    unaligned makes the next chunk start mid-block, and then
    ``paged_fill_cache`` (writes from the start of a block) and the chunked SDPA
    read (offsets by ``chunk_start_idx == num_computed`` exactly) disagree.

    Here the encoder cache is too small to ever hold the image, so the scheduler
    can only advance over the leading text -- block-aligned, not to the raw
    offset.
    """
    chunk = 4 * _BLOCK_SIZE  # 64
    offset = _BLOCK_SIZE + 5  # deliberately NOT a multiple of the block size
    img_len = 2 * chunk  # bigger than the encoder cache below
    isl = offset + img_len + _BLOCK_SIZE
    sched = _make_scheduler(chunk=chunk, max_num_seqs=1)
    # Cache far too small for the image: can_allocate() always fails, so the
    # encoder path truncates every chunk back to `offset`.
    _enable_encoder_cache(sched, cache_size=_BLOCK_SIZE, compute_budget=_BLOCK_SIZE)
    sched.add_request(_make_mm_request("r0", isl, offset=offset, length=img_len))

    out = sched.schedule()
    n = out.num_scheduled_tokens.get("r0", 0)

    assert n > 0, "the leading text before the image should still be schedulable"
    assert n <= offset, f"scheduled {n} tokens must not run past the image at {offset}"
    assert n % _BLOCK_SIZE == 0, (
        f"chunk truncated by the encoder path is {n} tokens, which is not a "
        f"multiple of block_size {_BLOCK_SIZE}; the next chunk would start "
        "mid-block and corrupt the cached-prefix fill"
    )
