# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Continuous batching on device (#5813).

Wave B is submitted only once wave A is decoding, so its prefill joins a live
decode batch. Every output must match a reference pass where each prompt runs
alone: a request whose KV was clobbered by the joiner, or a joiner reading a
slot's stale KV, diverges from that reference while still reading as fluent
text, so coherence checks alone would miss it.

The scheduler side (prefill first, then decode for old and new together) is
covered on CPU by unit_tests/test_ascend_scheduler_batching.py.
"""
import asyncio

import pytest
from conftest import GROUNDED_BATCH_CHECKS, check_host_memory
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind

MODEL = "Qwen/Qwen3-0.6B"
MAX_NUM_SEQS = 4
MAX_MODEL_LEN = 64
# Wave A must outlive B's arrival, so it generates for longer.
MAX_TOKENS_A = 24
MAX_TOKENS_B = 12
DECODE_TOKENS_BEFORE_ARRIVAL = 4
# The first request pays engine warmup and graph compilation.
ARRIVAL_TIMEOUT_S = 900

WAVE_A = GROUNDED_BATCH_CHECKS[:2]
WAVE_B = GROUNDED_BATCH_CHECKS[2:]
assert len(WAVE_A) + len(WAVE_B) <= MAX_NUM_SEQS, "waves must fit in one batch"


def _make_engine(additional_config: dict) -> AsyncLLMEngine:
    args = AsyncEngineArgs(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * MAX_NUM_SEQS,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.002,
        # Off so the reference pass doesn't seed the cache for the staggered one
        # and wave B really prefills on arrival. DP disables it anyway.
        enable_prefix_caching=False,
        additional_config={"min_context_len": 32, **additional_config},
    )
    return AsyncLLMEngine.from_engine_args(args)


def _sampling_params(max_tokens: int) -> SamplingParams:
    # Greedy and fixed length, so the reference is an exact oracle.
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )


async def _stream(engine, prompt, request_id, max_tokens, on_progress=None):
    """Run one request to completion, returning (token_ids, text)."""
    token_ids, text = [], ""
    async for out in engine.generate(
        prompt, _sampling_params(max_tokens), request_id=request_id
    ):
        token_ids.extend(out.outputs[0].token_ids)
        text += out.outputs[0].text
        if on_progress is not None:
            on_progress(len(token_ids))
    return token_ids, text


async def _reference(engine):
    """Each prompt alone, one at a time: no other request shares the batch."""
    refs = []
    for i, (prompt, _) in enumerate(WAVE_A + WAVE_B):
        max_tokens = MAX_TOKENS_A if i < len(WAVE_A) else MAX_TOKENS_B
        refs.append(await _stream(engine, prompt, f"ref-{i}", max_tokens))
    return refs


async def _staggered(engine):
    """Wave A decodes, wave B joins mid-flight.

    Also returns how many tokens each wave-A request had produced by then.
    """
    decoded = {}
    gate = asyncio.Event()

    def progress(i, n):
        decoded[i] = n
        if all(
            decoded.get(k, 0) >= DECODE_TOKENS_BEFORE_ARRIVAL
            for k in range(len(WAVE_A))
        ):
            gate.set()

    a_tasks = [
        asyncio.create_task(
            _stream(
                engine,
                prompt,
                f"a-{i}",
                MAX_TOKENS_A,
                on_progress=lambda n, i=i: progress(i, n),
            )
        )
        for i, (prompt, _) in enumerate(WAVE_A)
    ]

    gate_task = asyncio.create_task(gate.wait())
    done, _pending = await asyncio.wait(
        {gate_task, *a_tasks},
        timeout=ARRIVAL_TIMEOUT_S,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not gate_task.done():
        gate_task.cancel()
        for task in a_tasks:
            task.cancel()
        for task in done:  # report a wave-A failure instead of the timeout
            task.result()
        raise AssertionError(
            f"wave A did not decode {DECODE_TOKENS_BEFORE_ARRIVAL} tokens within "
            f"{ARRIVAL_TIMEOUT_S}s; decoded={decoded}"
        )
    decoded_at_arrival = dict(decoded)

    b_tasks = [
        asyncio.create_task(_stream(engine, prompt, f"b-{i}", MAX_TOKENS_B))
        for i, (prompt, _) in enumerate(WAVE_B)
    ]
    results = await asyncio.gather(*a_tasks, *b_tasks)
    return list(results), decoded_at_arrival


async def _collect(additional_config: dict):
    engine = _make_engine(additional_config)
    try:
        refs = await _reference(engine)
        results, decoded_at_arrival = await _staggered(engine)
        rss_gb = check_host_memory(MODEL)
    finally:
        shutdown = getattr(engine, "shutdown", None)  # API varies by version
        if callable(shutdown):
            shutdown()
    return refs, results, decoded_at_arrival, rss_gb


def _assert_overlapped(decoded_at_arrival, results):
    """Wave B really arrived mid-decode, and wave A really kept decoding."""
    for i, (prompt, _) in enumerate(WAVE_A):
        at_arrival = decoded_at_arrival.get(i, 0)
        final = len(results[i][0])
        assert at_arrival >= DECODE_TOKENS_BEFORE_ARRIVAL, (
            f"wave A req {i} ({prompt!r}) had only {at_arrival} tokens when wave B "
            "was submitted -- it was not decoding yet"
        )
        assert final > at_arrival, (
            f"wave A req {i} ({prompt!r}) finished ({final} tokens) before wave B "
            "arrived -- no decode overlap, so nothing was continuously batched"
        )


def _assert_matches_reference(refs, results):
    """Batched output must equal the isolated reference, token for token."""
    checks = WAVE_A + WAVE_B
    failures = []
    for i, ((prompt, expected), (ref_ids, ref_text), (ids, text)) in enumerate(
        zip(checks, refs, results)
    ):
        label = "A" if i < len(WAVE_A) else "B"
        print(f"  wave {label} req {i}: {prompt!r} -> {text!r} (ref {ref_text!r})")
        if expected.lower() not in ref_text.lower():
            failures.append(
                f"req {i} ({prompt!r}): reference itself is wrong, expected "
                f"{expected!r}, got {ref_text!r} -- not a batching failure"
            )
            continue
        if ids != ref_ids:
            first_diff = next(
                (k for k in range(min(len(ids), len(ref_ids))) if ids[k] != ref_ids[k]),
                min(len(ids), len(ref_ids)),
            )
            failures.append(
                f"req {i} ({prompt!r}) diverged from its isolated reference at "
                f"token {first_diff}: got {text!r}, expected {ref_text!r}"
            )
    assert not failures, "continuous batching corrupted output:\n" + "\n".join(failures)


def _assert_grounded(results):
    """Batched output must contain expected answers (tolerant of #5520 near-tie drift).

    Cross-chip reductions in TP are not bit-reproducible, so exact token matching
    is not guaranteed. This checks that outputs are at least grounded in their
    expected answers, catching corruption (garbage, repeat-loops, wrong slot) while
    tolerating benign fp drift from non-deterministic reductions."""
    checks = WAVE_A + WAVE_B
    failures = []
    for i, ((prompt, expected), (ids, text)) in enumerate(zip(checks, results)):
        label = "A" if i < len(WAVE_A) else "B"
        print(f"  wave {label} req {i}: {prompt!r} -> {text!r}")
        if expected.lower() not in text.lower():
            failures.append(
                f"req {i} ({prompt!r}): expected {expected!r}, got {text!r}"
            )
    assert not failures, "continuous batching corrupted output:\n" + "\n".join(failures)


def _run(additional_config: dict):
    refs, results, decoded_at_arrival, _ = asyncio.run(_collect(additional_config))
    print(f"\n===== continuous batching ({additional_config}) =====")
    print(f"  wave A tokens decoded when wave B arrived: {decoded_at_arrival}")
    _assert_overlapped(decoded_at_arrival, results)
    # Single device: exact tokens (no cross-chip reductions). TP/DP+TP: grounded
    # answers only, since cross-chip lm_head reductions are not bit-reproducible
    # (#5520) and near-tie argmax can flip between runs.
    is_single_device = not additional_config.get(
        "enable_tensor_parallel"
    ) and not additional_config.get("enable_data_parallel")
    if is_single_device:
        _assert_matches_reference(refs, results)
    else:
        _assert_grounded(results)


@pytest.mark.nightly
@pytest.mark.single_device
def test_continuous_batching_single_device():
    """New prompts joining a live decode batch on one device."""
    _run({})


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
def test_continuous_batching_tensor_parallel_llmbox():
    """Same with weights sharded across the TP axis. cpu_sampling keeps the
    2D-mesh device sampler (#4440) out of the picture, as the other wide-batch
    TP tests do."""
    _run({"enable_tensor_parallel": True, "cpu_sampling": True})


@pytest.mark.nightly
@pytest.mark.data_parallel
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
def test_continuous_batching_data_tensor_parallel_llmbox():
    """Same on a (dp, tp) mesh, where the joiner lands in a replica row that is
    already decoding, so a mis-offset per-replica write shows up as one row
    diverging from its reference."""
    _run(
        {
            "enable_data_parallel": True,
            "enable_tensor_parallel": True,
            "cpu_sampling": True,
        }
    )
