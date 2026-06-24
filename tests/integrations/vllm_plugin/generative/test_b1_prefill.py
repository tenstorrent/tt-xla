# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone regression for the b1-prefill optimization (tt-xla #5281).

A single `min_num_seqs=1` engine compiles BOTH a b1 (`[1, n]`) and a b32
(`[max_num_seqs, n]`) prefill graph; the scheduler picks per step by how many
requests are being prefilled. This test fires the same prefill at four request
shapes and prints a TTFT table comparing them:

  single   : 1 request               -> b1   (the win: lone request skips the 32-row graph)
  staggered: N reqs, ramped           -> realistic arrivals
  burst    : N(<=threshold) reqs, sim -> b1 served serially via prefill_batch_threshold
  batch    : M(>threshold) reqs       -> b32, correct to batch here (stable b32 anchor)

One test, test_b1_prefill_ttft, runs all four shapes once and asserts the two
wins b1-prefill delivers:
  1. single (b1) << batch (b32) -- the lone-request win (min_num_seqs).
  2. burst  (b1) << batch (b32) -- the small-burst win (prefill_batch_threshold).

Single layer (~2 min): the b1/b32 graph selection is depth-independent, so this
gates the feature without the full-model compile. Full-model TTFT figures live
in the PR discussion.
"""
import asyncio
import random
import statistics
import time
from dataclasses import dataclass

import pytest
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind

ISL = 1024  # b1 win grows with ISL
SMALL = 8  # burst size (<= threshold -> the gap target)
LARGE = 24  # batch size (> threshold -> stable b32 anchor)
RAMP_MS = 100  # staggered inter-arrival


@dataclass
class Stats:
    label: str
    ttfts_ms: list

    @property
    def median(self):
        return statistics.median(self.ttfts_ms)

    @property
    def p90(self):
        xs = sorted(self.ttfts_ms)
        return xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1) + 0.5))]

    def row(self):
        xs = self.ttfts_ms
        return (
            f"{self.label:<10} n={len(xs):<3} "
            f"min={min(xs):8.0f}  median={self.median:8.0f}  "
            f"p90={self.p90:8.0f}  max={max(xs):8.0f}  (ms TTFT)"
        )


def _distinct_prompt(seed: int):
    # Unique token sequence per request so prefix caching can't fake TTFT.
    rng = random.Random(f"b1test:{seed}:{ISL}")
    return TokensPrompt(
        prompt_token_ids=[1000 + rng.randrange(20000) for _ in range(ISL)]
    )


def _make_engine() -> AsyncLLMEngine:
    additional_config = {
        "enable_const_eval": True,
        "min_context_len": 128,
        "experimental_weight_dtype": "bfp_bf8",
        "experimental_kv_cache_dtype": "bfp_bf8",
        "cpu_sampling": False,
        "optimization_level": 1,
        "enable_trace": False,  # prefill-focused; trace is a decode opt
        "fp32_dest_acc_en": False,
        "num_hidden_layers": 1,  # single layer: nightly ~2 min; b1/b32 selection is depth-independent
        "min_num_seqs": 1,  # b1-prefill: also compile the [1, n] graph
        "prefill_batch_threshold": 16,  # route a burst of <=16 pending prefills to b1
    }
    args = AsyncEngineArgs(
        model="Qwen/Qwen3-8B",
        # budget must cover max_model_len * max_num_seqs; small max_model_len keeps it cheap
        max_model_len=2048,
        max_num_batched_tokens=2048 * 32,
        max_num_seqs=32,
        gpu_memory_utilization=0.3,
        additional_config=additional_config,
    )
    return AsyncLLMEngine.from_engine_args(args)


async def _one(engine, seed: int, delay_s: float) -> float:
    if delay_s:
        await asyncio.sleep(delay_s)
    sp = SamplingParams(
        max_tokens=1,
        ignore_eos=True,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )
    t0 = time.monotonic()
    async for _ in engine.generate(_distinct_prompt(seed), sp, request_id=f"r{seed}"):
        return (time.monotonic() - t0) * 1e3  # first (only) token = TTFT
    return float("nan")


async def _scenario(engine, label, n, ramp_s, seed0) -> Stats:
    ttfts = await asyncio.gather(
        *[_one(engine, seed0 + i, ramp_s * i) for i in range(n)]
    )
    return Stats(label, list(ttfts))


async def _sequential(engine, label, n, seed0) -> Stats:
    # n lone requests, each awaited before the next -> always num_reqs=1 (b1).
    # Median over a few is stable; one sample is noisy.
    ttfts = [await _one(engine, seed0 + i, 0) for i in range(n)]
    return Stats(label, ttfts)


async def _collect():
    engine = _make_engine()
    try:
        # Warm up both graphs (b1 via single, b32 via a batch); discard.
        await _scenario(engine, "warm_b1", 1, 0, 0)
        await _scenario(engine, "warm_b32", LARGE, 0, 100)
        single = await _sequential(engine, "single", 4, 1000)
        staggered = await _scenario(engine, "staggered", SMALL, RAMP_MS / 1e3, 2000)
        burst = await _scenario(engine, "burst", SMALL, 0, 3000)
        batch = await _scenario(engine, "batch", LARGE, 0, 4000)
    finally:
        sd = getattr(engine, "shutdown", None)  # shutdown API varies by version
        if callable(sd):
            sd()
    return {s.label: s for s in (single, staggered, burst, batch)}


def _print_table(m):
    print(f"\n===== b1-prefill TTFT summary (Qwen3-8B 1L, ISL {ISL}) =====")
    for k in ("single", "staggered", "burst", "batch"):
        print("  " + m[k].row())
    print(
        f"  (single=b1; burst={SMALL}<=threshold -> b1; "
        f"batch={LARGE}>threshold -> b32)"
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_b1_prefill_ttft():
    """Fire one min_num_seqs=1 engine at four request shapes, print the TTFT table,
    and assert the two wins b1-prefill delivers:

      1. b1 lone-request win  -- single (b1) << batch (b32); fails without
         min_num_seqs (a lone request would also pad to the 32-row graph).
      2. threshold burst win  -- burst (b1) << batch (b32); a <=threshold burst is
         served serially on b1 instead of one slow b32 batch (prefill_batch_threshold).
    """
    m = asyncio.run(_collect())
    _print_table(m)
    assert m["single"].median < m["batch"].median / 3, (
        f"b1 lone-request win: single (b1) {m['single'].median:.0f}ms not << batch "
        f"(b32) {m['batch'].median:.0f}ms -- is min_num_seqs/b1 active?"
    )
    assert m["burst"].median < m["batch"].median / 2, (
        f"threshold burst win: burst {m['burst'].median:.0f}ms ~ b32 batch "
        f"{m['batch'].median:.0f}ms -- is prefill_batch_threshold active?"
    )
