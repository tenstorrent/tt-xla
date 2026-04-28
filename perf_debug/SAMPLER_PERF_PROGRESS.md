# Sampler perf — running progress

Living tracker for non-greedy sampler perf work on branch `kmabee/vllm_demo_may1.perf_debug`. The snapshot of the *starting* state is in `SAMPLER_HOTSPOTS_20260427.md` (pre-improvements baseline, captured 2026-04-27). This doc tracks what's landed since and what's queued.

## Where we started (2026-04-27, pre-improvements)

`test_sampling_quality`, batch=2, `enable_trace=True`, `optimization_level=1`, old `apply_top_k_top_p` (single-core `probs.sort()` over 128K vocab):

| Model | greedy-device | greedy-cpu | **nongreedy-device** | nongreedy-cpu |
|---|---|---|---|---|
| Llama-3.2-1B | 77.90 | 61.62 | **5.81** | 60.09 |
| Llama-3.1-8B | 21.46 | 19.98 | **4.83** | 19.65 |

Bottleneck: `nongreedy-device` was 14× / 4× slower than any other path on 1B / 8B. Root cause: single-core `SortDeviceOperation` at ~30 ms/token. See `SAMPLER_HOTSPOTS_20260427.md` for the full hotspot table.

## Landed changes

### 1. Replace sort-based sampling with multi-core topk (tt-xla [#4334](https://github.com/tenstorrent/tt-xla/pull/4334))

Cherry-picked as `e4a1f124e`. Splits the 128K vocab into 4 chunks of ~32K, runs `torch.topk` per chunk to trigger multi-core `ttnn.topk` (~0.18 ms/chunk × 4) instead of single-core `ttnn.sort` (~30 ms). Replaces `apply_top_k_top_p` with `apply_top_k_top_p_fast`. New companion: composite `ttnn.gather` op (cherry-picked from `bf3360a6f`) used by the candidate-indices map-back step.

### 2. Add TopKOp to TTNNWorkarounds whitelist (tt-mlir [#8141](https://github.com/tenstorrent/tt-mlir/issues/8141), PR pending)

Locally committed in our tt-mlir submodule (`62a7af5c8` after rebase onto upstream merged gather). Without it, `opt_level=1` produced f32 typecasts on 3-of-4 chunked-topk indices outputs — silent garbage tokens. With it, all 4 chunks get correct si32 typecasts. Status: pinned locally, PR upstream pending.

### 3. tt::sampling fused custom op (cherry-picked as `aa73458bc`, default-on via TT_USE_TTNN_SAMPLING)

Replaces the multi-step softmax + Gumbel-max + gather pipeline in `apply_top_k_top_p_fast`'s tail with a single fused custom op `torch.ops.tt.sampling`. Pre-filter (chunked topk → 128 candidates) still runs; the fused op then does softmax + top-k + top-p + multinomial in one kernel call on the small candidate set. Polarity: `TT_USE_TTNN_SAMPLING` defaults to `"1"`, so the fused op is on unless explicitly disabled with `=0`.

**Trace-incompat caveat:** the new path's tail does `sampled_padded[:batch].view(-1)` where `batch = logits.shape[0]` is dynamic. Under torch.compile + XLA + metal trace, that slice forces a `from_device` round-trip, and the trace verifier rejects: *"All output tensors of trace function must be on device."* As a workaround, `_QUALITY_OPTS["enable_trace"]` auto-disables when `TT_USE_TTNN_SAMPLING != "0"`. Investigation notes for the trace-incompat are in [`TTNN_SAMPLING_TRACE_INCOMPAT.md`](TTNN_SAMPLING_TRACE_INCOMPAT.md).

### 4. Pad logits batch to 32 in chunked-topk pre-filter (`ae060a3be` cherry-picked)

Pads `logits` from `[batch, vocab]` to `[32, vocab]` at the top of `apply_top_k_top_p_fast`, then trims back to `[:batch]` after the chunked topk. Multi-core ttnn.topk is ~14× faster at batch=32 than batch=1 on hardware (1.08 ms vs 14.85 ms measured), so even paying the dummy-row cost is a net win on small batches. ttnn.sampling path also benefits because it calls `apply_top_k_top_p_fast` for its candidate-set pre-filter.

**Note:** this change was originally followed by a "disable pad-to-32 for ttnn.sampling" patch (`64a19144a`) due to "index corruption" at b>1. We **deliberately did NOT cherry-pick that disable** — testing showed coherent output on Llama-3.2-1B and Llama-3.1-8B at b=2, suggesting the corruption was resolved by intervening fixes (most likely tt-mlir #8141, the TopKOp workaround whitelist).

### 5. Skip `greedy_sample` + final `torch.where` when `all_random=True`

`Sampler.sample()` was unconditionally computing `greedy_sampled = self.greedy_sample(logits)` (an `argmax` over full vocab) and then using it in a final `torch.where(temp < EPS, greedy, random)`. When the batch is homogeneously non-greedy (`all_random=True`, i.e., every row's `temperature > 0`), the where always selects `random` and the greedy result is dead. Now gated on `if not sampling_metadata.all_random:` — the Python-level branch torch.compile specializes away, eliminating the ArgMax kernel + downstream Tilize entirely.

Verified end-to-end: `XLASupportedSamplingMetadata.all_random` mirrors `input_batch.all_random` (= `len(greedy_reqs) == 0` from `input_batch.py:761`). For `temperature=1.0` benchmark configs every request is bucketed `RANDOM` and the fast path fires. Mixed batches (some greedy + some random) still take the slow path correctly.

**Throughput-script (b=32) impact:** 3.04 → 1.85 ms/iter (−39%). Tracy: ArgMaxDeviceOperation 1.03 ms (34% of sampler) drops to 0; Tilize drops 269 → 96 µs as the argmax-output's tilize disappears with it; op count 50 → 44.

## Scoreboard — `nongreedy-device` at batch=2, opt=1

What every cherry-pick has cost or gained on the bottleneck path. Greedy-device shown as the no-sampler ceiling — the maximum `nongreedy-device` could realistically reach if sampler overhead vanished.

| State | trace | Llama-3.2-1B | Llama-3.1-8B |
|---|---|---|---|
| **Original baseline** (sort-based sampler, pre-cherry-picks) | True | 5.81  | 4.83  |
| chunked-topk + tt-mlir #8141 | True | 21.41 (3.69×) | 12.49 (2.59×) |
| ttnn.sampling fused op, no pad-32 | False | 30.21 (5.20×) | 14.72 (3.05×) |
| **ttnn.sampling + pad-32** (current) | False | **42.16 (7.26×)** | **17.06 (3.53×)** |
| nongreedy-cpu (soft target — what CPU sampling currently delivers) | False | 57.86 | 19.02 |
| greedy-device target (no-sampler ceiling) | False | 75.66 | 20.93 |

**Where the gap to the targets sits:**
- 1B: **42.16 / 57.86 → 73% of CPU-sampling parity** (soft target); **42.16 / 75.66 → 56% of greedy-device ceiling** (hard target).
- 8B: **17.06 / 19.02 → 90% of CPU-sampling parity** (essentially at parity); **17.06 / 20.93 → 82% of greedy-device ceiling.** Sampler is no longer the dominant cost on 8B.

**On the targets:** nongreedy-cpu is the *soft* target — production currently runs CPU sampling for non-greedy, so any device-side number ≥ that means the device-side path is at least as fast as what users see today. greedy-device is the *hard* ceiling — what nongreedy-device could reach if sampler overhead were eliminated entirely, since greedy uses an argmax fast-path with no sampler graph.

**Speedups all referenced to the original sort-path baseline.** Note the scoreboard's first three rows have different `trace` settings — comparing within each row is apples-to-apples; comparing across rows mixes trace ON/OFF effects. The 1B trace=True chunked-topk row (21.41) is what the trace-incompat fix would unblock for the ttnn.sampling+pad-32 row to become — likely 1B going past 42 and 8B past 17 once trace+ttnn.sampling work together. Crossing the soft target on 1B (currently 42.16 → soft 57.86) is the most likely deliverable from fixing the trace incompat.

## Where we are now (full per-config table, 2026-04-28)

`test_sampling_quality`, batch=2, `optimization_level=1`. The `_TTNN_SAMPLING_ACTIVE` workaround forces trace=False for ttnn.sampling, hence the right two columns:

| Model | Config | Pre-cherry-pick (sort, **trace=True**) | chunked-topk + tt-mlir (**trace=True**) | ttnn.sampling no pad-32 (**trace=False**) | **ttnn.sampling + pad-32 (current, trace=False)** |
|---|---|---|---|---|---|
| Llama-3.2-1B | greedy-device | 77.90 | 77.61 | 75.41 | 75.66 |
| Llama-3.2-1B | greedy-cpu | 61.62 | 61.76 | 59.60 | 58.56 |
| Llama-3.2-1B | **nongreedy-device** | **5.81** | **21.41** | **30.21** | **42.16** |
| Llama-3.2-1B | nongreedy-cpu | 60.09 | 60.47 | 57.97 | 57.86 |
| Llama-3.1-8B | greedy-device | 21.46 | 21.13 | 21.08 | 20.93 |
| Llama-3.1-8B | greedy-cpu | 19.98 | 19.57 | 19.22 | 19.28 |
| Llama-3.1-8B | **nongreedy-device** | **4.83** | **12.49** | **14.72** | **17.06** |
| Llama-3.1-8B | nongreedy-cpu | 19.65 | 19.32 | 18.87 | 19.02 |

Run dir: `sampling_quality_batch_2_tt_sampling_no_trace_opt_level_1_20260428_141842/`. Outputs verified coherent across all eight configs (proper fox/river stories on the prompt). The non-bottleneck paths (greedy-device / greedy-cpu / nongreedy-cpu) at trace=False sit 1–4% below their trace=True numbers, purely from losing trace globally for the test config. Those would recover if the ttnn.sampling+trace incompat is fixed. As expected, **pad-to-32 only moves the nongreedy-device path** (the only one that goes through `apply_top_k_top_p_fast`'s chunked-topk pre-filter): +40% on 1B (30.21 → 42.16), +16% on 8B (14.72 → 17.06). All other paths within ±2% of the prior state — confirming the change is well-isolated.

## Sampler-only op breakdown at b=2 (post-fix, e2e tracy capture)

Captured 2026-04-28 via tracy on `test_sampling_quality[llama3.2-1b-b2-nongreedy-device]` with `enable_trace=False` (trace must be off for tracy device profiling — tracy tries to read perf counters via event sync inside trace, which tt-metal forbids). Signpost added in `model_runner.py:1400` (host-side, just before the compiled `sample_from_logits` call) to slice by decode step.

The signpost fires BEFORE each sampler call, so most iter ranges include [sampler + post-host + model forward of next step] (~979 ops, ~72.5 ms). The **last iter is special** — no following forward, so it captures the sampler call alone:

| Iter | n_ops | total_ms | What's in this range |
|---|---|---|---|
| 7–10 (representative) | 979 | ~72.6 | sampler N + host + forward N+1 |
| 11 (last) | **125** | **53.8** | **sampler-only** |

**Sampler-only hotspot at b=2 (iter 11):**

| Op | sum_ms | count | % |
|---|---|---|---|
| FillPadDeviceOperation | 26.73 | 14 | 49.7% |
| TopKDeviceOperation | 9.74 | 4 | 18.1% |
| PadDeviceOperation | 9.00 | 4 | 16.7% |
| SoftmaxDeviceOperation | 2.54 | 3 | 4.7% |
| BinaryNgDeviceOperation | 2.17 | 20 | 4.0% |
| ReduceDeviceOperation | 1.76 | 2 | 3.3% |

**Read:**
- Sampler at b=2 ≈ 53.8 ms; throughput script at b=1 was ≈ 55 ms with the same chunked-topk graph. **Sampler is roughly batch-invariant** for b=1→b=2 — confirms the bottleneck is per-decode op cost, not batch scaling.
- Inferred forward at b=2 ≈ 72.5 - 53.8 = 18.7 ms (consistent with Llama-3.2-1B at b=2 with this layer count).
- **FillPad is now the dominant op at 50% of sampler time.** Largest known unaddressed target after the chunked-topk + tt-mlir #8141 fix. 14 calls × 1.9 ms each.

**Trace caveat:** tracy capture above is at `enable_trace=False` (required for tracy `-p`). The 21.41 tok/s e2e number was measured at `enable_trace=True`. So tracy ms/iter and e2e tok/s are not directly comparable — tracy gives op-level breakdown without trace, e2e gives wall-clock with trace. Use tracy to track which ops dominate; use e2e tok/s to track end-user throughput.

## Throughput-script tracy progression (2026-04-28)

Op-level breakdown via `perf_debug.test_sampler_throughput` with `TRACY_PROFILING_ACTIVE=1` and `--batch {1, 2, 32}`. Measures **the sampler @torch.compile graph in isolation** (no model_forward, no host overhead). Different from the e2e tracy section above — that one captures the sampler+forward inside a real test_sampling_quality run.

| State | b=1 | b=2 | b=32 | Notes |
|---|---|---|---|---|
| ttnn.sampling + pad-32 (current production) | 20.18 ms | 19.65 ms | 3.04 ms | Slice 45% + FillPad 45% at b<32 |
| + skip greedy when `all_random` (this branch) | — | — | **1.85 ms** | ArgMax 0; TopK 40% / Typecast 27% now dominate |

**Key reading:**
- **b=1 ≈ b=2 ≈ 20 ms.** Per-iter sampler time is essentially batch-invariant for b<32 — the work is fixed-shape, not per-row. Going from b=2 → b=32 collapses the FillPad/Slice batch-pad-to-32 overhead (~17 ms/iter) into no-ops.
- **At b=32, the dominant batch-padding cost vanishes.** FillPad goes from 9.2 ms (45%) → 0; Slice from 9.2 ms (7 calls) → 0.19 ms (4 calls — the 3 batch-padding slices gone, 4 vocab-chunk slices remain). The fact that `b=1` and `b=2` are nearly identical confirms this is fixed-cost shape work, not batch-amplified compute.
- **The fused `SamplingDeviceOperation` itself is 40 µs / 1–2% in every configuration.** The kernel is great; we're entirely paying for shape work around it.

**Sampler-only hotspot at b=32 + greedy-skip (post-fix, throughput script):**

| Op | sum_µs | count | % | What is it |
|---|---|---|---|---|
| TopKDeviceOperation | 740 | 4 | 40% | chunked-topk pre-filter (4 vocab chunks) |
| TypecastDeviceOperation | 492 | 17 | 27% | dtype shuffling (suspicious — 4× per topk avg) |
| PadDeviceOperation | 226 | 4 | 12% | tile-alignment (≠ FillPad) |
| SliceDeviceOperation | 174 | 4 | 9% | 4× vocab chunk slices |
| TilizeDeviceOperation | 96 | 1 | 5% | layout |
| SamplingDeviceOperation | 40 | 1 | 2% | the fused kernel |

Total: 1.85 ms / 44 ops. **22 K tok/s sampler-only at b=32.**

**Implication for production batch sizes:** at typical inference b=2, we're paying ~17 ms/token in pure batch-pad-to-32 overhead that doesn't exist at b=32. For the e2e Llama numbers, the sampler-side savings from increasing actual batch size are roughly the difference (20 → 3 ms ≈ 17 ms/iter saved per token). Need a 1B/8B benchmark sweep at b={1,2,16,32} to see how that translates to e2e tok/s and whether sampler is still on the critical path at higher batch sizes.

## How to recapture this for future changes

```bash
# 1. Set _QUALITY_OPTS["enable_trace"] = False temporarily in
#    tests/benchmark/test_vllm_benchmarks.py (tracy + trace incompatible)
# 2. Reset device
tt-smi -r 0 && sleep 1
# 3. Capture (max-output-tokens 6 keeps the .tracy and CSV manageable)
TRACY_PROFILING_ACTIVE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python -m tracy -p -r \
    -o tracy_e2e_<descriptor> \
    -m pytest -svv \
    --max-output-tokens 6 \
    "tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[llama3.2-1b-b2-nongreedy-device]" \
    |& tee tracy_e2e_<descriptor>.log
# 4. Analyze — focus on the LAST iter (sampler-only)
python perf_debug/analyze_tracy.py tracy_e2e_<descriptor>/reports/*/ops_perf_results_*.csv
# 5. Restore enable_trace=True afterward for production benchmarking.
```

## Up next

1. **Cherry-pick `e8375bd15` and `87425d2b4`** — the broadcast-bug fix and the batch-variable restore for `apply_top_k_top_p_fast`'s pad-to-32 logic. Don't need them for the basic happy path (verified at b=2 with k/p both None for the ttnn.sampling caller), but they harden the chunked-topk-only path's edge cases (k/p tensors present) which we still want to support.
2. **Fix the ttnn.sampling + metal-trace incompatibility** so all paths can have `enable_trace=True` again. Investigation notes in [`TTNN_SAMPLING_TRACE_INCOMPAT.md`](TTNN_SAMPLING_TRACE_INCOMPAT.md). Once fixed, expect 1B `nongreedy-device` to push past 42 and 8B past 17 (these are the trace-OFF numbers).
3. **Op-level tracy breakdown of ttnn.sampling + pad-32 at b=2** to see what dominates now. FillPad at 50% was the post-chunked-topk picture; pad-to-32 may have shifted that distribution. Tells us whether the next sampler-side win is fusing more small ops, or shrinking the FillPad shapes that survive.

## Validation commands

```bash
# Full sampling-quality sweep (all batches, all configs)
bash run_sampling_quality.sh

# Per-config diff vs starting baseline
for cfg in greedy-device greedy-cpu nongreedy-device nongreedy-cpu; do
    new=$(grep "Sample per second" $LOG_DIR/llama3.1-8b-b2-$cfg.log | awk '{print $NF}')
    old=$(grep "Sample per second" sampling_quality_baseline_batch2_20260427_044112/llama3.1-8b-b2-$cfg.log | awk '{print $NF}')
    echo "$cfg: $old → $new"
done

# Tracy-level hotspot diff
python perf_debug/analyze_tracy.py BASELINE.csv --vs AFTER.csv
```
