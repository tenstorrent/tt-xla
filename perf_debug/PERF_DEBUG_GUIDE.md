# tt-xla / vLLM perf debug guide

Practical knowledge transfer for performance debugging on tt-xla, focused on
the vLLM plugin path (Llama / OPT / etc). Companion doc:
[`SAMPLER_TESTING_PLAN.md`](./SAMPLER_TESTING_PLAN.md) for sampler-specific
testing, and `RoPE_OPT1_GARBAGE_DEBUG_PLAN.md` for the RoPE fusion bug
write-up that motivated some of these techniques.

---

## 0. Environment basics

- All builds and runs happen **inside the dev container** (silicon-attached).
  Bare-metal shells have file access but not toolchain/runtime.
- Python venv: `source venv/activate` from `/localdev/kmabee/tt-xla`.
- Always `tt-smi -r` between runs that put the device into a bad state
  (failed test, killed process, etc). Skipping this causes mysterious hangs
  on the next run.
- vLLM `EngineCore` runs in a subprocess that *holds the device*. If you
  try to run another `torch.compile(backend="tt")` while an LLM is alive,
  it will hang. See conftest at
  `tests/integrations/vllm_plugin/sampling/conftest.py` for the
  cache-flush dance.
- `pytest_runtest_teardown` flushes the cached LLM only between modules.
  Within a module, fixtures are shared.

### Useful env vars

| Var | Effect |
|---|---|
| `TTXLA_LOGGER_LEVEL=DEBUG` | Dumps full MLIR IR (StableHLO + TTIR + TTNN) at every pipeline stage. Verbose. |
| `TTMLIR_ENABLE_PERF_TRACE` | Enable performance tracing in tt-mlir build (only if explicitly built with it) |
| `TT_USE_TTNN_SAMPLING=1` | Routes non-greedy sampling through the `tt::sampling` custom op (vs. CPU fallback) |
| `VLLM_ENABLE_V1_MULTIPROCESSING=0` | Disables v1 engine multiprocessing — needed for tracy to capture in-process |
| `VLLM_ENABLE_TRACE` | Toggles metal trace path (only useful when `TTRotaryEmbedding` override is active and configured to support trace) |

---

## 1. Per-token throughput measurement (the headline number)

### Quick benchmark via vLLM-side test

`tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison` and
`test_vllm_trace_benchmark` produce sample/sec readings.

Example invocation (Llama-3.1-8B greedy, batch 1):

```bash
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" \
    --max-output-tokens 32 \
    |& tee 8b_greedy.log

grep "Sample per second" 8b_greedy.log
```

The user's `sampling_quality_*` directories (e.g.
`sampling_quality_baseline_20260427_005244/`) capture per-model per-mode
runs — useful as a fixed comparison baseline across changes.

### Perf reference points (Apr 2026)

| Config | Llama-3.1-8B greedy | Llama-3.2-1B greedy |
|---|---|---|
| TTRotaryEmbedding ON + RoPE fusion blocked (correct) | 13.9 tok/s | 24.5 tok/s |
| TTRotaryEmbedding OFF (native vLLM RoPE), fusion blocked by `isPackedCosSinPair` | 14.8 tok/s | 29.2 tok/s |
| TTRotaryEmbedding ON + fusion fires (BUGGY but fast) | 17.6 tok/s | 35.5 tok/s |
| Pre-`e60ba14a8` (no TTRotaryEmbedding, fusion behavior depended on tt-mlir version) | ~18 tok/s | ~35 tok/s |

Use these as sanity checks. ±0.5 tok/s is normal noise; >5% drops warrant investigation.

---

## 2. Tracy profiling (op-level timing)

Tracy gives you a flame-graph-style view of every op dispatched to the
device, plus host gaps and sync latency.

### Capture

```bash
tt-smi -r 0
sleep 1

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python -m tracy -p -r --sync-host-device \
    -o tracy_my_run \
    pytest -svv --max-output-tokens 3 \
    "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" \
    |& tee tracy_my_run.log
```

Notes:
- `-r` = report mode (writes CSV + reports)
- `--sync-host-device` = guarantees host waits for each op (needed for
  accurate per-op timing; otherwise ops are queued and timings overlap)
- `-o tracy_my_run` = output directory; produces
  `tracy_my_run/reports/*/ops_perf_results_*.csv`
- `--max-output-tokens 3` = capture 3 decode tokens (skip JIT warmup; first
  decode is much slower than steady-state)
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` = needed so tracy attaches to the
  same process as vLLM EngineCore

Reference script: `run_tracy_dispatch_profile.sh` runs the canonical
greedy-vs-nongreedy comparison.

### Analyze

The user's `perf_debug/analyze_tracy.py` parses a tracy CSV and breaks
down per-token time into forward / sampling / host gap categories:

```bash
python perf_debug/analyze_tracy.py \
    tracy_my_run/reports/*/ops_perf_results_*.csv
```

It classifies known ops into `FORWARD_OPS` (matmul, sdpa, layernorm,
embedding, paged-cache-update, NLP heads concat) vs. `SAMPLING_OPS` (sort,
softmax, argmax, reduce, scatter, etc) and reports a breakdown.

`perf_debug/annotate_tracy_csv.py` adds derived columns to the CSV (e.g.
op category, normalized timings) for downstream analysis in Excel/pandas.

### Open the trace visually (optional)

```bash
tracy-profiler tracy_my_run/tracy_my_run.tracy
```

Useful for visually spotting gaps between ops (host overhead).

### Things to look for in tracy

- **Gaps between ops** = host overhead (tensor transfer, dispatch, Python
  bookkeeping). If gaps dominate, the bottleneck is host-side, not the
  device kernel.
- **Long tail ops** — sort the CSV by duration descending. Often a few ops
  dominate (matmul, sdpa, sort).
- **Sampling vs. forward time split** — sampling can take 30%+ of decode
  time at low batch sizes. See `perf_debug/issue_3940_sampling_perf.md` for
  the deep dive.
- **Trace mode**: with `enable_trace=True`, the dispatch-overhead gaps
  disappear because the entire program is replayed from a recorded trace.
  Useful 1.4-2.5x speedup at low batch sizes.

---

## 3. MLIR IR inspection

### Capture full IR pipeline

```bash
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv <test> 2>&1 | tee debug.log
```

The log will contain MLIR modules at every pipeline stage:
`shlo` → `shlo_compiler` → `ttir` → `ttnn`.

### Extract specific stages

User's script: `/localdev/kmabee/scripts/extract_mlir_graphs.py`

```bash
python3 /localdev/kmabee/scripts/extract_mlir_graphs.py debug.log --type ttnn --subdir my_run
```

Produces `/tmp/my_run/graph_N_ttnn.mlir` per module. Llama-3.2-1B typically
produces ~99 modules over a full pytest run (one per torch.compile
invocation including profile run, KV cache setup, multiple decode shapes).
**Graph 3 is usually the main decode graph** for Llama.

### Useful greps to characterize the graph

```bash
# Is RoPE fusion firing?
grep -c "ttnn.rotary_embedding" /tmp/my_run/graph_3_ttnn.mlir

# Are SDPA fusions firing?
grep -c "ttnn.scaled_dot_product_attention\|ttnn.transformer.scaled_dot_product_attention_decode" /tmp/my_run/graph_3_ttnn.mlir

# Number of matmul ops (rough proxy for decode complexity):
grep -c "ttnn.matmul" /tmp/my_run/graph_3_ttnn.mlir

# Total ops (use the script's summary table column)
```

### Interpreting fusion-firing vs. -blocked op count deltas

A blocked fusion replaces a single fused op with N elementwise ops. For
Llama-3.2-1B, expected per-decode RoPE op delta:

- Fused: 32 sites (16 layers × 2 RoPE for Q and K) × 1 op each = **32 ops**
- Unfused: 32 sites × ~6.5 ops (chunk + 4× mul + sub + add + concat + slice
  setup) = **~208 ops**

So if you see graph_3 going from 971 → 1179 ops (a Δ of +208), that's
exactly 32 RoPE fusions getting blocked. This is how we confirmed the
dim -2 fix's behavior in the conversation that produced this doc.

---

## 4. PCC correctness sanity

`tests/benchmark/test_llms.py` runs the model on CPU (HF transformers)
and TT device, computes PCC of prefill and first-decode logits. **Use this
to confirm a perf change doesn't break correctness:**

```bash
pytest -svv "tests/benchmark/test_llms.py::test_llama_3_2_1b" --optimization-level=1
```

Healthy output:

```
Prefill PCC verification passed with PCC=0.998251
First decode PCC verification passed with PCC=0.998394
PASSED
```

PCC < 0.99 is a red flag — there's a numerical regression somewhere.

**Caveat**: this test uses HF transformers, not the vLLM compile path.
Bugs that only surface in the vLLM-specific code (e.g. `TTRotaryEmbedding`
in `overrides.py`) won't be caught by this benchmark. For vLLM-path
correctness, use end-to-end coherence/output tests at
`tests/integrations/vllm_plugin/sampling/test_sampling_params.py::test_output_coherence_nongreedy`
and `tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_llama3_3b_generation_opt_level_1`.

---

## 5. Common pitfalls

### 5.1 cmake submodule reset (silent fix-loss)

If you cherry-pick a commit into the **embedded tt-mlir submodule** at
`/localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir`, then run
`cmake --build build`, **cmake will reset the submodule to the pinned
commit** (from `third_party/CMakeLists.txt`) and your cherry-pick is
silently lost.

To prevent: **stage the change but don't commit it** — staged-only
changes survive the submodule checkout. Or edit
`third_party/CMakeLists.txt` to point at a different tt-mlir SHA that
includes your fix.

### 5.2 cmake stale cache

If you previously configured tt-mlir with `TTMLIR_ENABLE_RUNTIME=OFF`,
the `else` branch in tt-mlir's CMakeLists.txt writes
`TT_RUNTIME_ENABLE_TTNN=OFF` etc. via `set(... CACHE BOOL ... FORCE)`.
Re-configuring with `TTMLIR_ENABLE_RUNTIME=ON` doesn't override these
because `option()` respects existing cache values. Symptoms:

```
CMake Error at runtime/test/CMakeLists.txt:2 (message):
  Runtime tests require at least one backend runtime to be enabled
```

Fix: nuke `build/` or pass explicit `-DTT_RUNTIME_ENABLE_TTNN=ON`.

### 5.3 Lit tests `UNSUPPORTED` without opmodel

The `// REQUIRES: opmodel` directive in tt-mlir lit tests means they're
silently skipped if `TTMLIR_ENABLE_OPMODEL=OFF` in the build. Symptom:

```
UNSUPPORTED: TTMLIR :: ttmlir/Dialect/TTNN/optimizer/.../rope.mlir
```

Fix: rebuild with `-DTTMLIR_ENABLE_OPMODEL=ON`. Note: enabling opmodel
pulls in tt-metal headers and increases build time significantly.

### 5.4 Lit tests need a system_desc

Lit tests use `%system_desc_path%` substitution. Generated by:

```bash
cd /localdev/kmabee/tt-mlir
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
```

`ttrt query` requires silicon access **once**; after the file exists, the
test just reads it.

If `ttrt` itself is missing (`ModuleNotFoundError: ttrt.runtime._ttmlir_runtime`),
your build doesn't have `TTMLIR_ENABLE_RUNTIME=ON`.

### 5.5 Stochasticity in tests

Non-greedy (temperature > 0) sampling is stochastic. A test passing/failing
*once* doesn't tell you much. For perf, tracy reports averages so single
runs are usually fine; for correctness, run 2-3 times before concluding.
Greedy (temperature=0) is deterministic and what you want for clean
regression checks.

### 5.6 Test_output_coherence heuristic limits

`test_output_coherence_nongreedy` checks for non-Latin scripts, n-gram
repetition, and ASCII ratio. It catches *egregious* corruption (verbatim
loops, foreign chars) but **misses subtle "looks-fluent-but-broken"**
outputs. See `heuristic_explore.py` for an exploration of why simple
heuristics are fundamentally limited here. Don't rely on coherence tests
as the primary regression signal — use deterministic greedy reproducers
or PCC.

---

## 6. Common workflows

### "Did this change break perf?"

1. Capture baseline tok/s before the change (`grep "Sample per second"`).
2. Apply change.
3. Capture new tok/s same way.
4. Δ > 5% → investigate. Δ < 5% → noise.

For investigation: run tracy on both, diff the per-op breakdown.

### "Did this change break correctness?"

1. Run `test_llama_3_2_1b --optimization-level=1` benchmark for PCC
   (HF-based, CPU reference, won't catch vLLM-specific bugs).
2. Run `test_output_coherence_nongreedy` (vLLM path, heuristic).
3. Run `test_llama3_3b_generation_opt_level_1[batch1]` (vLLM path, full
   generation).
4. For deterministic spot-check: greedy decode of `"I like taking walks
   in the"` on Llama-3.2-1B at opt_level=1, max_tokens=20. Healthy output
   like `" woods. I like to go to the beach. I like to go to the park..."`.
   Repetition loop = bad.

### "Why is this op slow?"

1. Tracy capture, find the op in `ops_perf_results_*.csv`.
2. Sort by duration; check whether the slow ops are:
   - Forward-pass kernels (matmul, sdpa, layernorm) → kernel performance
   - Sampling-graph ops (sort, softmax, argmax) → sampling perf
   - to_layout / to_memory_config / typecast → layout shuffling overhead
   - Gaps between ops (no op recorded) → host dispatch overhead
3. Compare against the same op in a "fast" baseline run.

### "Is fusion X firing?"

1. Capture IR with `TTXLA_LOGGER_LEVEL=DEBUG`.
2. Extract: `extract_mlir_graphs.py debug.log --type ttnn -o /tmp/run`.
3. `grep -c "ttnn.<fused_op_name>" /tmp/run/graph_3_ttnn.mlir`.
4. Cross-reference op count against expected (e.g. 32 RoPE sites in
   Llama-3.2-1B; 16 layers × 1 SDPA per decode; etc.).

---

## 7. Reference: where to look

| Topic | File / location |
|---|---|
| vLLM plugin overrides (RoPE replacement, RMSNorm) | `integrations/vllm_plugin/vllm_tt/overrides.py` |
| vLLM plugin sampler | `integrations/vllm_plugin/vllm_tt/sampler.py` |
| vLLM plugin model_runner (compile config, generate loop) | `integrations/vllm_plugin/vllm_tt/model_runner.py` |
| Compile options / TTConfig | `integrations/vllm_plugin/vllm_tt/platform.py` |
| Custom ops registry (tt::sampling) | `python_package/tt_torch/custom_ops.py` |
| RoPE fusion patterns | `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/RoPEFusingPattern.cpp` |
| RoPE workaround (seq_len padding) | `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.cpp` |
| TTNN fusing pass orchestration | `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNFusing.cpp` |
| TTNN pipelines (where opt_level=1 wires up the optimizer) | `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp` |
| Existing benchmark tests | `tests/benchmark/test_llms.py`, `tests/benchmark/benchmarks/llm_benchmark.py` |
| vLLM coherence test | `tests/integrations/vllm_plugin/sampling/test_sampling_params.py` |
| vLLM generation tests | `tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py` |
| RoPE lit tests | `third_party/tt-mlir/src/tt-mlir/test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/rotary_embedding/rope.mlir` |

---

## 8. Lessons from recent perf-debug episodes (cheat sheet)

### RoPE fusion correctness ↔ perf trade-off

The vLLM trace PR ([#4388](https://github.com/tenstorrent/tt-xla/pull/4388))
introduced `TTRotaryEmbedding` to keep RoPE on-device for metal trace.
This produces an MLIR pattern (`outer + cos + sin`) that triggers
`RoPEExpandedFusing` in tt-mlir, but the kernel can't broadcast cos/sin's
dim -2 from 1 to >1 → silent wrong outputs.

Two layered fixes block the bad fusion: `isPackedCosSinPair` (#8054) for
the packed-cache case, and a dim -2 mismatch check (this conversation's
PR) for the on-the-fly case. **Both fixes block fusion → unfused expanded
form runs → ~18% decode tok/s drop on 8B.**

Current workaround: disable `ISINSTANCE_OVERRIDES` in `overrides.py` so
Llama uses native vLLM cache+gather RoPE, which fuses cleanly via the
existing kernel path. Trade-off: metal trace doesn't apply to Llama
anymore.

Future work to recover perf:
1. Kernel-side: support broadcast on cos/sin dim -2 (would let
   `TTRotaryEmbedding`'s pattern fuse correctly).
2. Or: rewrite `TTRotaryEmbedding` to emit a pattern the kernel handles —
   current attempts hit either `isPackedCosSinPair` (which blocks) or the
   dim -2 mismatch (also blocks).

### Sampling perf

See `perf_debug/issue_3940_sampling_perf.md` and
`perf_debug/issue_3940_update_apr12.md` for the multi-week investigation.
TL;DR:
- `tt::sampling` custom op exists for non-greedy sampling and is faster
  than CPU fallback.
- Pad-to-32 batch trick speeds up multi-core topk significantly.
- Metal trace gives 1.4-2.5x speedup at decode steady state.
- See [`SAMPLER_TESTING_PLAN.md`](./SAMPLER_TESTING_PLAN.md) for the test
  pyramid for sampler changes.
