# Chunked-prefill qualification sweep — results

Qualifies the refactored on-device chunked SDPA op (tt-xla #4986) in tt-mlir before opening the tt-mlir PR. Driver: [`run_chunked_prefill_qual_sweep.sh`](run_chunked_prefill_qual_sweep.sh).

## Environment

- **Hardware:** p150 (Blackhole), single device.
- **tt-mlir:** `26bc58fa2` (branch `kmabee/issue_8788_add_chunked_sdpa_op`: chunked SDPA op + verifiers + EmitC/EmitPy + lit tests), built **non-debug** (`TT_RUNTIME_DEBUG=OFF`).
- **tt-xla:** branch `kmabee/chunked_prefill_isue_4986_explore.rebase`, tt-mlir pinned to the SHA above.

## Settings (all runs)

| knob | value | env var |
|---|---|---|
| optimization level | 1 | `_BENCH_OPTIMIZATION_LEVEL=1` |
| trace | on | `TT_BENCHMARK_TRACE=1` |
| sampling | device | `TT_BENCHMARK_CPU_SAMPLING=0` |
| weights | BFP8 | (default) |
| KV cache | BFP8 | `TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8` |
| batch | 32 | `TT_BENCHMARK_BATCH_SIZE=32` |
| prefill chunk | 2048 | `TT_BENCHMARK_PREFILL_CHUNK_SIZE=2048` |
| gpu mem util | **0.35** | `TT_BENCHMARK_GMU=0.35` |
| max_num_batched_tokens | **65536** (= batch × chunk) | `TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=65536` |

`gmu=0.35` and `max_num_batched_tokens=65536` are the two changes vs the prior known-good config (`gmu=0.15`, default 2M batched tokens): right-sizing the prefill-step token budget to `batch × chunk` frees enough activation memory to raise the KV budget to 0.35 — including at 64K, where `gmu≈0.3` previously OOM'd at warmup.

## Results — 8/8 PASS

Cells show **decode tps / TTFT (ms)**.

| model | 128 | 4K | 32K | 64K |
|---|---|---|---|---|
| **llama-3.2-3b** | ✅ 25.3 / 1144 | ✅ 24.7 / 1155 | ✅ 25.3 / 1155 | ✅ 24.2 / 1165 |
| **llama-3.1-8b** | ✅ 16.2 / 2121 | ✅ 16.3 / 2125 | ✅ 16.4 / 2120 | ✅ 15.6 / 2140 |

64K KV-cache allocation at `gmu=0.35`: llama-3.2-3b → 208,896 tokens; llama-3.1-8b → 182,784 tokens.

## Findings

- All 8 runs pass; **zero `DataType mismatch` (#5185) asserts** — confirms the non-debug runtime and a healthy chunked-prefill path under device sampling.
- **`gmu=0.35` fits at 64K** for both models with `max_num_batched_tokens=65536` — the headroom the reduced batched-token budget buys.
- Decode throughput is flat from 128 → 64K context (chunked SDPA holds up); no regression vs the earlier `gmu=0.15` runs.

## Reproduce

```bash
./run_chunked_prefill_qual_sweep.sh   # writes logs + SUMMARY.md to a timestamped subdir
```
