# Chunked-prefill compile-time investigation (p150)

Investigating why vLLM warmup/compile is much slower for `max_model_len > 128` even on a warm kernel cache, and whether the chunked-prefill changes can reduce it further.

## TL;DR

- The slowdown above 128 is **not** tt-metal kernel JIT (it persists with a warm cache, 0 `RUNNING SYSTEM COMMAND`) and **not** context length (it's flat across 4K/32K/64K).
- It's the **backbone precompile phase**: it recompiles a full graph per `(token-bucket, prefix_chunk)` combo. Above 128 the bucket ladder fills out (capped at the chunk size) **and** chunked prefill engages, so each prefill bucket is compiled **twice** (`prefix_chunk=False` and `prefix_chunk=True`).
- Chunked prefill already did the big win (caps the ladder at `prefill_chunk_budget` instead of `max_model_len` → flat across context). The remaining lever is the **per-bucket `False`+`True` doubling**.

## Evidence

### Warm cache, still slow (Qwen3-8B rerun)
| seq | wall | kernel JIT (`RUNNING SYSTEM COMMAND`) |
|---|---|---|
| 128 | 6:11 | 0 |
| 4096 | 19:43 | 0 |

→ kernel compilation is not the cost; re-running does not speed up the >128 cells.

### Per-phase timing (llama-3.2-1b, `TTXLA_LOGGER_LEVEL=INFO`, warm)
| | seq 128 | seq 4096 |
|---|---|---|
| `Token paddings` | `[1, 128]` | `[1, 128, 256, 512, 1024, 2048]` (capped at chunk, not 4096) |
| chunked SDPA active | no (budget 128 = max_len) | **yes** (budget 2048 < 4096) |
| **backbone compile** | **88.7 s** | **387.2 s (4.4×)** |
| decode_postprocess | 11.9 s | 9.7 s (flat) |
| total wall | 2:14 | 7:09 |

The entire delta is the backbone phase. `decode_postprocess` is constant.

### Why 4.4×
`_precompile_backbone` runs one full compile per `(num_tokens bucket, prefix_chunk)`:
- 128: ladder `[1,128]`, chunked inactive → `prefix_chunk=False` only → ~2 compiles → 88 s.
- 4096: ladder `[1,128,256,512,1024,2048]`, chunked active → `1 decode + 5 prefill buckets × {False, True}` = **~11 compiles** → 387 s (~35 s/graph).

Flat across 4096/32K/64K because the ladder is capped at the chunk budget (2048), so all of them compile the same 11 graphs.

### Note on the sweep config
The `gmu=0.35` sweeps passed `TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=65536`, but the effective per-step chunk budget is `prefill_chunk_size=2048` (the `Token paddings` cap at 2048 confirms it), so chunked prefill **was** active and the ladder stayed capped — consistent with the flat compile times.

## Cost model

```
backbone_compile ≈ Σ over compiled graphs of (torch front-end + tt-mlir compile) per graph
num graphs at >128 = 1 (decode) + N_prefill_buckets × {False, True when chunked}
```
where `N_prefill_buckets = len(num_tokens_paddings) - 1` (exponential ladder up to `prefill_chunk_budget`).

## Improvement opportunities (sized against the 387 s 4096 backbone)

1. **Drop `prefix_chunk=False` for chunkable buckets (chunked-specific).** When `_chunked_sdpa_active`, route the first/no-prefix chunk through the chunked-SDPA op with `chunk_start_idx=0` instead of a separate standard-prefill graph. Removes ~5 of 11 compiles → backbone ~387 s → **~210 s (~40%)**, same win at 32K/64K. **Gate:** chunked SDPA at `chunk_start_idx=0`/empty prefix must be bit-exact vs standard causal SDPA over the chunk.
2. **Cache the AST parse in `tt_torch/backend/metadata_propagation.py`.** It does `ast.parse(whole file)` twice per FX node, uncached — paid by every graph. Memoize per `(path, mtime)`; helps all buckets (and 128). Low-risk.
3. **Coarsen the token-padding ladder** (`_get_token_paddings`), e.g. `[1,512,2048]` — fewer buckets, trades a little runtime padding for fewer compiles.

## (b) Backbone phase split — warm 1B-4096, py-spy `--native`, 180 s / 14,735 samples

The warm-cache backbone compile is **~100% PyTorch front-end**, ~0% tt-mlir C++, ~0% device wait:

| slice | % of backbone |
|---|---|
| Dynamo / FX / proxy-tensor / pytree tracing | **55.3%** |
| `run_decompositions` / AOT-autograd / functionalize | **28.7%** |
| `metadata_propagation` (the uncached per-node `ast.parse`) | **15.5%** |
| DCE / passes | 0.4% |
| tt-mlir C++ compile | ~0% |
| device / lock / exec wait | ~0% |

**Interpretation:**
- On a warm kernel cache, backbone "compile" is the **torch.export / Dynamo tracing run once per dummy graph** — the tt-mlir compiler and device are negligible in steady state. (Cold-cache runs instead spend their time in tt-metal kernel JIT subprocesses; that's a separate, one-time cost.)
- This **confirms fix #1 is the top lever**: the dominant cost is per-graph and paid for every `(bucket, prefix_chunk)` combo, so removing the 5 `prefix_chunk=False` prefill graphs (~11 → 6) cuts the front-end work proportionally → ~40% off the warm backbone.
- **Fix #2 (AST cache) is a clean, isolated ~15.5% slice** — `metadata_propagation` alone is 1/6th of the entire backbone, from re-parsing source files per FX node. High ROI, low risk, helps every config (incl. 128).
- The remaining ~84% (Dynamo/FX/AOT) is inherent torch.export per-graph cost — only reducible by compiling fewer graphs (#1, #3) or caching/reusing traced graphs across buckets.

### Recommendation (priority)
1. **#1 unify first-chunk into chunked op** (~40% off backbone at ≥4K; gated on `chunk_start_idx=0` bit-exact check).
2. **#2 AST-parse cache** (~15% off, trivial/safe — do regardless).
3. **#3 coarsen the token-padding ladder** (fewer graphs, trades runtime padding).

## Fixes implemented + measured (llama-3.2-1b @ 4096, warm, p150)

Clean attribution (both runs share the same machine; this box is noisy — see variance note):

| build | backbone | vs prev |
|---|---|---|
| baseline | 387.2 s | — |
| **#2** (gate `extract_nodes_info` on `XLA_HLO_DEBUG`) | 358.2 s | −7.5% |
| **#2 + #1** (`TTXLA_UNIFY_FIRST_CHUNK=1`) | **197.4 s** | **−45%** |
| **combined** | 387.2 → 197.4 s | **−49%** |

### #2 — gate `extract_nodes_info` (backend.py) — LANDED
`extract_nodes_info` (per-node AST walk) ran unconditionally but its result is only consumed when `XLA_HLO_DEBUG=1`. Now skipped otherwise. Provably correct waste-removal (result is functionally inert when the flag is off); ~7.5% measured, near the machine's noise floor (the profile's 15.5% was sample-window bias). Zero risk → keep.

### #1 — unify first chunk into chunked SDPA (`TTXLA_UNIFY_FIRST_CHUNK`, default OFF) — PROTOTYPE
When chunked SDPA is active, route the first/no-prefix prefill chunk through it (`chunk_start_idx=0`) instead of a separate standard-prefill graph → drops the `prefix_chunk=False` prefill graphs (11→6 per the bucket ladder).
- **Perf: −45% backbone** (358 → 197 s), and conservative — the ON run paid one-time cold JIT for the newly-engaged chunked-path kernels (`fill_cache`, `embeddings_tilize`); warm would be lower.
- **Correctness: see RESOLVED section below** (bit-exact under bf16 KV).

### #1 correctness — RESOLVED (bf16 bit-exact)
ON-vs-OFF, llama-3.2-1b @ 4096, greedy:
- **bf16 KV cache (no quantization): outputs byte-identical** (full 651-char generation matched). This proves chunked SDPA at `chunk_start_idx=0`/empty-prefix == standard causal prefill.
- bfp8 KV cache: coherent, diverges ~12 tokens — now explained: the chunked op reads K/V from the **BFP8-quantized paged cache**, while standard prefill computes K/V inline. Benign quantization, not a bug.
- Backbone win confirmed both ways: bf16 331→205s (−38%), bfp8 358→197s (−45%).

**Nuance for default-on:** with BFP8 KV, enabling #1 routes *non-chunking* (short) prompts through the chunked op too, so their first-chunk attention now reads the quantized cache instead of inline K/V — output changes slightly (same approximation chunked-prefill continuation already uses, but a behavior change for short prompts). With bf16 KV there is no change. So: correct, but default-on under BFP8 is a small quality/perf tradeoff worth an accuracy spot-check.

### Remaining confirmation for #1 (nice-to-have; evidence already strong)
- **Multi-chunk prompt** (prompt > chunk budget) ON-vs-OFF, bf16, greedy. Not yet run — the benchmark's short prompts never chunk, so it needs a dedicated long-prompt harness. Expected to pass: #1 only changes the *first* chunk's attention (now bit-exact-proven); the continuation path is unchanged, and the K/V cache *write* is the same op in both modes, so forward state into later chunks is identical.
- BFP8 accuracy spot-check before default-on (the short-prompt numerics nuance above).

## Follow-up items

### #1 (TTXLA_UNIFY_FIRST_CHUNK) before default-on
- **BFP8 accuracy spot-check.** Under BFP8 KV, #1 also routes *short* (non-chunking) prompts through the chunked op, so their first-chunk attention reads the quantized cache instead of computing K/V inline → slightly different output (the same approximation chunked-prefill continuation already uses). Under bf16 KV there is zero change (bit-exact). Correct, but an accuracy spot-check is the gating item before defaulting on.
- **Multi-chunk (>2048-token) ON-vs-OFF run** (bf16, greedy). Not yet run — the benchmark's short prompts never chunk, so this needs a dedicated long-prompt harness. Expected to pass: #1 only changes the *first* chunk's attention (now bit-exact-proven), and the K/V cache *write* is the same op in both modes, so forward state into continuation chunks is identical.

## Next big items to investigate (post #1/#2)

Profiled leaf hotspots in the warm backbone (pre-#2 profile; #2 removes the ast.parse slice):
`ast.parse` 10.5% (→ removed by #2), **`ttnn::prim::detail::reshape_map_output_page` 8.7%**, mmap/munmap churn ~6%, FX codegen (`_exec_with_source`) 2.9%.

1. **Serializable / persistent compiled executables (biggest lever).** The runtime logs `Failed to deserialize executable: UNIMPLEMENTED: Deserializing serialized executable not supported` — so nothing is cached across runs and every warmup recompiles from scratch. Implementing executable serialize/deserialize (PJRT/tt-mlir runtime) would (a) let repeat runs skip compile entirely — the biggest real-world win given how often warmup is paid — and (b) unblock **process-parallel precompile** (see multi-core below). Investigate where the deserialize stub is and what the flatbuffer/runtime needs.
2. **`reshape_map_output_page` ~9% (ttnn C++).** Hot during the torch_xla dynamo-bridge graph extraction (`extract_graph_helper`). Investigate whether reshape page-mapping is recomputed redundantly per graph or can be cached; fix likely lives in ttnn/metal, so may be an upstream item.
3. **Prefill token-padding ladder — investigated below.**

## Prefill bucket ladder: usage, coarsening, and downsides

**What it is / when used.** `num_tokens_paddings` (e.g. `[1, 128, 256, 512, 1024, 2048]`; min bucket is `min_context_len`, not 32) gives static device shapes. Each forward pads the **max per-request** scheduled tokens up to the smallest bucket ≥ it (model_runner.py:1097): decode → 1, a prefill chunk of N tokens → smallest bucket ≥ N. One graph + one set of device buffers is compiled per bucket. So a bucket keeps prefill compute proportional to the actual chunk length (less padding waste).

**Which buckets the benchmark actually exercises** (llama-3.2-1b @ 4096, instrumented `BUCKET_USE`):
- `padded=1`: 254 steps (decode), `padded=128`: 4 steps (prefill, 15-token prompt → 128).
- **Buckets 256/512/1024/2048 are compiled but never hit** for this short-prompt workload. They would be hit by longer prompts / multi-token chunks (a ~2000-token chunk → 2048).

**Coarsening experiment** (`TTXLA_PREFILL_PADDINGS` override, #1 off):

| ladder | prefill bucket | backbone | TTFT | decode tps |
|---|---|---|---|---|
| full `[1,128,256,512,1024,2048]` | 128 | 328 s | 459 ms | 45.4 |
| smart `[1,128,2048]` (drop unused) | 128 | **193 s (−41%)** | **460 ms (=)** | 45.4 |
| over-coarse `[1,2048]` (drop 128 too) | 2048 | 155 s (−53%) | **8590 ms (18.7×)** | 44.8 |

**Conclusions.**
- The runtime benefit of a bucket = keeping prefill latency proportional to chunk length. Removing the bucket a chunk needs forces it to pad to the next-larger bucket — here removing 128 blew TTFT up **18.7×** (459 ms → 8.59 s) because a 15-token prefill ran as 2048.
- Dropping **unused** buckets is ~free: smart coarsening cut backbone **−41%** with identical TTFT/decode. But which buckets are unused is **workload-dependent** — a fixed coarse ladder that's right for short prompts is the catastrophic case for longer ones.
- **Robust fix (recommended): lazy / on-demand bucket compilation** — precompile only decode + the smallest prefill bucket, and let torch_xla compile other buckets the first time a runtime chunk hits them. This gets the smart-coarsening savings for *any* workload with no padding penalty; the cost is a one-time compile stall on first use of a new bucket (fine for dev/benchmark; for latency-SLA serving, precompile the expected hot buckets and lazy the tail).
- **Escape hatch (landed): `TTXLA_PREFILL_PADDINGS`** lets an operator set the ladder explicitly when the workload is known (e.g. short-prompt benchmarks → `1,128,2048`).

### Multi-core assessment
The warm-compile bottleneck is **GIL-bound Python** (torch.export / Dynamo / AOT tracing, run once per graph) — threads won't help. The viable path is **process-parallel precompile** (each worker compiles a subset of the token buckets), which **requires executable serialization (#1 above)** so workers can hand compiled executables back / warm a shared cache. Cold-cache tt-metal kernel JIT (the 166 `riscv-tt-elf-g++` subprocesses) is already its own multi-process build and is avoided on warm cache, so it's not the warm-path lever. Net: **#1 (serialization) is the prerequisite for the multi-core win.**

## Repro

```bash
# per-phase timing + bucket ladder
TTXLA_LOGGER_LEVEL=INFO _BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_TRACE=1 \
TT_BENCHMARK_CPU_SAMPLING=0 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_BATCH_SIZE=32 \
TT_BENCHMARK_PREFILL_CHUNK_SIZE=2048 TT_BENCHMARK_GMU=0.35 TT_BENCHMARK_MAX_MODEL_LEN=4096 \
TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=65536 \
python -m pytest -svv tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark -k "llama-3.2-1b"
# then: grep -E "Token paddings|Compilation finished in" <log>
```
