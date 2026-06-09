# Handoff — chunked prefill (#4986): benchmarking + the real (no-gather) implementation

Purpose: give a new session a rolling start on two tasks:
1. **Benchmark sweep** of the current chunked-prefill implementation.
2. **Alternative implementation** using `ttnn.transformer.chunked_scaled_dot_product_attention` (removes the gather workaround), then **benchmark again**.

Read the companion docs for depth: `CHUNKED_PREFILL_EXPLAINED.md` (teaching guide; "Part 0 — Current status"), `DEBUG_JOURNEY_batch_prefill_5116.md` (full investigation + the chunked+trace root cause), `CHUNKED_PREFILL_4986_testing_tips.md` (how to test).

---

## 0. Environment / how to run

Single device (P150 / Blackhole):
```
source venv/activate
export TT_MESH_GRAPH_DESC_PATH=$PWD/third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0   # device profiler/multiproc crash workaround
```
IR dump: `TTXLA_LOGGER_LEVEL=DEBUG ...` then `~/scripts/extract_mlir_graphs.py <log> --type all`.

## 1. Current state (branches / build)

- **tt-xla branch:** `kmabee/chunked_prefill_isue_4986_explore` (rebased on latest main). Commits on top of main:
  - `40254f210` Uplift tt-mlir to #5116 fix + tt_forge_models
  - `ed73fac61` **chunked prefill (#4986)** — the feature + both batch>1 correctness fixes
  - `9123955f4` #5116 batch-prefill determinism: investigation + regression test
  - `85bdc75d3` chunked prefill: recompilation-guardrail fix
  - `3eaa4158f` docs
  - Backup before history cleanup: `backup/chunked-prefill-pre-cleanup` (`3368813cd`).
- **tt-mlir:** pinned at **`cc7e160f3`** = `66d2edc` uplift + the #5116 gather-index fp32 fix (cherry-picked; also on branch `kmabee/issue-5116-gather-index-fp32-dest-acc`). Built + installed. The single fix file is `lib/Dialect/TTNN/Transforms/TTNNSetComputeKernelConfig.cpp`.
- **Repro scripts (untracked, in `tmp/`):** `chunked_prefill_equiv.py` (single-chunk vs multi-chunk equivalence, knobs `<budget> <batch>`, `CPE_FP32_ACC`), `chunked_min.py` (small multi-chunk, `CM_BUDGET/CM_BATCH/CM_NLAYERS/CM_MAX_TOKENS`), `trace_repro.py` (chunked+trace failure on OPT-125M), `batch_determinism.py`, `ttnn_prod_shapes.py`.

## 2. What's been achieved (so the new session doesn't re-derive it)

**#5116 (pre-existing, NOT chunked-prefill) — FIXED in tt-mlir.** Batch>1 prefill of identical prompts diverged because tt-mlir lowers the per-user last-token **gather index** to a bf16-accumulating matmul; flat indices ≥512 round to the wrong row (bf16 integer step = 4 above 512). Fix: force `fp32_dest_acc_en` on matmuls feeding an embedding's index operand. Filed tt-xla #5116, assigned `mmilosevicTT`. **Required for any correct batch>1 prefill** (chunked or not).

**Chunked prefill (#4986) correctness — DONE for batch>1 (no-trace).** Two real bugs in *our* code, both fixed (in `scheduler/ascend_scheduler.py`):
- **BUG #1:** `_block_aligned_chunk` took the raw budget remainder when `< block_size`, leaving `num_computed` mid-block → block-granular fill misplaced KV. Fix: defer the request instead.
- **BUG #2:** the scheduler packed a *fresh* request (`num_computed=0`) into the same step as a *continuation* (`num_computed>0`); the fresh one was forced through the cached-prefix path and corrupted. Fix: only batch **same-`num_computed`-stage** prefills per step.
- **Validated (Llama-3.2-3B, identical prompts, greedy, fp32 OFF, no trace):** batch=1 token-identical to single-shot; batch 4/8/16 all-identical across users; **chunked == single-shot at the same batch** (equivalence); same-stage batching (multiple fresh, one chunked) correct.

**Recompilation guardrail — FIXED.** `VLLM_XLA_CHECK_RECOMPILATION=1` falsely tripped: it locked the graph count after step 1, but chunked prefill first-uses several chunk-size shapes across the first few steps. Fix (`model_runner.py`): warm-up spans multiple steps — accept new graphs until the set stabilizes, then enforce. Passes now.

## 3. The one open issue: chunked prefill + `enable_trace`

**Symptom:** `enable_trace=True` + chunked prefill fails to compile in `capture_model`:
```
'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device.
%N = ttnn.from_device(page_table)[1x4 si32] ... is not on device
```
Repro: `tmp/trace_repro.py` (OPT-125M, seq128, b1). Isolation: trace-alone ✓, chunked-alone ✓, only the combination fails.

**Root cause (confirmed via traceback):** the **gather workaround** breaks trace. `capture_model` precompiles the prefix-chunk gather graph (even at seq128); `_gather_paged_to_dense` does `torch.index_select(cache, 0, page_table.reshape(-1))`, which tt-mlir lowers by moving the **page-table index to host** (`from_device`) — and that host page-table becomes a trace output, which trace forbids. (Correctness is unaffected; only trace/perf.)

## 4. The gather workaround vs the real fix (the crux for next session)

**Need:** a prefill chunk must attend over the prefix tokens already in the paged KV cache.

**Current workaround (tt-xla only, `attention.py::_compute_full_attention` prefix-chunk path):**
1. write chunk K/V to cache (`paged_fill_cache`),
2. **gather** full prefix+chunk K/V into a dense tensor via `page_table` (`_gather_paged_to_dense`, `index_select`),
3. dense `scaled_dot_product_attention` with `is_causal=False` + an explicit `[users,1,L,S]` causal+offset mask built on host.
- **Limitations:** (a) dense gather re-materializes `[users, kv_heads, max_model_len, head]` K & V each chunk → memory/perf, can OOM at large `batch × max_model_len`; (b) **incompatible with trace** (see §3).

**Real fix (proposed by Het Shah — VERIFIED the op exists in tt-metal):** use `ttnn.transformer.chunked_scaled_dot_product_attention(Q, K, V, page_table_tensor, chunk_start_idx, ...)` (in tt-metal at `.../operations/transformer/sdpa/sdpa.hpp`; scalar- and tensor-`chunk_start_idx` overloads). It reads the paged K/V via `page_table` **on device** with a `chunk_start_idx` prefix offset and masks causally internally — exactly the cached-prefix prefill attention. It is **not** wired through tt-mlir yet.
- Plan: (1) **keep `paged_fill_cache` for the chunk write — do NOT switch to `paged_update_cache`.** A prefill chunk is a *contiguous multi-token* write, which is `paged_fill_cache`'s job; `paged_update_cache` is the *decode single-position* primitive. The canonical reference (`models/tt_transformers/tt/attention.py`) confirms this exact split (prefill `paged_fill_cache` with an offset `chunk_page_table` at L1010–1047; decode `paged_update_cache` at L688–695), and our code already matches it. The non-zero chunk offset is carried by the rolled `fill_page_table` (== the reference's `chunk_page_table`); no fill-op change needed. (Het's original step-(1) suggestion to use the update-idx fill here is a likely-wrong detour.) (2) call `chunked_scaled_dot_product_attention(Q_chunk, K_cache, V_cache, page_table, chunk_start_idx_tensor=num_computed)` instead of gather+masked-SDPA — use the **`chunk_start_idx_tensor` (device-tensor) overload**, not the scalar, so the offset stays on-device and trace-compatible; (3) add a tt-xla `torch.ops.tt` custom op + **tt-mlir lowering (custom call) + sharding rules** for it. **This (the attention op) is the only remaining work — the fill stays as-is.**
- **Removes both limitations:** no dense gather (memory/perf) and **trace-compatible** (page-table consumed on device, like `paged_scaled_dot_product_attention_decode` which works under trace today).
- **Constraint:** `chunk_start_idx` must be a multiple of `q_chunk_size` — dovetails with BUG #1 (block-aligned chunks); keep chunk starts aligned.

### Scope clarification — what's DONE vs what remains (re: Jonathan Azpur's feedback)

The feature has two separable parts; don't re-build the first.

**(A) Orchestration / filling the cache in chunks — ALREADY DONE on this branch.**
Split the prompt into block-aligned chunks, schedule them across steps, write each
chunk's K/V into the paged cache at the *growing* offset, track `num_computed` per
request. This is the AscendScheduler + model_runner + attention fill, validated
(batch=1 token-identical; batch>1 correct after BUG #1/#2). So "enable chunked
prefill from tt-xla / in vLLM" is **complete**, not a TODO.

**Non-0-index cache write — ALREADY HANDLED, and `paged_fill_cache` is the correct
primitive (keep it).** Chunk N is written at offset `num_computed` via
`paged_fill_cache` + a `fill_page_table` rolled by whole blocks (chunks are
block-aligned, so the block-granular roll is exact) — identical to the canonical
reference's prefill path (`paged_fill_cache` + `chunk_page_table`). Do **not** switch
the chunk write to `paged_update_cache`: that's the *decode single-position* op, not
a contiguous multi-token chunk write (see §4 plan step 1).

**(B) Per-chunk cached-prefix attention — the ONLY part that's a workaround.**
Currently: gather prefix+chunk K/V dense + masked SDPA. Het's
`chunked_scaled_dot_product_attention` replaces exactly this — called **per chunk**
during the prefill loop (`chunk_start_idx = num_computed`), reading the paged cache
on-device. This is the endgame and removes the gather's memory cost + the
trace incompatibility.

**So the remaining work is NOT "enable chunked prefill first"** (done) — it is a
drop-in replacement of the gather (part B): wire `chunked_scaled_dot_product_attention`
through tt-mlir (custom-call lowering + sharding — the one piece not yet wired) and a
tt-xla custom op, emit it in `attention.py`'s chunked path, and optionally switch the
fill to `paged_update_cache`.

## 5. Task 1 — benchmark sweep

Goal: sweep Llama-3.2-3B and Llama-3.1-8B at `b1` and `b32` × `seq_len ∈ {128, 16384, 65536}`, BFP8 KV cache, `gpu_mem_util=0.15`, `optimization_level=1`, `enable_trace`.

Invocation (per cell):
```
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_TRACE=1 TT_BENCHMARK_GMU=0.15 \
TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_BATCH_SIZE=<1|32> \
TT_BENCHMARK_MAX_MODEL_LEN=<128|16384|65536> TT_BENCHMARK_PREFILL_CHUNK_SIZE=2048 \
pytest -svv tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark -k "<llama-3.2-3b|llama-3.1-8b>"
```
Knobs live in `tests/benchmark/test_vllm_benchmarks.py` (`_config`). Note: opt>0 + trace auto-sets `cpu_sampling=True` (else TTConfig raises).

**Blocking caveats / decisions for the sweep:**
- 16K/64K need chunked prefill (`PREFILL_CHUNK_SIZE`) — without it `max_num_batched_tokens = batch×max_model_len` explodes and the non-chunked assertion blocks capping it.
- **But chunked + trace is broken (§3).** So either: (a) land the real fix (§4) first, then sweep with trace; or (b) sweep with `TT_BENCHMARK_TRACE=0` on the chunked (16K/64K) cells (trace on for seq128). Recommend (a) if doing the real fix anyway, else (b).
- `gmu=0.15` is conservative: KV cache = `batch × max_model_len` slots; expect **OOM on large `batch×seq`** cells (e.g. b32×16K ≈ 29 GB, b32×64K ≈ 100+ GB for 3B). Report which fit; that's a valid result.
- Capture per cell: pass/fail/OOM, tokens/s, and check **all users' outputs identical** (benchmark only PCC-checks User 0 — see testing tips).

## 6. Task 2 — implement the real fix, then re-benchmark

Implement §4 (chunked SDPA + paged_update_cache + tt-mlir lowering/sharding). Validate correctness with the existing oracles (`tmp/chunked_prefill_equiv.py`: batch 1/4/8/16, multi-chunk == single-shot; identical prompts → identical outputs). Confirm it's **trace-compatible** (the §3 repro `tmp/trace_repro.py` should pass). Then re-run the §5 sweep **with trace enabled** and compare perf/memory vs the gather workaround (the gather's dense re-materialization should disappear → larger `batch×seq` should fit).

## 7. Debug tooling that paid off (reusable)

- **Per-op activation row-diff:** env-gated (`TT_DUMP_USERS`) patch in tt-mlir `runtime/lib/ttnn/program_executor.cpp` that, after each op, compares each user's output block to user 0 and logs the first diverging op. (Reverted; re-apply from `DEBUG_JOURNEY` history — it found both #5116 and the chunked bugs.) Note runtime per-op `debug::Hooks` are gated behind `TT_RUNTIME_DEBUG==1` (off by default).
- **Oracles:** identical-prompts-in-a-batch (greedy must be identical across users); single-chunk vs multi-chunk equivalence; same-batch single-shot as the oracle (NOT batch-1 — batch-size numerics differ benignly).
- **`fp32_dest_acc_en` as a precision-vs-logic discriminator:** if a divergence vanishes under fp32 it's a compute-config/precision issue (tt-mlir family); if not, it's a logic bug (our code).
