# Chunked Prefill on the TT vLLM backend — a guided explanation

Audience: an engineer who will own/defend the chunked-prefill PR (tt-xla #4986)
in review. Goal: by the end you can explain **every** change and **why** it's
there, from first principles. Read top-to-bottom the first time.

Files touched: `platform.py`, `model_runner.py`, `attention.py`,
`scheduler/ascend_scheduler.py`, plus benchmark plumbing.

---

## Part 0 — Current status (read me first)

Where the feature stands as of this round of debugging (Llama-3.2-3B, identical
prompts, greedy, `fp32_dest_acc_en=False` unless noted):

**Working / fixed:**
- **batch=1 multi-chunk** is token-identical to single-shot. ✅
- **batch>1 single-shot** is correct **once the #5116 fix is present** — a
  *pre-existing* tt-mlir bug (the per-user last-token gather index is computed by
  a bf16-accumulating matmul; flat indices ≥512 round to the wrong row). Fixed in
  tt-mlir and pinned by this branch. Not a chunked-prefill bug; chunked prefill
  just rides on top of it. (See `DEBUG_JOURNEY_batch_prefill_5116.md`.)
- **batch>1 multi-chunk** — two real bugs in *our* chunked code, both fixed:
  1. **Non-block-aligned chunk** (`_block_aligned_chunk` took the raw budget
     remainder when it was `< block_size`, leaving `num_computed` mid-block and
     misplacing the KV fill). Fixed: defer instead.
  2. **Mixed-stage prefill packing** (a fresh request, `num_computed=0`, packed
     into the same step as a continuation, `num_computed>0`, was forced through
     the cached-prefix path and corrupted). Fixed: only batch same-stage prefills
     per step; defer stage-mismatched ones.
- Validated: batch 4 / 8 / 16 multi-chunk now produce **identical output across
  identical prompts** (the core correctness property); same-stage batching
  (multiple fresh users, one chunked) is correct.

**Open items (tracked):**
- **Recompilation after warm-up** can trigger on some chunked configs
  (`VLLM_XLA_CHECK_RECOMPILATION=1` flagged `num_xla_graphs 10→11` at one budget):
  a precompile-coverage gap (not all chunk-step `(L_bucket, S_bucket)` shapes are
  enumerated in `_precompile_backbone`). Needs the precompile enumeration widened.
- **batch-size-dependent numerics vs the single-chunk reference:** at batch≥8 the
  (consistent-across-users) output drifts from the batch-1 single-chunk result in
  late decode. This is *not* per-user divergence (all users agree); it looks like
  batch-size-dependent bf16 numerics. The right oracle is same-batch single-shot
  (not batch-1) — re-test before treating as a bug.
- **Mixed-stage fix is the scheduler option (Option 1).** A higher-throughput
  alternative (Option 2: make the cached-prefix path correct for mixed-stage
  batches) is flagged as a perf follow-up — see Part 6.

---

## Part 1 — Concepts you need first

### 1.1 Prefill vs decode
An LLM request has two phases:
- **Prefill**: process the whole prompt (say 500 tokens) in one shot to populate
  the model's attention **KV cache** and produce the first output token.
- **Decode**: generate the rest one token at a time, each step attending to all
  previous tokens' K/V (read from the cache).

Prefill is compute-heavy and parallel over prompt tokens; decode is one token at
a time. They have very different shapes, so the TT backend compiles them
separately.

### 1.2 KV cache, paged KV cache, blocks, page tables
Attention needs every previous token's **K**(ey) and **V**(alue). We store them
in a **KV cache** so we don't recompute. vLLM uses a **paged** KV cache: the
cache is split into fixed-size **blocks** (a.k.a. pages) of `block_size` tokens.
Each request gets a **page table**: a list of physical block ids holding its
sequence, in order. `block_size` on TT is **32** (also the hardware **tile**
height — remember this, it matters later).

So "the K/V for logical position `p`" lives in block `page_table[p // 32]` at
offset `p % 32`.

### 1.3 Batching, and what "batch>1" means here
The server processes several requests ("users") together as a batch to use the
hardware efficiently. In prefill, a batch of N users is a tensor with a leading
dimension of N. Each row is an independent request.

### 1.4 How the TT backend compiles (this is the crux of #4986)
The TT backend runs models via `torch.compile(backend="tt")` in
**`DYNAMO_TRACE_ONCE`** mode: a graph is compiled **once per input shape** and
reused. Tenstorrent hardware wants **static shapes** — a different sequence
length is a different compiled graph.

To avoid compiling a graph for every possible prompt length, the runner
**buckets** the token count to a fixed ladder of sizes:
`num_tokens_paddings = [1, 128, 256, 512, …]` (powers of two, ≥ 32). A prompt of
500 tokens is padded up to the 512 bucket and run on the 512-shaped graph.

At engine start, **`capture_model()`** pre-compiles a graph for **every** bucket
(`_precompile_backbone`). After that, a strict invariant holds: **no graph may
compile at runtime** (`VLLM_XLA_CHECK_RECOMPILATION` asserts this) — a runtime
recompile is a latency cliff and a correctness smell.

**Key consequence:** the largest bucket is `max(num_tokens_paddings)`. Before
this PR that ladder went all the way up to `max_model_len`. So:
- The number of buckets (hence engine-init compile time) grows with
  `max_model_len`.
- The biggest prefill graph allocates activations sized to
  `batch × max_model_len` — at batch 32 × 16K that was a ~3 GB activation buffer
  → DRAM OOM.

### 1.5 The TT prefill attention paths (in `attention.py`)
For a prefill step the backend does two things per layer:
1. **Write** the new tokens' K/V into the paged cache: `ttnn.paged_fill_cache`
   using a **`fill_page_table`** (the page table rolled so the chunk lands in the
   right suffix blocks — more below).
2. **Compute** attention. The existing **fast path** (`_compute_full_attention`)
   does a dense self-attention `SDPA(Q, K, V, is_causal=True)` over **just the
   tokens in this call** — it does *not* read the cache.

That fast path is correct only when the whole prompt is in this one call (the K
it attends to *is* the whole prompt). That assumption is exactly what chunking
breaks.

### 1.6 The scheduler (`AscendScheduler`, "prefill-first")
vLLM's stock scheduler interleaves prefill and decode in one step (continuous
batching). The TT backend instead uses a **prefill-first** custom scheduler: in a
given step it schedules prefills *or* decodes, never a mix, because the TT model
runner processes all rows of a step with one shared shape. Original behavior: if
a prompt didn't fit the per-step token budget, it was **skipped** (and if it
exceeded a limit, dropped).

---

## Part 2 — The problem (#4986)

The per-step token budget is `max_num_batched_tokens`. Before this PR the runner
asserted:

```
max_model_len * max_num_seqs <= max_num_batched_tokens
```

i.e. the budget had to be big enough to prefill every sequence to full length in
one step. That ties everything to `max_model_len`:
- buckets/compile-time scale with `max_model_len` (~50 s per doubling on 3B);
- prefill activation = `batch × max_model_len` → OOM at large batch×len;
- a long prefill monopolizes the device (tail latency).

The fix is **chunked prefill**: cap the per-step budget at a small fixed
**chunk size** (e.g. 2048), decoupled from `max_model_len`, and process a long
prompt as several chunks across steps.

---

## Part 3 — The solution in two stages

- **Stage A — decouple buckets/memory.** Cap `num_tokens_paddings` (and thus the
  biggest prefill graph + activation) at the chunk budget, not `max_model_len`.
  KV/page-table buffers stay sized to `max_model_len` (the cache still spans full
  context). This alone gives flat compile time + bounded prefill DRAM, but only
  works if prompts fit one chunk — unless Stage B makes multi-chunk correct.

- **Stage B — functional multi-chunk.** Teach the scheduler to slice a long
  prompt into chunks, and teach attention to make chunk N attend to the K/V of
  chunks 1..N-1 that are already in the paged cache.

The hard part is Stage B's attention, because of §1.5: the fast path ignores the
cache.

---

## Part 4 — Walkthrough of every change

### 4.1 `platform.py` — turn chunked prefill on and set the budget
- New `TTConfig.prefill_chunk_size`, **default `0` = disabled**. The feature is
  **opt-in**: set `additional_config={"prefill_chunk_size": N}` (N>0) to enable.
  When unset, the branch below is a no-op and behavior is exactly as before
  (legacy assertion + full-length buckets). Advertised limitation: at
  `max_num_seqs > 1` output may be inconsistent across requests (pre-existing
  precision bug, §6.2; correct at `max_num_seqs == 1`).
- In `check_and_update_config`, a new **non-MLA, generative** branch (only when
  `prefill_chunk_size > 0`):
  ```python
  budget = max(min(scheduler_config.max_num_batched_tokens, chunk_size), floor)
  scheduler_config.enable_chunked_prefill = True       # real vLLM field
  scheduler_config.chunked_prefill_enabled = True       # TT-internal flag
  scheduler_config.max_num_batched_tokens = budget
  ```
  `floor = max(block_size, max_num_seqs)` so the budget always holds at least one
  block and one decode token per running seq.

  Why two flags? `chunked_prefill_enabled` is **not** a real vLLM
  `SchedulerConfig` field — it's a TT-internal flag the existing code already uses
  (the MLA branch sets it; the scheduler's commented `super().schedule()` reads
  it). We set both so the TT convention and vLLM's own field agree. (A reviewer
  may ask "why not just `enable_chunked_prefill`?" — answer: we follow the
  established TT convention; unifying them is a separate cleanup.)

### 4.2 `model_runner.py` — bucket/activation capping (Stage A)
- Relax the assertion: under chunked prefill require only
  `max_num_batched_tokens >= max(block_size, max_num_seqs)`; keep the old
  whole-prompt assertion for the legacy non-chunked path.
- `self.prefill_chunk_budget = min(max_num_batched_tokens, max_model_len)` and
  pass it as `max_token_size` to the existing `_get_token_paddings(...)`. That
  single change shrinks the bucket ladder, hence `max_num_tokens =
  paddings[-1]`, hence every activation buffer sized from it.
- **Untouched on purpose:** `num_reqs_max_model_len`, `max_num_blocks_per_req`,
  and the `*_page_table_dev_*` / `cache_position` buffers — these stay at
  `max_model_len` because the KV cache still spans the full context.

### 4.3 `attention.py` — the cached-prefix prefill path (Stage B, the crux)
`TTMetadata` already carries `attn_mask` and `fill_page_table`. The change is in
`_compute_full_attention`:

```python
has_paged_cache = isinstance(kv_cache,(list,tuple)) and kv_cache[0].numel() > 0
prefix_chunk_mode = (attn_metadata.attn_mask is not None
                     and has_paged_cache and not shared_kv_mode)
```
- **Fast path** (first/only chunk, `attn_mask is None`): unchanged — dense
  causal self-attention over this chunk. No regression.
- **prefix_chunk_mode** (a chunk whose prefix is already cached): after
  `_handle_paged_attention` has written *this* chunk to the cache, gather the
  **full** `[0:S]` K/V (prefix + this chunk) from the paged cache with the
  existing `_gather_paged_to_dense(...)`, then run
  `SDPA(Q_chunk, K_full, V_full, is_causal=False, attn_mask=mask)`.

Why these exact choices:
- We gather from `page_table` (logical order), not `fill_page_table` (the
  rolled write-order table).
- `S` is derived from the page-table shape, so it's a **compile-time constant**
  (`= max_model_len`) → the graph shape is stable, no per-step recompiles.
- `is_causal=False` because the explicit `mask` already encodes causality **and**
  the prefix offset (the op can't express "query at absolute position p0+r").

The mask is built in `model_runner._build_prefix_chunk_mask`: shape
`[users, 1, L, S]`, where for user `u` with `p0 = num_computed_tokens[u]` and
`n = num_scheduled[u]`, query row `r` may attend to key columns `c <= p0 + r`
for `r < n` (0 allowed, −inf blocked). This is the standard additive attention
mask, generalized to a rectangular `L×S` with a per-user prefix offset. (The
same float-mask SDPA path is already used in production by the pooling runner —
that's why no kernel change was needed.)

### 4.4 `model_runner.py` — wiring the mask + precompiling the slow path
- In `_prepare_inputs`, build the mask only when it's a prefill chunk with a
  cached prefix: `padded_total_num_scheduled_tokens > 1 and
  any(num_computed_tokens > 0)`. Otherwise `attn_mask=None` (fast path / decode).
- `_dummy_run(prefix_chunk=True)` builds a dummy mask **and a distinct
  `fill_page_table` tensor** so the precompiled graph has the same input arity as
  runtime (a continuation chunk uses a separate rolled `fill_page_table`; if the
  dummy aliased it to `page_table`, one extra graph would compile at runtime).
- `_precompile_backbone` now also compiles the prefix-chunk graph for each
  `num_tokens > 1` bucket (via the `_run_backbone_dummies` helper from
  `/simplify`). Result: passes `VLLM_XLA_CHECK_RECOMPILATION=1` at batch=1.

### 4.5 `scheduler/ascend_scheduler.py` — slice prompts into chunks
Several coordinated changes:
- **Chunk instead of skip:** when `num_new_tokens > token_budget` and chunked
  prefill is on, take `num_new_tokens = self._block_aligned_chunk(num_new_tokens,
  token_budget)` instead of skipping.
- **`_block_aligned_chunk` (extracted helper):** rounds the chunk down to a
  multiple of `block_size`, and avoids leaving a 1-token remainder. **Why
  block-aligned?** The cached-prefix write (`fill_page_table`) is rolled by
  *whole blocks* (`num_computed // block_size`); if a chunk weren't a multiple of
  32 the next chunk's prefix wouldn't be block-aligned and the roll would land in
  the wrong block. **Why no 1-token remainder?** A lone final token would be a
  1-token "prefill" chunk, but attention decides prefill vs decode by
  `query_len > 1`, so it'd be misrouted to the decode path.
- **Partial-prefill continuation:** a chunk that doesn't finish the prompt is
  **not** added to `self.running` (the decode loop requires every running request
  to have exactly one uncomputed token). Instead it's re-enqueued
  (`step_skipped_waiting`) so the prefill loop continues it next step. Its status
  is `RUNNING`, so it's emitted as a *cached* request (`scheduled_running_reqs`),
  not a new one. Only when the final chunk completes is it added to
  `self.running`.
- **Invariant fix:** the original `scheduled_reqs <= len(self.running)` assertion
  is relaxed by `+ num_partial_prefill_scheduled` (partials are scheduled but
  intentionally not in `running`).
- **`_get_prompt_limit`:** returns `max_model_len` under chunked prefill (the old
  `min(max_model_len, per-step budget)` would reject any prompt longer than the
  chunk budget *before* it ever got chunked).
- **Latent bug fixed:** the `num_computed > 0` branch called a non-existent
  `kv_cache_manager.create_empty_block_list()`. Replaced with the manager's
  `empty_kv_cache_blocks` singleton — and it must be that exact singleton because
  `allocate_slots` compares it **by identity** to decide whether to call
  `allocate_new_computed_blocks` (which asserts the request has no blocks yet);
  a fresh empty instance would wrongly trip that on chunk ≥ 2.

### 4.6 Benchmark plumbing
`VLLMBenchmarkConfig.max_num_batched_tokens` (+ `TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS`)
so the benchmark can set the budget independently of `batch × len`.

---

## Part 5 — Why it's correct (the equivalence argument)

Claim: greedy output is identical whether the prompt is prefilled in one chunk
or many.

- For chunk N, the queries are the chunk's tokens at absolute positions
  `[p0, p0+n)`. They must attend to all keys `[0, p0+r]` causally.
- `_handle_paged_attention` writes chunk N's K/V into the cache *before* the
  attention compute, so the gathered `K_full[:, :, 0:p0+n]` contains the true
  prefix (written by earlier chunks) followed by this chunk.
- The mask allows exactly `c <= p0 + r`, which is the same set of keys a
  single-shot prefill's causal mask would allow for those positions.
- Therefore each chunk's per-position attention output equals what a single-shot
  prefill would compute → identical logits → identical greedy tokens.

**Verified on device:** a 566-token prompt at chunk budget 128 (5 chunks)
produces **token-for-token identical** output to a single-shot prefill, at
batch=1. Test: `tests/integrations/vllm_plugin/generative/test_chunked_prefill.py`.

---

## Part 6 — What this PR does NOT solve (be ready for this in review)

1. **Memory cost of the gather (perf, not correctness).** The cached-prefix path
   gathers a dense `[users, kv_heads, max_model_len, head]` K and V each chunk
   (~1 GB at batch32×16K). It's smaller than the backbone activation Stage A
   removed, and only during multi-chunk prefill — but it partly offsets the
   savings at large batch. The performant successor is a **paged
   flash-attention prefill kernel** (a tt-mlir/tt-metal op that reads K/V from
   the paged cache with a query-position offset; no dense gather, no materialized
   mask). Flagged, not built.
2. **Pre-existing batch>1 prefill non-determinism (#5116)** — now **fixed in
   tt-mlir** (the gather-index bf16 → fp32 fix), and this branch pins that
   tt-mlir. It was never a chunked-prefill bug; chunked prefill rode on top of it.
   See `DEBUG_JOURNEY_batch_prefill_5116.md`. (Status: resolved; keep the pin.)
3. **Mixed-stage prefill — Option 1 (scheduler) shipped; Option 2 (attention)
   deferred.** The fix serializes prefill requests at different `num_computed`
   stages (Part 0, bug #2). The higher-throughput alternative is to make the
   cached-prefix path correct for mixed-stage batches so they stay batched.
   **⚠️ Perf follow-up:** revisit Option 2 if staggered-prefill throughput is
   insufficient; the per-op-dump tooling to pin the mixed-step mechanism is in the
   debug-journey history (`TT_DUMP_USERS` patch to `program_executor.cpp`).
4. **Strict no-recompile at batch>1**: not all chunk-step graph shapes are in the
   precompile sweep yet (`VLLM_XLA_CHECK_RECOMPILATION=1` can trip on some
   budgets). Functional; one-time stall. Needs the precompile enumeration widened.

---

## Part 7 — Run it yourself

Single-device env:
```
export TT_MESH_GRAPH_DESC_PATH=$HOME/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
```
- Correctness (single-chunk vs multi-chunk equivalence):
  `pytest -svv tests/integrations/vllm_plugin/generative/test_chunked_prefill.py`
- Stage A wins (flat compile / no OOM):
  `TT_BENCHMARK_BATCH_SIZE=1  TT_BENCHMARK_MAX_MODEL_LEN=8192  TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=2048 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_GMU=0.5 pytest -svv tests/benchmark/test_vllm_benchmarks.py -k "llama-3.2-3b and not embedding"`
  then the same with `TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=16384` (no OOM).
- No-recompile check at batch=1: prepend `VLLM_XLA_CHECK_RECOMPILATION=1` to a
  multi-chunk run.

---

## Part 8 — Glossary + self-check

**Glossary.** prefill/decode (§1.1); KV cache / paged cache / block / page table
(§1.2); `block_size`=32=tile height (§1.2); bucketing / `num_tokens_paddings`
(§1.4); `capture_model` / precompile / no-recompile invariant (§1.4);
`paged_fill_cache` / `fill_page_table` (§1.5); SDPA (§1.5); prefill-first
scheduler (§1.6); `max_num_batched_tokens` = per-step token budget (§2);
`chunked_prefill_enabled` = TT-internal flag (§4.1); `_block_aligned_chunk`
(§4.5); `prefix_chunk_mode` (§4.3); `_build_prefix_chunk_mask` (§4.3).

**Could you answer these in review?**
1. Why does the old `max_model_len * max_num_seqs <= max_num_batched_tokens`
   assertion exist, and why is it safe to relax under chunked prefill? (§2, §4.2)
2. Why must chunks be block-aligned? (§4.5)
3. Why does `_compute_full_attention` gather from `page_table` but write with
   `fill_page_table`? (§1.5, §4.3)
4. Why `is_causal=False` in the prefix-chunk SDPA? (§4.3)
5. Why is `S` (gather length) a constant, and why does that matter? (§4.3, §1.4)
6. Why is a partial-prefill request kept out of `self.running`? (§4.5)
7. Why the `empty_kv_cache_blocks` *singleton* (not a fresh empty)? (§4.5)
8. Is the batch>1 divergence caused by this PR? (§6.2 — no, pre-existing)
9. What's the perf cost of the gather, and what removes it? (§6.1)
