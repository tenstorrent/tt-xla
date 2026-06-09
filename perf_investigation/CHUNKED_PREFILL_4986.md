# Chunked Prefill in vllm_tt — implementation notes (tt-xla #4986)

Status: **working end-to-end on device** (Llama-3.2-3B-Instruct, BFP8 weights +
BFP8 KV, single P150, `TT_VISIBLE_DEVICES=0`).

## What & why

Before this work the TT vLLM backend keyed prefill precompile buckets and the
prefill activation buffer to `max_model_len`, and the model-runner asserted
`max_model_len * max_num_seqs <= max_num_batched_tokens`. Consequences:
engine-init compile time grew ~linearly with `max_model_len`; peak prefill
activation was `batch × max_model_len` (batch32 × 16K → ~3 GB alloc → DRAM OOM);
long prefills monopolized the device.

The change caps the per-step prefill token budget (`max_num_batched_tokens`) at a
small `prefill_chunk_size`, **decoupled** from `max_model_len`, and processes
longer prompts in chunks. KV-cache / page-table buffers stay sized to
`max_model_len`.

**Opt-in.** `prefill_chunk_size` defaults to `0` (disabled) — set
`additional_config={"prefill_chunk_size": N}` (N>0) to enable. When off, behavior
is unchanged. **Known limitation (advertised):** at `max_num_seqs > 1` output may
be inconsistent across requests due to a pre-existing batched-prefill precision
bug (`fp32_dest_acc_en` / tt-mlir #8666; see
`ISSUE_batch_prefill_nondeterminism.md`); correct at `max_num_seqs == 1`. Fix
tracked in parallel.

## Stage A — decouple buckets / activation (compile-time + DRAM)

- `platform.py` `TTConfig.prefill_chunk_size` (default 0 = opt-in). When > 0,
  `check_and_update_config`'s non-MLA generative branch enables chunked prefill
  and caps `max_num_batched_tokens = min(current, prefill_chunk_size)` (floored at
  `max(block_size, max_num_seqs)`).
- `model_runner.py`: relaxed the `max_model_len*max_num_seqs <= budget` assertion
  (only enforced for the non-chunked legacy path); `num_tokens_paddings` now
  capped at `prefill_chunk_budget = min(max_num_batched_tokens, max_model_len)`.
  Page-table/KV buffers untouched.
- Benchmark: `VLLMBenchmarkConfig.max_num_batched_tokens` + env var
  `TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS`.

**Validated (BFP8 weights+KV):**
- batch1 × max_model_len=8192, budget=2048 → **PASS** (impossible before: old
  assertion required budget ≥ 8192). Engine init 139 s; `compile_ranges_endpoints=[2048]`.
- batch32 × max_model_len=16384, budget=2048 → **PASS, no OOM** (the issue's
  headline 3 GB-activation failure case). 32 reqs × 128 tok, ~26 tok/s.

## Stage B — functional multi-chunk prefill (correctness)

Approach (tt-xla only, no tt-mlir kernel): a prefill chunk whose prefix is already
in the paged KV cache (`num_computed > 0`) gathers the full prefix+chunk K/V from
the paged cache and runs masked SDPA. The explicit `[users,1,L,S]` mask encodes,
per user with prefix `p0` and chunk `n`, that query row `r` attends to keys
`c <= p0+r` for `r < n`. `S = max_model_len` (constant) so the gather/mask shapes
are stable per `L` bucket.

- `attention.py` `_compute_full_attention`: new `prefix_chunk_mode` branch
  (triggered when `attn_mask is not None and paged cache present`), gathers via the
  existing `_gather_paged_to_dense`, `is_causal=False`. Fast path (first chunk /
  decode / pooling) unchanged. The tt SDPA op's float-mask path is the same one the
  pooling runner already uses in production, so no kernel change was needed.
- `model_runner.py` `_prepare_inputs`: builds the mask (`_build_prefix_chunk_mask`)
  when `padded_total_num_scheduled_tokens > 1 and any(num_computed > 0)`; else
  `attn_mask=None`.
- `ascend_scheduler.py`: chunk instead of skip when the prompt exceeds the budget;
  chunks are block-aligned (so the cumulative prefix stays block-aligned for the
  `fill_page_table` roll); partial-prefill requests are kept out of `self.running`
  and re-enqueued for continuation; `_get_prompt_limit` returns `max_model_len`
  under chunked prefill.

**Validated:** same 566-token prompt, greedy, BFP8 — single-chunk (budget=2048)
vs multi-chunk (budget=128 → 5 block-aligned chunks) produce **token-for-token
identical** output. This is the equivalence oracle in
`tests/integrations/vllm_plugin/generative/test_chunked_prefill.py`.

Because the cached-prefix attention path is the same one prefix-caching uses, this
also fixes the known prefix-cache greedy-degeneration bug
(`project-tt-prefix-caching-corrupts-greedy`) — to be confirmed/reconciled
separately.

## Design notes: two distinctions worth being explicit about

Two reasonable concerns come up about chunked prefill on this stack; both are valid
and worth recording precisely.

**(1) "Which parts of the KV cache get updated" — and why batch>1 is the hard part.**
Each chunk must write its new K/V into the correct *suffix* blocks of that request's
paged cache. tt-xla does this with `paged_fill_cache` + a `fill_page_table` rolled by
`num_computed // block_size`; this is why chunks must be **block-aligned** (a
non-aligned prefix breaks the whole-block roll). The genuinely difficult case is
batch>1 with *multiple requests simultaneously mid-chunk at different prefill
positions in one forward pass*, each needing its own write offset. The TT
`AscendScheduler` sidesteps this by being **prefill-first and serializing**: it
spends the whole per-step token budget on a single request's chunk, so there is
never more than one partial prefill in a batch at once (the "at most 1 partial
request" invariant, preserved by this change). That keeps per-request KV-update
bookkeeping tractable. Note: a *separate*, pre-existing batched-prefill correctness
issue does exist (identical greedy prompts diverge at batch>1, reproduces with
chunking disabled — see Limitations); it is in the batched fill/SDPA machinery, not
in the chunking logic.

**(2) There is no dedicated chunked-prefill op in the tt-xla → tt-mlir stack.**
Correct — there is no paged flash-attention *prefill* kernel with a query-position
offset. This implementation deliberately avoids needing one: it composes ops already
present in the stack — `paged_fill_cache` (write the chunk), `_gather_paged_to_dense`
(pull prefix+chunk back out of the paged cache; already used by the shared-KV path),
and `scaled_dot_product_attention` with an explicit float mask (the path the pooling
runner already exercises in production). So *correct* chunked prefill needs **no
tt-mlir change**; the trade-off is the dense gather's memory cost. A purpose-built
paged-prefill kernel (see Limitations) is the performant successor, not a
prerequisite.

## Issues encountered (in order)

1. **Env / multi-device.** Default run picked up a 2-device mesh ("Device count
   mismatch: 1 vs 2", then a compile failure). Fix: scope to one chip with
   `TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto` +
   `TT_VISIBLE_DEVICES=0`. The `TT_FATAL: ... eth core connects to remote mmio`
   lines are benign (cores skipped).
2. **`chunked_prefill_enabled` is not a vLLM SchedulerConfig field** — it is a
   TT-internal flag read by AscendScheduler/TTModelRunner. Must set it *and* the
   real `enable_chunked_prefill`. Reads use `getattr(..., False)`.
3. **`_get_prompt_limit` clamped to the per-step budget** (`min(max_model_len,
   max_num_scheduled_tokens)`), so a prompt longer than the chunk budget was
   rejected (`FINISHED_IGNORED`) before chunking. Fixed: return `max_model_len`
   under chunked prefill.
4. **vLLM hangs on an ignored request** (async scheduler) — operational gotcha; a
   hard `kill -9` of the EngineCore leaves the device needing `tt-smi -r`. Use
   SIGTERM / let it exit. (Avoided once #3 was fixed.)
5. **Prefill-first scheduler had no partial-prefill continuation.** It popped a
   request from `waiting`, appended to `self.running`, and the decode loop asserted
   every running request has exactly 1 uncomputed token. Cascade of fixes:
   - Keep partial requests out of `self.running`; re-enqueue (`step_skipped_waiting`)
     for the prefill loop to continue; only add to `running` when fully prefilled.
   - Continued chunks have status RUNNING → emit cached request data
     (`scheduled_running_reqs`), not the `raise RuntimeError(invalid status)`.
   - Relaxed the `scheduled_reqs <= len(self.running)` invariant by the count of
     in-flight partial prefills.
   - The `num_computed > 0` else branch called a **non-existent**
     `kv_cache_manager.create_empty_block_list()` (latent bug). Replaced with the
     manager's `empty_kv_cache_blocks` singleton — important: `allocate_slots`
     compares it **by identity** to decide whether to call
     `allocate_new_computed_blocks` (which asserts the request has no blocks yet),
     so a freshly-constructed empty `KVCacheBlocks` wrongly trips it on chunk ≥ 2.

## Known limitations / to flag

- **Recompilation — RESOLVED.** The cached-prefix slow-path graph is now
  precompiled in `_precompile_backbone` (a `_dummy_run(prefix_chunk=True)` variant
  per L bucket > 1, with a non-None `attn_mask` and a *distinct* `fill_page_table`
  tensor so the traced input arity matches runtime). Verified: the multi-chunk
  run passes with `VLLM_XLA_CHECK_RECOMPILATION=1` (no recompilation after warmup).
  Note: the distinct-`fill_page_table` detail mattered — initially one graph still
  compiled at runtime (8 precompiled vs 9) because the dummy aliased
  `fill_page_table` to `page_table` while runtime continuation chunks use a
  separate rolled buffer.
- **Attention-side memory at large batch (tt-mlir/kernel item).** The tt-xla-only
  path gathers a dense `[users, kv_heads, max_model_len, head]` K and V per chunk
  step. For batch32 × 16K this is ~1 GB each — it reintroduces attention-side
  memory that partially offsets Stage A's activation savings. **Proper fix is a
  tt-metal/tt-mlir paged flash-attention *prefill* kernel** with a query-position
  offset (sibling to `paged_scaled_dot_product_attention_decode`): reads K/V
  directly from the paged cache via `page_table` + per-user `seq_len`, applies
  causal masking with the chunk's absolute offset internally — no dense gather, no
  materialized mask. This is the performance endgame; the gather+mask path is the
  correct, no-kernel-change milestone. → open a tt-mlir issue.
- **Pre-existing batched-prefill non-determinism (NOT chunking, separate bug).**
  At batch=4 with four *identical* greedy prompts, the outputs diverge per row
  (e.g. row0==row1, row2/row3 differ). Critically this reproduces with **chunking
  disabled** (large budget, single-step batched prefill: `chunk_multi_b128_batch4`
  vs the no-chunk baseline `chunk_single_b4096_batch4` — both diverge), so it is a
  pre-existing TT batched-prefill issue orthogonal to #4986, not introduced by
  chunked prefill. batch=1 chunked prefill is token-exact vs single-chunk. The
  equivalence test deliberately uses batch=1 to isolate chunking. → worth its own
  investigation (suspect batched SDPA / KV indexing or batch-dependent bf16
  numerics); chunked prefill rides on top of whatever batched-prefill correctness
  exists.
- **Strict no-recompilation at batch>1.** With `VLLM_XLA_CHECK_RECOMPILATION=1`,
  batch>1 multi-chunk trips one runtime recompile (8 vs 9 graphs) — one batched
  prefix-chunk shape is not yet covered by the precompile sweep (batch=1 is fully
  covered and passes strict mode). Functionally fine (the check is off by default);
  the precompile sweep needs to also enumerate the batched prefix-chunk shape.
- **Sliding-window models** (e.g. Gemma): the explicit mask + `sliding_window_size`
  kwarg may double-apply; the prefix-chunk mask does not yet bound the lower edge.
  Llama-3.2-3B has no sliding window. Needs attention if extending to SWA models.
- The mask is currently built on CPU and copied each prefix-chunk step (fixed shape
  per L bucket, so no recompile, but a per-step H2D of an `L×max_model_len` tensor).

## Repro commands

Single device env (prepend to all):
```
export TT_MESH_GRAPH_DESC_PATH=$HOME/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export TT_VISIBLE_DEVICES=0
```

Stage A (compile/DRAM decoupling):
```
TT_BENCHMARK_BATCH_SIZE=1  TT_BENCHMARK_MAX_MODEL_LEN=8192  TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=2048 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_GMU=0.5 \
  pytest -svv tests/benchmark/test_vllm_benchmarks.py -k "llama-3.2-3b and not embedding"
TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=16384 TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=2048 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_GMU=0.6 \
  pytest -svv tests/benchmark/test_vllm_benchmarks.py -k "llama-3.2-3b and not embedding"
```

Stage B (correctness):
```
pytest -svv tests/integrations/vllm_plugin/generative/test_chunked_prefill.py
```
