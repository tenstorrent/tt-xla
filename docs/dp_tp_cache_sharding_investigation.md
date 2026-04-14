# DP+TP KV Cache Sharding Investigation

**Test:** `test_gpt_oss_20b_tp_galaxy_batch_size_64`
**Date:** March 2026
**Branch:** `mvasiljevic/gpt_oss_dp_tp_sharding`

## The Original Error

```
TT_FATAL @ paged_update_cache_device_operation.cpp:
Non-paged update_cache: input batch dim (input.padded_shape[1]=16)
must match cache batch dim (cache.padded_shape[0]=64).
```

The benchmark has two phases: **warmup** (16 steps) and **benchmark** (20 steps).
During warmup, `update_cache` works fine. During benchmark, it crashes because the
input tensor's batch dimension (16, after 4-way batch sharding on Galaxy) doesn't
match the cache tensor's batch dimension (64, unsharded).

## Root Cause Analysis

### How GSPMD sharding inference works

The test uses `_batch_parallel_input_sharding_fn` which marks `input_ids` with
`("batch", None)` sharding. During compilation, XLA's GSPMD partitioner propagates
this batch sharding through the graph. When it reaches `update_cache`, it sees the
input has batch dim 16 (sharded) and **infers** that the cache should also be batch-
sharded, inserting the necessary partitioning.

Key insight: The cache is marked with `(None, "model", None, None)` — batch dimension
is `None` (replicated). GSPMD **overrides** this during compilation, inferring that
the cache must be batch-sharded to match the input. This inference is tied to the
specific compiled graph.

### Why it fails in the benchmark phase

1. **Warmup** compiles graphs with GSPMD inferring batch sharding on the cache → works
2. **Benchmark** creates a **fresh** cache (new tensor, no XLA history)
3. `torch.compile` **reuses the warmup's compiled graph** (Dynamo cache hit)
4. The reused graph expects batch-sharded cache (padded_shape[0]=16)
5. The fresh cache is batch-replicated (padded_shape[0]=64)
6. `update_cache` sees 16 ≠ 64 → `TT_FATAL`

### Why `max_output_tokens=5` worked

With `max_output_tokens=5`, only 5 decode steps run. The compiled graph from warmup
expects cache positions 0-21 (17 prefill + 5 decode - 1). The benchmark also runs 5
decode steps, so positions align → Dynamo reuses the graph successfully and GSPMD
sharding matches. With `max_output_tokens=20`, positions diverge → different graph
shape → but Dynamo still tries to reuse → shape mismatch on cache.

## Attempted Fixes and Outcomes

### Attempt 1: Explicit batch sharding on cache

**Change:** Mark cache with `("batch", "model", None, None)` instead of
`(None, "model", None, None)` from the start.

**Result:** All-zero logits, PCC failure.

**Why it failed:** The prefill step runs with unsharded inputs (full batch). Explicitly
batch-sharding the cache from the start creates a mismatch during prefill — the model
writes to a batch-sharded cache but reads with unsharded attention, producing garbage.
GSPMD inference handles this correctly by only applying batch sharding when appropriate;
forcing it breaks the prefill-to-decode transition.

### Attempt 2: Reuse warmup cache in benchmark

**Change:** Instead of creating a fresh cache for the benchmark, pass
`past_key_values=input_args["past_key_values"]` (the warmup's cache).

**Result:** All-zero logits, PCC failure.

**Why it failed:** The warmup cache contains stale KV data from 16 warmup decode steps.
The benchmark starts fresh (cache_position=0), but the cache tensor's XLA IR history
is entangled with the warmup computation. XLA doesn't properly handle reusing a tensor
that was already consumed by a previous graph execution for a new independent
computation.

### Attempt 3: Reuse warmup cache + zero it

**Change:** Same as Attempt 2 but call `layer.keys.zero_()` and `layer.values.zero_()`
before reusing.

**Result:** `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`

**Why it failed:** In-place `zero_()` on XLA tensors creates a mutation in the XLA
trace that conflicts with the compiled graph. XLA lazy tensors don't support arbitrary
in-place mutations on tensors that are already part of a compiled computation graph.

### Attempt 4: `torch._dynamo.reset()` before benchmark (THE FIX)

**Change:** Before creating the benchmark's fresh cache, call `torch._dynamo.reset()`
to clear Dynamo's graph cache, then `torch.compile(model, backend="tt")` to get a
fresh compiled model.

**Result:** Correct logits, PCC=0.998810. The `TT_FATAL` is resolved.

**Why it works:** Resetting Dynamo forces recompilation for the benchmark's fresh cache.
GSPMD re-infers batch sharding on the new cache tensors during the new compilation.
The compiled graph and the cache tensors are now consistent.

**New problem:** The first 3 benchmark iterations are ~10x slower because compilation
happens during the timed benchmark phase instead of during warmup.

### Attempt 5: Compile warmup to absorb recompilation cost

**Change:** After `torch._dynamo.reset()`, run a short "compile warmup" (2 steps) with
a temporary cache to absorb the compilation cost before the timed benchmark starts.

**Result:** Compilation cost is absorbed. But see Attempt 6.

### Attempt 6: Reuse compile warmup's cache for benchmark

**Change:** After the compile warmup (Attempt 5), reuse its cache for the benchmark
to avoid a second `transfer_to_device`.

**Result:** All-zero logits, PCC failure (same as Attempt 2).

**Why it failed:** Same fundamental issue as Attempt 2 — XLA tensors with existing IR
history (from the compile warmup's 2 steps) produce all-zero logits when reused for
a new computation starting from cache_position=0. The XLA lazy evaluation system does
not support "resetting" a tensor's computational history.

## Current Solution (Attempt 5 without Attempt 6's optimization)

```
Warmup (16 steps)
    ↓
torch._dynamo.reset()
torch.compile(model, backend="tt")
    ↓
Compile warmup (2 steps, fresh cache, absorbs compilation cost)
    ↓
Create FRESH cache (past_key_values=None)
transfer_to_device()
xs.mark_sharding(cache, (None, "model", None, None))
    ↓
Benchmark (20 steps, timed)
```

The `transfer_to_device` for the fresh cache is a one-time cost before the benchmark
loop, not per-iteration.

## Prefill Input Sharding Investigation

The current solution (Attempt 5) only batch-shards decode inputs. Prefill runs with
full-batch (unsharded) `input_ids`. The following attempts tried to shard prefill
inputs to match decode, so the entire pipeline runs with DP-sharded activations.

### Attempt 7: Batch-shard only prefill inputs (keep cache replicated)

**Change:** Apply `input_sharding_fn` (marks `input_ids` with `("batch", None)`)
before the prefill forward pass. Cache remains `(None, "model", None, None)`.

**Result:** All-zero logits, PCC failure. No inf/NaN in cache.

**Why it failed:** GSPMD does not propagate batch sharding through the prefill graph
the same way it does for decode. For decode (seq_len=1), GSPMD overrides the cache's
replicated batch annotation and infers batch sharding. For prefill (seq_len=17),
GSPMD does not make this inference — the cache stays batch-replicated while the
activations are batch-sharded, producing a mismatch that yields zeros.

### Attempt 8: Batch-shard both prefill inputs AND cache

**Change:** Same as Attempt 7, plus change `cache_batch_spec` from
`(None, "model", None, None)` to `("batch", "model", None, None)` when
`input_sharding_fn` is active. Both inputs and cache are batch-sharded from the start.

**Result:** All-zero logits. **inf/NaN values in the KV cache** after every forward
pass (including warmup step 0 and decode step 1). All steps broken — not just prefill.

**Diagnostics (added to `generate_and_benchmark`):**

```
[DIAG WARMUP] step0 logits: nonzero=0/218783744   ← prefill: all zeros
[DIAG WARMUP] step1 logits: nonzero=0/12869632    ← decode: also all zeros

[DIAG PREFILL] input_ids: nonzero=1088/1088 min=25 max=15040  ← inputs are correct
[DIAG PREFILL] cache layer0 keys:   nonzero=0/4194304          ← fresh cache, correct
[DIAG PREFILL] logits: nonzero=0/218783744                     ← all-zero output
[DIAG PREFILL] post-fwd cache layer0 keys: nonzero=1045208/4194304 min=-inf max=inf std=nan
[DIAG PREFILL] post-fwd cache layer0 values: nonzero=1045320/4194304 min=-inf max=inf std=nan
```

The model received valid inputs, produced **inf/NaN in the KV cache**, and returned
all-zero logits. Expected nonzero count for 17 cache positions:
`64 × 8 × 17 × 64 = 557,056`. Actual: `1,045,208` (~2x expected), confirming memory
layout corruption — values written to wrong positions.

## TTIR Graph Analysis

### Reference: old graphs (run27b3) — prefill with UNSHARDED inputs

In the original working configuration, prefill had unsharded inputs and the cache was
`(None, "model", None, None)`. The TTIR prefill graph (g1) showed:

- Cache entered at local `64x1x128x64` (batch-replicated, TP-only sharded)
- GSPMD inserted `all_to_all` to redistribute batch from 64 to 16 inside the graph
- `update_cache` operated at batch=16 after redistribution
- Cache output was `shard_to_full [4,8,1,1]` (DP+TP sharded)

This was a one-time layout transition: prefill converted the cache from replicated
to DP+TP sharded, and subsequent decode steps consumed the sharded layout.

### New graphs (run1981) — prefill with BATCH-SHARDED inputs + cache

With the Attempt 8 changes (`input_sharding_fn` before prefill, cache marked
`("batch", "model", None, None)`), the newest TTIR graphs show that **GSPMD handles
this correctly at the graph level**:

**g0 (317KB) — Warmup prefill (seq_len=17):**

```
Cache input:  64x8x128x64 (global) → 16x1x128x64 (local)
              shard_dims=[0,1], shard_shape=[4,8,1,1]
              ↓ mesh_shard full_to_shard at function entry
              ↓
input_ids:    64x17 (global) → 16x17 (local)
              ↓ mesh_shard full_to_shard (batch dim only)
              ↓
Compute:      all ops run at batch=16
              ↓
fill_cache:   16x1x128x64 cache, 1x1x17x64 new KV
              160 ops total (5 layers × 2 K/V × 16 batch entries)
              batch_offset = 0..15
              ↓
Cache output: 16x1x128x64 → shard_to_full [4,8,1,1] → 64x8x128x64
```

- Cache enters **already batch-sharded** (local batch=16)
- **Zero `all_to_all` operations** — no redistribution needed
- Uses `fill_cache` (writes all 17 positions per batch entry) instead of `update_cache`
- All compute at batch=16

**g3 (247KB) — Compile warmup decode (seq_len=1, after dynamo reset):**

- Cache enters at local `16x1x128x64` (batch-sharded)
- Uses `update_cache` on `16x1x128x64` — standard decode pattern
- Zero `all_to_all`, all compute at batch=16

**g4 (1.7KB) — Auxiliary `ReplicateShardedData`:**

- All-gathers `16x17 → 64x17` i64 tensor (cache positions)
- Pure layout/replication helper, no model compute

### Key finding: GSPMD generates correct graphs

The TTIR graphs are **structurally correct**. GSPMD properly batch-shards the cache
at input for both prefill and decode when `("batch", "model", None, None)` is used.
The prefill graph uses `mesh_shard full_to_shard` (same as decode), not `all_to_all`.
`fill_cache` operates on the correct local batch=16 with `batch_offset = 0..15`.

### The bug is downstream of TTIR

Despite correct TTIR graphs, the run produces inf/NaN in the KV cache. The bug is
in the **TTIR→TTNN lowering or TTNN runtime execution** of `fill_cache` /
`paged_update_cache` when the cache tensor is batch-sharded on the DP axis. The
correct TTIR shapes (`16x1x128x64` cache, `1x1x17x64` fill values) do not translate
to correct TTNN execution on the physical device memory layout.

Evidence:
- TTIR graph: `fill_cache` at batch=16 with `batch_offset=0..15` — correct
- Runtime result: `min=-inf max=inf std=nan` in cache, ~2x expected nonzero count
- This points to a memory layout mismatch at the TTNN level, not a graph-level issue

## Next Steps

The TTIR graphs are correct. The fix is in the TTNN layer:

1. **Investigate `fill_cache` lowering:** Check how `ttir.fill_cache` with
   batch-sharded inputs lowers to `ttnn.paged_update_cache`. The `batch_offset`
   semantics may not account for the DP batch sharding — offset 0..15 may
   index into the wrong physical memory when the tensor is sharded across 4 devices.

2. **Investigate `paged_update_cache` device operation:** The C++ implementation
   (`paged_update_cache_device_operation.cpp`) validates `input.padded_shape[1]`
   vs `cache.padded_shape[0]`. With batch-sharded cache, these shapes may be
   inconsistent between what TTIR expects and what TTNN physically allocates.

3. **Compare TTNN graphs:** Diff the TTNN-level g0 graph (`ttnn_*_g0_*.mlir`) to see
   how `fill_cache` is lowered and whether the `paged_update_cache` shapes match the
   TTIR intent.

## Key Learnings

1. **GSPMD handles batch-sharded cache correctly for prefill.** When the cache is
   explicitly marked `("batch", "model", None, None)`, GSPMD generates correct TTIR
   graphs for both prefill (using `fill_cache`) and decode (using `update_cache`).
   The earlier hypothesis that GSPMD always uses `all_to_all` for prefill was based
   on old graphs where inputs were unsharded — with sharded inputs + cache, GSPMD
   uses `mesh_shard full_to_shard` at entry (same as decode).

2. **`fill_cache` vs `update_cache`:** GSPMD uses `fill_cache` for prefill (writes
   all seq_len positions per batch entry, one `fill_cache` per batch index) and
   `update_cache` for decode (writes 1 position for all batch entries at once).
   The prefill `fill_cache` path with batch-sharded tensors may have a lowering bug.

3. **The inf/NaN is a TTNN-level bug, not a GSPMD bug.** The TTIR graphs are correct
   (local batch=16, `batch_offset=0..15`, no `all_to_all`), but the runtime produces
   corrupt values. The bug is in how `fill_cache` or `paged_update_cache` handles
   batch-sharded device memory layouts.

4. **XLA tensor IR history is sticky:** You cannot reuse an XLA tensor that participated
   in a previous computation for a new independent computation. The tensor's lazy IR
   is entangled with the previous graph. Always create fresh tensors.

5. **`torch.compile` graph caching hides sharding mismatches:** Dynamo's cache doesn't
   account for GSPMD sharding state. A cached graph compiled with batch-sharded cache
   will be reused for an unsharded cache, causing `TT_FATAL` at the TTNN level.

6. **`torch._dynamo.reset()` is the escape hatch:** When you need a fresh compilation
   with different tensor shapes/sharding, reset Dynamo's cache. But absorb the
   recompilation cost in an untimed warmup phase.

7. **In-place ops on XLA tensors in compiled graphs are dangerous:** Operations like
   `zero_()` on tensors that are part of compiled XLA graphs cause `RuntimeError`.
   XLA's functional IR model doesn't mix well with PyTorch's mutation semantics.

8. **Always verify IR graphs match the actual run configuration.** Old TTIR graphs
   from a run with unsharded inputs led to an incorrect root cause analysis
   (`all_to_all` hypothesis). The newest graphs from the actual failing run showed
   correct graph structure, shifting the bug to the TTNN runtime layer.
