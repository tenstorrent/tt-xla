# Concat Force-Equal Workaround — Investigation

## Context
Gemma-4-31B on n300_llmbox (1D 8-chip mesh) with the
`pad_attention_heads` feature needs `pad_attention_heads_force_equal=True`
to compile. Without `force_equal`, padding picks the cheaper "replicate
K/V" strategy (Q=32 heads, K/V=8 heads each), and the concat operation
inside `XlaQKVParallelLinear` (which cats per-shard Q/K/V back together)
fails with:

```
TT_FATAL: Cannot get runtime args for kernel reader_concat_interleaved_start_id
that is not placed on core 2-0
  at tt-metal/tt_metal/impl/kernels/kernel.cpp:572
```

Stack trace points at `apply_descriptor_runtime_args(Program&,
ProgramDescriptor)` — the program-cache **HIT** slow path in
`mesh_device_operation_adapter.hpp:553`.

With `force_equal=True`, padding aligns Q==K==V (32 heads each) and the
concat shapes become balanced; the error disappears. So the workaround
trades extra zero-padded compute for a side-step of the tt-metal bug.

## What I found in tt-metal

### Code path (cache HIT, Contract 1)
`mesh_device_operation_adapter.hpp:537-555`:

```cpp
// Contract 1 — simple per-coord factory ... no rt-arg bindings
if (!sv.resolved_bindings.rt_args.empty()) {
    // fast path: just patch buffer addresses via resolved bindings
    tt::tt_metal::apply_resolved_bindings(...);
} else {
    // SLOW path: re-create descriptor from fresh args, apply rt-args
    auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value,
                                 mesh_dispatch_coordinate);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}
```

`ConcatProgramFactory` has only `create_descriptor` and declares **no**
rt-arg bindings, so it always takes the slow path on cache hit.

`apply_descriptor_runtime_args` iterates `desc.kernels[k].runtime_args`,
which is a `vector<(CoreCoord, vector<uint32_t>)>`, and for each entry
calls `Kernel::runtime_args(core)`. That `runtime_args(core)` does:

```cpp
TT_FATAL(logical_core.x < this->core_to_runtime_args_.size() && ...,
         "Cannot get runtime args for kernel {} that is not placed on core {}",
         this->name(), logical_core.str());
```

`core_to_runtime_args_` is sized from the kernel's `core_ranges` at
**compile** time. So the assertion fires when the **fresh** descriptor's
runtime-args set includes a core that **wasn't** in the cached program's
`core_ranges` at compile time.

### What concat sets

`concat_program_factory.cpp`:
- `all_cores`, `num_cores` come from `split_work_to_cores(grid, num_output_pages, rm_orientation=false)`
- `reader_desc.core_ranges = all_cores;`  (line 231) — kernel placed here
- `cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, rm_orientation)` — list of cores for rt-args
- `reader_desc.runtime_args.emplace_back(core, ...)` for each core in `cores`

`split_work_to_cores` (line 337 of work_split.cpp) builds `all_cores`
via `num_cores_to_corerangeset(target_num_cores, grid_size, row_wise)`.
For row_wise=false with starting (0,0), this enumerates cores in column-major
order starting from origin, matching `grid_to_cores(num_cores, ...,
row_wise=false)`'s enumeration `(i / grid_y, i % grid_y)`. They should
agree.

### Why the cache hit fires for a "new" shape

Default cache hash is over `(operation_attributes, tensor_args)`, which for
concat is `(ConcatParams, ConcatInputs{vector<Tensor>})`. Tensor's
`attribute_values()` returns `(storage, tensor_spec)` — so input
TensorSpec (including logical/padded shape) is part of the hash.

**So two concat calls with different input shapes should NOT collide
in the cache.** The same shapes should produce the same `num_output_pages`
and thus the same `all_cores`/`cores`. The mismatch shouldn't be possible
in theory.

But the assertion fires. Possibilities:

1. **`storage` hashing is loose.** If `Storage` only hashes type (DEVICE)
   and not the underlying mesh-buffer pointer or per-shard shape, two
   tensors with the same `tensor_spec` but different ND-sharded shard
   shapes could collide. ND sharding shards are part of the buffer, not
   the tensor_spec.

2. **`sub_core_grids` not normalized.** If two calls pass the same
   `output_mem_config` but the device's `compute_with_storage_grid_size()`
   has been somehow modified between calls (rare but possible if active
   ETH cores get disabled mid-run), `all_cores` could differ across calls
   even with the same operation_attributes.

3. **Per-mesh-coord descriptor variance.** Cache miss builds one Program
   per mesh coordinate range. If any coord's chip has a smaller compute
   grid (e.g. bad core), that coord's program's `all_cores` is smaller.
   On cache hit, `invoke_per_coord` is called per coord — if the new call
   produces a descriptor that wants a core not in *that coord's* program,
   it asserts.

4. **`ConcatInputs` is `vector<Tensor>` and reflection might skip the
   vector size in some configurations** — unlikely but worth checking.

### Why `force_equal` dodges it

With `force_equal`, the cat operands are Q/K/V each of size `padded_q *
head_dim` per chip. They're identically shaped, so the multi-input concat
collapses to a degenerate case (3 equal-shaped tensors → trivial
work-split). My current theory: this drives `num_output_pages` to a
value where `num_cores == grid.x * grid.y` always, which means `all_cores
== full grid` and the assert can't fire because `cores` is also the full
grid.

Without `force_equal`, Q has 4 heads/chip and K/V have 1 head/chip each:
`6 * head_dim` per chip total, and `num_output_pages` ends up at a value
where `num_cores < max_cores`. Now the placement is a proper subset of
the grid, opening the possibility of mismatch under hypotheses 1–3
above.

## Proposed fixes (not yet attempted)

### Fix A — Concat factory always uses full grid for `core_ranges`

```cpp
// In concat_program_factory.cpp
const CoreRange full_grid_range({0,0}, {num_cores_x-1, num_cores_y-1});
reader_desc.core_ranges = CoreRangeSet(full_grid_range);
writer_desc.core_ranges = CoreRangeSet(full_grid_range);
// keep runtime_args only for `cores` (num_cores subset)
```

Trade-off: kernel is compiled on all 64 cores even when only a few have
work. Compile cost goes up slightly but the cache-hit mismatch becomes
impossible because `core_ranges` is invariant across shapes.

**Risk**: kernels with no rt-args might still execute (with stale or zero
args from L1) — would need to verify the framework no-ops kernels with
no rt-args set per-core, or add an explicit "skip if no work" guard.

### Fix B — Make ConcatProgramFactory declare a custom `compute_program_hash`

Force every distinct input-spec combination to miss the cache, sidestepping
the slow-path mismatch entirely. Defeats program caching benefit but is
a one-line change.

### Fix C — Provide an explicit `override_runtime_arguments` instead of
relying on `apply_descriptor`

This would let the factory verify that the cached program's `all_cores`
covers the new `cores` set, and rebuild if not. More invasive.

## Recommendation

For the immediate PR, keep `force_equal=True` as a documented workaround
flag and file a tt-metal issue with the assertion + this analysis. Fix A
is the most surgical permanent fix and should be filed as a follow-up.

## Sibling issue — Qwen2.5-7B + force_equal hang

The push test parametrization `qwen2.5-7b-force-equal` hangs at the first
decode step on n300_llmbox. The engine initializes, warmup completes
("init engine ... took 72.85 seconds"), then the first `llm.generate()`
freezes at `Processed prompts: 0%` and never produces output.

The pattern is **inverted** from Gemma-4-31B:

| Case | Cat shapes | Result |
|---|---|---|
| Qwen2.5-7B, force_equal=False | Q=7/chip, K=V=1/chip (unequal) | **PASSES** |
| Qwen2.5-7B, force_equal=True  | Q=K=V=4/chip                   | **HANGS** |
| Gemma-4-31B, force_equal=False | Q=4/chip, K=V=1/chip (unequal) | **FAILS** (concat assert above) |
| Gemma-4-31B, force_equal=True  | Q=K=V=4/chip                   | **PASSES** |

So the same flag is rescue for Gemma but death for Qwen. Things that
differ between Qwen-force_equal and Gemma-force_equal even though both
produce padded_q=padded_kv=32:

- **head_dim**: Qwen=128, Gemma sliding=256, Gemma global=512.
- **kv_replication_factor c**: Qwen=7, Gemma sliding=2, Gemma global=8.
  (c=7 is a prime that doesn't divide 8 = mesh axis size.)
- **num_layers**: Qwen=28, Gemma=60 (50 sliding + 10 full).
- **max_model_len in the push test**: Qwen=32, Gemma=128.

The c=7 replication is the most suspicious axis: K_head_i gets duplicated
7 times consecutively via `repeat_interleave`, then zero-padded by 4
heads to reach 32. After 8-way TP split, chip 7 receives heads
[28, 29, 30, 31] which are all zero-padded (no real data). This is the
only configuration above where any chip ends up with **all-zero KV**.
A NaN / div-by-zero in attention softmax on that chip is one plausible
hang signature (an attention reduction across chips with one chip
producing NaN can spin in a wait/sync).

### What hasn't been ruled out (no hardware right now)

- All-zero KV on chip 7 → NaN-in-softmax → all-reduce stall.
- KV cache layout interaction with `max_model_len=32 + padded_kv=32`
  (page size collapses to 1, edge case in paging).
- Custom AscendScheduler interaction (Qwen2 path takes scheduler
  decisions Gemma doesn't).

### Next debug steps when board is available
1. Re-run with `TT_METAL_LOGGER_LEVEL=DEBUG` and check what op stalls.
2. Force chip-7 KV heads to be **not** all-zero (e.g. replicate from
   real head 0 instead of zero-padding) and see if the hang clears.
3. Try Qwen2.5-7B + force_equal with **larger** `max_model_len` (256+)
   to rule out small-cache edge cases.
4. py-spy on the hung worker PID to confirm where the host stack is
   parked (XLA enqueue vs metalium poll).

The push test `qwen2.5-7b-force-equal` is currently `pytest.mark.skip`'d
pointing at this investigation. The nightly llmbox Gemma-4-31B test
keeps end-to-end coverage of the same `force_equal` code path until
the Qwen hang is understood.

## Reference logs
- `gemma4_31b_8chip_1dmesh_optA_v4_notrace.log` — original Gemma concat failure
- `gemma4_31b_8chip_1dmesh_optA_v2.log` — earlier Gemma failure
- `qwen25_force_equal_after_reset.log` — Qwen hang on fresh-reset board
- `qwen25_pad_3cases.log` — Qwen pass (no force_equal) + hang (with force_equal)
