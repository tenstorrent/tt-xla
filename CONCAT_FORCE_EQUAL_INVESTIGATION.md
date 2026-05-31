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

## Live attempt — Fix A prototype (2026-05-31)

Patched `concat_program_factory.cpp` in-place (uncommitted, kept in
working tree) with the following:

1. Pin `reader_desc.core_ranges` and `writer_desc.core_ranges` to the
   full compute grid (or `sub_core_grids` when provided), regardless
   of `num_cores` returned by `split_work_to_cores`.
2. After the work-split runtime-args loop, emplace **zero-valued
   runtime args** of the correct shape on every full-grid core that
   isn't in the work-split `cores`. The kernel's first arg is
   `num_tiles`; setting it to 0 short-circuits the per-tile loop, so
   the kernel is effectively a no-op on those cores.

### Build sequence (notes for next session)

```
# Edit concat_program_factory.cpp in tt-metal source
cd /localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release
cmake --build . --target _ttnncpp.so -- -j 16   # rebuilds + links
# Manual sync — the tt-mlir ExternalProject install step is unreliable here
cp ttnn/_ttnncpp.so /localdev/kmabee/tt-xla/third_party/tt-mlir/install/lib/_ttnncpp.so
# Also sync libtt_metal.so and _ttnn.so if any non-ttnn source was touched.
```

Important caveat I burned ~30 min on: running `cmake --build build`
from tt-xla root does pick up tt-metal source changes, but the install
step that copies `_ttnncpp.so` to `install/lib/` is occasionally
silently stale — the source `.so` had my new code, the install
`.so` did not. Always `md5sum` both after a rebuild and `cp` manually
if they differ.

### What Fix A demonstrably did

Adding a `TT_FATAL` tripwire confirmed `ConcatProgramFactory::create_descriptor`
IS reached on first call with `num_cores=64` (full 8x8 grid). Then
running with the real fix:

- **Eliminated the "Cannot get runtime args for kernel ... not placed
  on core X-Y" assertion** — never fired again in any subsequent run.
- **Then a different cache-related assertion surfaced**: "Index N is
  larger than runtime args size 0" — same root cause family (cached
  program's per-core rt-args storage smaller than fresh call wants).
- **Adding zero-rt-args on every full-grid no-work core eliminated
  that one too** — never fired in subsequent runs either.

### What Fix A could not conclusively demonstrate

After the two assertions cleared, the gemma4-31b-it benchmark ran for
22+ minutes of compile without errors, but never reached the
"Warmup complete" or "Starting benchmark" milestones before hitting
the test timeout. The log went silent ~7 min in — last entry was a
normal `_build_params_and_consts` warning, then nothing for ~22 min.
Could be one of:

- **Genuinely slow compile**: without `force_equal`, more diverse
  concat shapes → more kernel cache misses → longer compile. The
  force_equal version completes in 10 min.
- **Silent hang**: a tt-metal-side issue triggered by the zero-rt-args
  no-work cores (e.g., kernel dispatch waiting on a core that's
  silently faulted).
- **Resource exhaustion**: hugepages or shm leaks across multiple
  failed attempts.

We did NOT get a single full PASSED run without `force_equal`.
The patch is live in the working tree but not validated end-to-end.

### Patch summary (concat_program_factory.cpp)

```cpp
// Replace:
//   reader_desc.core_ranges = all_cores;
//   writer_desc.core_ranges = all_cores;
// With (before emplacing runtime args):
const CoreRangeSet kernel_core_ranges = (sub_core_grids.has_value() && !output.is_sharded())
    ? sub_core_grids.value()
    : CoreRangeSet(CoreRange({0, 0},
        {device->compute_with_storage_grid_size().x - 1,
         device->compute_with_storage_grid_size().y - 1}));
reader_desc.core_ranges = kernel_core_ranges;
writer_desc.core_ranges = kernel_core_ranges;

// And after the runtime-args emplace loop, fill no-work cores:
if (!sub_core_grids.has_value() || output.is_sharded()) {
    std::set<CoreCoord> cores_with_work(cores.begin(), cores.end());
    const size_t reader_rt_size = 3 + 3 * num_input_tensors;
    const size_t writer_rt_size = rm_layout ? 4 : 3;
    for (uint32_t x = 0; x < num_cores_x; ++x) {
        for (uint32_t y = 0; y < num_cores_y; ++y) {
            CoreCoord c{x, y};
            if (cores_with_work.count(c) == 0) {
                reader_desc.runtime_args.emplace_back(c, std::vector<uint32_t>(reader_rt_size, 0));
                writer_desc.runtime_args.emplace_back(c, std::vector<uint32_t>(writer_rt_size, 0));
            }
        }
    }
}
```

### 60-min final retry (2026-05-31, 17:18 → 18:16)

Ran with `timeout 3600` on a freshly-reset board. Same pattern:
- Padding applied (sliding kv_repl=1, global kv_repl=2 — exactly
  what previously triggered concat assert).
- KV cache configured (1280 tokens, 10x concurrency).
- Three "Failed to deserialize executable" compile cycles in the
  first ~6 min, last log line at 17:24:36.
- Then **complete log silence for 52 minutes** while EngineCore
  stayed alive at 114% CPU.
- No errors, no assertions, no exceptions surfaced before the
  60-min timeout killed it.

`py-spy dump` on the hung EngineCore showed Python parked in:
```
sync (torch_xla/torch_xla.py:87)
_precompile_backbone (vllm_tt/model_runner.py:2220)
capture_model (vllm_tt/model_runner.py:2400)
compile_or_warm_up_model (vllm_tt/worker.py:351)
```

So Python is blocked on a `torch_xla.sync()` call — waiting for the
XLA → tt-mlir → tt-metal compile/dispatch on the C++ side to finish.
The 114% CPU is a worker thread on the C++ side doing actual work.
Whether that's genuinely-slow-compile or a deadlock-without-error is
unclear from py-spy alone — would need gdb on the C++ side to know.

### Conclusion

**Fix A demonstrably eliminates the two concat assertions** that
forced us to use `force_equal=True`. The "Cannot get runtime args for
kernel ... not placed on core" assertion never fires after the patch,
and the related "Index N is larger than runtime args size 0"
assertion (which surfaced as a follow-on cache mismatch) is also
eliminated by emplacing zero-rt-args on the no-work full-grid cores.

**But Fix A is not sufficient to drop `force_equal` in practice.**
Without `force_equal`, the test never completes the precompile
sweep within a 60-min budget — much slower than the 10-min
`force_equal=True` baseline. Whether this is fundamentally-more-work
(many more distinct concat shapes to compile) or a downstream
silent-hang masked by my patch is unknown.

**Action**: keep `pad_attention_heads_force_equal=True` as the
shipped workaround. The tt-metal concat_program_factory.cpp patch is
left in the working tree (uncommitted, per your instruction) as a
candidate upstream change. If you want to push it upstream, it needs
(a) a definitive validation run that completes without `force_equal`,
and (b) a microbenchmark to confirm the full-grid kernel placement
doesn't regress concat perf for the common all-cores case (it
shouldn't, since `cores` already == full grid in most calls).

### Where the C++ patch lives

`/localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_program_factory.cpp`

Two hunks, ~30 LOC total. See the "Patch summary" block above for
the exact edits.

### Build sequence reminder (for the next person)

```
# After editing concat_program_factory.cpp:
cd /localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release
cmake --build . --target _ttnncpp.so -- -j 16
# Manual sync — ExternalProject install is unreliable here:
cp ttnn/_ttnncpp.so /localdev/kmabee/tt-xla/third_party/tt-mlir/install/lib/_ttnncpp.so
md5sum /localdev/kmabee/tt-xla/third_party/tt-mlir/install/lib/_ttnncpp.so \
       ttnn/_ttnncpp.so   # should match
```

If you touch anything in `tt_metal/` (not just `ttnn/`), also
`cp ../tt_metal/libtt_metal.so /localdev/kmabee/tt-xla/third_party/tt-mlir/install/lib/`
and `cp ttnn/_ttnn.so /localdev/kmabee/tt-xla/third_party/tt-mlir/install/lib/`.

## Proposed fixes (original plan, pre-attempt)

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
