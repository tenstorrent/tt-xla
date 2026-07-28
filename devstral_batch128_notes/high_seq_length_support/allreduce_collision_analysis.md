# Devstral DP+TP all_reduce hang — program-cache-collision / stale-semaphore analysis

Read-only investigation. Subject log: `/data/ssalice/temp/tt-xla/devstral_dptp_test.log`
(mesh [4,8], axis0=DP=4, axis1=TP=8, chunked prefill, opt_level=1).
Comparison log: `/data/ssalice/temp/tt-xla/devstral_test.log` (earlier run, decomposition fired).

All source paths below are under
`third_party/tt-mlir/src/tt-mlir/` (tt-mlir) or
`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/` (tt-metal).

---

## TL;DR verdict

- **Q1 — "byte-identical CCL graph → program-cache collision → stale-semaphore deadlock":**
  the *collision precondition* is **CONFIRMED**, but the stated *stale/baked-semaphore*
  causal step is **REFUTED for the reduce_scatter path**. Static reading of the RS override
  helper shows that on every cache hit **all** addresses — input/intermediate/output buffers
  **and** every GlobalSemaphore — are re-written into the kernel runtime args
  (`reduce_scatter_minimal_async_program.cpp:936-966`), the GlobalSemaphores are held alive in
  the cache entry, and the only compile-time-baked semaphore (`:1215`) is a fixed
  program-relative offset that stays valid. **There is no address baked-at-create-and-
  not-refreshed-on-hit** — so the "minted-on-miss, never-refreshed, goes stale" story does not
  hold here. This is corroborated empirically: a standalone `ttnn.all_gather` whose spec is
  **identical** to a graph-1 all_gather (hence a guaranteed cache hit) executes **successfully**
  in graph-2's warmup at log line 15053, immediately before the all_reduce hangs at 15165 — a
  direct counterexample to "any byte-identical CCL cache-hit deadlocks."
  The hang is real and is **specific to the all_reduce's internal reduce_scatter**; the most
  consistent remaining explanation is **cross-device asymmetric (re)allocation of the freshly
  allocated reduce_scatter intermediate buffer after graph-1's trace capture perturbed L1/DRAM**
  — each device refreshes its own address correctly, but the ring's peer writes assume symmetric
  allocation across the 8 TP devices. This is enabled/masked by the spec-only hash but is an
  allocation-symmetry issue, **not** a baked-semaphore issue. It remains **PLAUSIBLE, unproven**.
- **Major correction to the framing:** the "FUSED" `ttnn.all_reduce` is fused only at the
  MLIR/flatbuffer level. At tt-metal runtime it is **NOT** the monolithic
  `AllReduceAsyncDeviceOperation`; for this tensor it decomposes into
  **`ReduceScatterDeviceOperation` + `AllGatherDeviceOperation`** — the very device ops a
  D58 fix would target. So "D58 fixed RS/AG but not all_reduce" cannot by itself explain this
  hang: fixing RS/AG *would* protect the fused all_reduce on this path.
- **Q2 — why fused here but decomposed earlier:** **CONFIRMED**. tt-mlir commit
  `1d91fcf556` (#8961, "Reshape sub-4D all_reduce to 4D; drop reduce_scatter decomposition",
  merged 2026-07-06) **removed** the `TTNNAllReduceWorkarounds` RS+AG decomposition pattern
  and its `enable-all-reduce-workaround` option. The two runs differ because
  **libTTMLIRCompiler was rebuilt across #8961**, not because of an opt-level/flag flip.
  The old decomposition was gated by `allReduceWorkaroundEnabled`, never by `optimizationLevel`.

---

## Q1 — Does the deadlock mechanism apply to the fused `ttnn.all_reduce` in this log?

### (0) What the "fused" all_reduce actually is at runtime — CONFIRMED, reframes everything

The tt-mlir `AllReduceOp` runtime handler calls composite `ttnn::all_reduce`:
- `runtime/lib/ttnn/operations/ccl/all_reduce.cpp:36-37` → `::ttnn::all_reduce(input, clusterAxis, ...)`.
- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp:45-56` → calls
  `::ttnn::experimental::all_reduce_async(..., std::nullopt /*barrier*/, std::nullopt /*rs*/,
  std::nullopt /*ag*/, Sum, ...)` — the overload at
  `all_reduce_async.cpp:273`. **It never calls `ttnn::prim::all_reduce_async`** (the minimal
  `AllReduceAsyncDeviceOperation` at `all_reduce_async.cpp:453/485`).

So the `AllReduceAsyncDeviceOperation::compute_program_hash`
(`.../experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:105-124`)
is **for a code path this log does not exercise**. Do not rest the verdict on it. (For the
record: that hash passes `tensor_args` and no `buffer()->address()`, so it too is spec-only —
but it is not reached here.)

Path selection inside the line-273 overload for the hung op
(input `tensor<1x1x4096x12288xbf16>`, TILE, cluster_axis=1, num_devices along axis1 = 8):
- `finding_scatter_dim` (`all_reduce_async.cpp:32-53`): tile-normalized shape `[1,1,128,384]`,
  first reverse dim divisible by 8 is dim 3 → **dim = 3**.
- `composite_dim = 3` (`:308`).
- `use_composite_all_gather(input, 3)` (`composite_common.cpp:284-304`): TILE, gather dim = rank-1,
  `12288 % 32 == 0` → **false**.
- `use_composite_reduce_scatter(input, 3, axis=1)` (`composite_common.cpp:31-60`):
  `12288 % 8 == 0`, TILE, output `1536 % 32 == 0` → **false**.
- Branch guard `composite_all_gather || composite_reduce_scatter || (dim != composite_dim)`
  (`:312`) = `false` → takes the **RS + AG** branch (`:349`).
- rs/ag/barrier semaphores are all `nullopt` → the `else` arms run:
  `ttnn::reduce_scatter(...)` (`:366`) and `ttnn::all_gather(...)` (`:410`).

`ttnn::reduce_scatter` (`ccl/reduce_scatter/reduce_scatter.cpp:71,86`) re-checks
`use_composite_reduce_scatter` (false) → `ttnn::prim::reduce_scatter` =
**`ReduceScatterDeviceOperation`**. `ttnn::all_gather` (`ccl/all_gather/all_gather.cpp:72,77`)
re-checks `use_composite_all_gather` (false) → `ttnn::prim::all_gather` =
**`AllGatherDeviceOperation`**.

**CONFIRMED: the fused all_reduce on this shape = `ReduceScatterDeviceOperation` +
`AllGatherDeviceOperation` at metal runtime.**

### (a) Are GlobalSemaphores minted on cache-miss and baked, and NOT refreshed on hit? — CONFIRMED

`ReduceScatterDeviceOperation` program factory
(`.../ccl/reduce_scatter/device/reduce_scatter_program_factory.cpp`):
- `create_mesh_workload` (called on cache **miss**) mints 3 sync + 1 barrier GlobalSemaphores
  via `create_global_semaphore(...)` (`:41-49`), then bakes them into the kernels through
  `create_at(...)` / the ring/line builder (`:54-61`, `:116-138`).
- They are stored in `shared_variables_t` (`:140-143`).
- `override_runtime_arguments` (called on cache **hit**, `:148-181`) does **not** create new
  semaphores; it passes `shared_variables.barrier_semaphore` /
  `shared_variables.multidevice_semaphores` (`:175-176`) — the originals — and only updates the
  input/intermediate/output tensor addresses (`:177-179`).

So the task's stated premise — *semaphores minted on miss, baked in, not refreshed on hit* —
is **CONFIRMED** for the exact device op in play.

**Decisive nuance — the override helper refreshes everything, so nothing goes stale:** those
GlobalSemaphores are **held alive** by the cached program's `shared_variables`
(`reduce_scatter_program_factory.cpp:140-143`), so the baked L1 address stays reserved and valid.
More importantly, the actual override body
(`reduce_scatter_minimal_async_program.cpp:907-970`,
`ring_reduce_scatter_minimal_async_helper_override_runtime_arguments`) **re-writes every address
into the runtime args on each hit**: `input.buffer()->address()`, `intermed.buffer()->address()`,
`output.buffer()->address()`, and `semaphore.at(dir).address()` / `barrier_semaphore.address()`
(`:936-966`). In the builder (create-on-miss) these same values are *initial* runtime-arg values
(`:778-841`); the only compile-time (baked) kernel args are `TensorAccessorArgs(...)`
(buffer-type/sharding accessor metadata, `:622-624`, `:674-675`) — which are identical for
same-spec tensors — plus, in the line builder only, one fixed program-relative
`CreateSemaphore` offset (`:1215`). **No buffer or semaphore address is baked-at-create and left
un-refreshed on a hit.** This directly refutes the "stale baked semaphore" causal step for
reduce_scatter/all_gather. The residual risk is not staleness but **cross-device symmetric-
allocation divergence**: the fabric ring computes a peer's target from the (correctly refreshed)
local address assuming all ring peers allocated symmetrically; if the freshly allocated
intermediate lands at divergent addresses across the 8 TP devices after trace capture, the peer
writes miss and the receiving semaphore never signals — a hang the spec-only hash allows but does
not itself cause.

### (b) Does the hash include the input buffer address (the "D58" mitigation)? — NO (CONFIRMED for current tree)

Program hash keys, current tree:
- RS: `reduce_scatter_device_operation.cpp:108-121` hashes dim, num_links, cluster_axis,
  memory_config, intermediate mem cfg, topology, chunk/worker/buffer knobs, compute cfg,
  `use_l1_small_for_semaphores`, `subdevice_core_range_set`, and **`tensor_args`** — no
  `buffer()->address()`.
- AG: `all_gather_device_operation.cpp:152-163` — same shape, **no** buffer address.
- all_reduce_async (minimal, not on this path): `:113-123` — same, no buffer address.

Why "hash `tensor_args`" ≠ "hash the buffer address": a `Tensor`'s hash attributes are
`(storage, tensor_spec)` (`ttnn/api/ttnn/tensor/tensor.hpp:256-257`), and
`DeviceStorage::attribute_values()` returns an **empty tuple**
(`ttnn/api/ttnn/tensor/storage.hpp:204-205`). So a Tensor hashes to its **spec only**
(shape/dtype/layout/memory-config) — the runtime L1/DRAM buffer address contributes nothing.
**Two same-spec tensors on different physical buffers produce identical program hashes.**
`hash_operation` = `hash_objects_with_default_seed(type_hash, objects...)`
(`ttnn/api/ttnn/operation.hpp:24-26`).

D58 (buffer-address-in-hash) status: **not present anywhere in the current tree** for RS, AG,
or all_reduce; `git log` of `reduce_scatter_device_operation.cpp` shows **no** buffer-address-hash
commit; **no git stash** exists in tt-xla, tt-mlir, or tt-metal. I therefore **cannot confirm
from evidence that D58 was ever applied to RS/AG at log time** — I can only reason conditionally,
as instructed. Note the internal inconsistency this exposes: because the fused all_reduce
*decomposes to RS+AG*, a D58 fix on RS/AG **would have protected the fused all_reduce too**. So
either D58 was not actually effective at log time (consistent with what I find), or the causal
story is not simply "D58 missing on all_reduce."

### (c) Log confirmation: phase-1 succeeds, only the byte-identical phase-2 op hangs — CONFIRMED

Executed `ttnn.all_reduce` ops (grep `Executing operation:.*ttnn.all_reduce`, `devstral_dptp_test.log`):

| line  | loc                | graph / pass                                  | result |
|-------|--------------------|-----------------------------------------------|--------|
| 8524  | dot.263_all_reduce_4d | graph-1 warmup (after "Trace cache miss" @8385) | OK |
| 8552  | dot.338            | graph-1 warmup                                 | OK |
| 8662  | dot.514            | graph-1 warmup                                 | OK |
| 8688  | dot.589            | graph-1 warmup                                 | OK |
| 8831  | dot.263            | graph-1 trace-capture pass                     | OK |
| 8859  | dot.338            | graph-1 trace-capture pass                     | OK |
| 8969  | dot.514            | graph-1 trace-capture pass                     | OK |
| 8995  | dot.589            | graph-1 trace-capture pass                     | OK |
| 15165 | dot.269_all_reduce_4d | graph-2 warmup (after "Trace cache miss" @15029) | **HANG** |

Log tail: line 15165 executes `%61 = "ttnn.all_reduce"(%60) <{cluster_axis=1, reduce_type=sum}>`,
then `error | Metal | Timeout detected` and `TT_THROW: TIMEOUT: device timeout, potential hang
detected` at `system_memory_manager.cpp:757`. So the hang is the **first all_reduce of graph-2's
warmup run**, i.e. the first time this op is hit *after* graph-1's trace was captured.

Graph identity (device-op-relevant content): the module dumps of graph-1 (`:6488`) and graph-2
(`:13345`) differ **only** in the loc debug label (`dot.263`→`dot.269`) and the layout-alias name
(`ttnn_layout60`→`ttnn_layout63`); the inline executed forms (`:8524` vs `:15165`) carry an
**identical** layout `memref<128x384x!ttcore.tile<32x32,bf16>, dram, interleaved>` and identical
attributes. Neither loc nor alias name feeds the program hash. **CONFIRMED byte-identical for
hashing.** Therefore graph-2's all_reduce (its internal RS+AG) **necessarily hits** graph-1's
cached programs.

Counts (whole log): `ttnn.all_reduce`=25 text matches, `ttnn.reduce_scatter`=0,
`ttnn.all_gather`=7, `all_reduce_4d`=33 — matching the brief. (25/7 are module-dump + execution
text matches, not 25 distinct executed all_reduces; only the 9 rows above are executed.)

**Direct counterexample (verified), important:** a **standalone** `ttnn.all_gather` executes in
graph-2's warmup at line 15053 and **succeeds**, immediately before the all_reduce hangs at 15165.
Its spec is **identical** to graph-1's all_gathers at lines 8413/8720 —
`ttnn.all_gather"(%3) <{all_gather_dim=2, cluster_axis=1}> : tensor<32x128x1536xbf16, ...
memref<128x48x!ttcore.tile<32x32,bf16>, dram, interleaved>>` byte-for-byte — so it is a
**guaranteed cache hit** in graph-2, post-trace, and it runs fine. This refutes "any byte-identical
CCL cache-hit deadlocks" and localizes the failure to the all_reduce's internal **reduce_scatter**
specifically (the op that additionally allocates and cross-writes an intermediate buffer across the
ring). The earlier log `devstral_test.log` is also verified (not assumed): `all_reduce`=0,
`reduce_scatter`=8, `all_gather`=11, `all_reduce_4d`=24 — decomposition present.

### VERDICT (Q1)

- **CONFIRMED (collision precondition):** spec-only hashing (empty `DeviceStorage` attributes) ⇒
  byte-identical graphs collide; graph-2's all_reduce → RS+AG **hits** graph-1's cached programs;
  semaphores are minted on miss and baked in; phase-1 all_reduces succeed and the first
  byte-identical phase-2 all_reduce is the hang site.
- **REFUTED (the stated causal step, for this path):** "GlobalSemaphores minted on miss and
  **not refreshed** on hit → stale reuse → deadlock." The RS override helper re-writes every
  buffer **and** semaphore address on each hit (`reduce_scatter_minimal_async_program.cpp:936-966`),
  the semaphores are held alive (`reduce_scatter_program_factory.cpp:140-143`), and no address is
  baked-and-not-refreshed. Empirically, a byte-identical all_gather cache-hits post-trace and
  **succeeds** (log 15053 vs 8413/8720). So the specific mechanism in the hypothesis is not what
  hangs this run.
- **PLAUSIBLE (not proven), the surviving mechanism:** a **cross-device symmetric-allocation
  divergence** of the reduce_scatter **intermediate** buffer after graph-1's trace capture. The
  spec-only hash lets graph-2 reuse graph-1's program with per-device-refreshed local addresses,
  but the ring's peer writes assume all TP peers allocated symmetrically; post-trace divergence
  ⇒ peer writes miss ⇒ receiving semaphore never signals ⇒ deadlock. Forcing a per-buffer-address
  cache miss (the "D58" idea) would mask this by recompiling per allocation, but the root issue is
  allocation symmetry, not the program hash. A plain reduce_scatter kernel/fabric hang at this
  shape under trace (unrelated to caching) is also not excluded.
- **What would settle it definitively** (none present in this log, which is tt-xla `RuntimeTTNN`
  DEBUG level, not tt-metal `LogOp` trace level):
  1. tt-metal debug log showing `ReduceScatterDeviceOperation::compute_program_hash is called`
     plus program-cache **hit/miss** per op per device — to prove graph-2's RS/AG hit and whether
     any device diverged (hit on some, miss on others).
  2. Per-device dump of the reduce_scatter **intermediate** buffer L1/DRAM address (and the
     input/output addresses) across all 8 TP devices in graph-2's warmup — to prove/deny the
     symmetric-allocation divergence. A dump of the GlobalSemaphore addresses at create() vs the
     override-written values would confirm they are refreshed (expected: yes) and rule the
     semaphore path out definitively.
  3. Causal A/B: re-run with the program cache disabled (or graph-2 forced to recompile, e.g.
     buffer-address added to the RS/AG `compute_program_hash`). If the hang disappears, the
     collision/symmetry mechanism is confirmed causally; if it persists, the hang is a
     reduce_scatter kernel/fabric issue at this shape independent of caching.

---

## Q2 — Why did the decomposition NOT fire (fused all_reduce) here, when the earlier log decomposed?

### Current tree (what produced `devstral_dptp_test.log`)

`lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp:430-503`:
- Inside `if (decompositionWorkaroundsEnabled)` (`:431`) the pattern set registers
  `TTNNCollectiveReshapeWorkaround<ttnn::AllReduceOp>` (`:437-438`),
  `TTNNAllGatherWorkarounds` (`:439`), and
  `TTNNCollectiveReshapeWorkaround<ttnn::ReduceScatterOp>` (`:440-441`).
- **There is no `TTNNAllReduceWorkarounds` (RS+AG decomposition) pattern anywhere in the tree**
  (grep returns only an unrelated match in `ShardyCCLToStableHLOCCLPatterns.cpp`).
- `TTNNCollectiveReshapeWorkaround` (`.../Decomposition/CollectiveReshapeOpRewritePattern.h`)
  only **pads/reshapes to 4D** and re-emits a native `ttnn.all_reduce`; it does not lower to
  RS+AG. This is exactly why the log shows fused `ttnn.all_reduce` **and** the `_all_reduce_4d`
  loc suffix simultaneously.
- The gate `decompositionWorkaroundsEnabled` comes from `applyDecompositionWorkarounds`
  (default **true**, `include/.../OpValidator.h:32`; wired in `OpValidator.cpp:28-29`;
  option in `Passes.td:43` / `TTNNPipelines.h:290`) and is **independent of `optimizationLevel`**.
  `optimizationLevel` only guards unrelated patterns (`:490` PagedUpdateCache < 2, `:499` SDPA cfg).

So at opt_level=1 nothing turns a decomposition "off" — **the decomposition code no longer
exists** in this compiler.

### The cause of the difference — CONFIRMED: a compiler rebuild across #8961

tt-mlir commit **`1d91fcf556` — "[TTNN] Reshape sub-4D all_reduce to 4D; drop reduce_scatter
decomposition (#8961)"** (Milan Topalovic, 2026-07-06), which is an **ancestor of the current
tt-mlir HEAD** (`git merge-base --is-ancestor 1d91fcf556 HEAD` → yes). Its diff to
`TTNNWorkaroundsPatterns.cpp`:
- removes `#include ".../AllReduceReshapeOpRewritePattern.h"` and `.../ReduceScatterOpRewritePattern.h`;
- **deletes** `class TTNNAllReduceWorkarounds : public OpRewritePattern<ttnn::AllReduceOp>` and its
  `rewriter.create<ttnn::ReduceScatterOp>(...)` / `rewriteAsAllGatherLocalReduce(...)` body;
- **deletes** `patterns.add<TTNNAllReduceWorkarounds>(&getContext())`;
- replaces the separate all_reduce/RS reshape workarounds with the templated
  `TTNNCollectiveReshapeWorkaround<AllReduceOp>` / `<ReduceScatterOp>`.
Commit message: *"Remove the TTNNAllReduceWorkarounds reduce_scatter + all_gather decomposition
and its enable-all-reduce-workaround option, so all_reduce now lowers to native ttnn.all_reduce."*

### Pre-#8961 gating (matches the line numbers in the brief)

`git show 1d91fcf556^:...TTNNWorkaroundsPatterns.cpp`:
- `class TTNNAllReduceWorkarounds` at **line 346**;
- registered at **line 687** `patterns.add<TTNNAllReduceWorkarounds>(&getContext())`, inside
  `if (allReduceWorkaroundEnabled)` (**line 686**), itself inside
  `if (decompositionWorkaroundsEnabled)` (**line 632**).

Answering the brief's sub-questions:
- **Registered unconditionally or gated?** Gated — behind the **`allReduceWorkaroundEnabled`**
  option (the `enable-all-reduce-workaround` pass option), nested inside
  `decompositionWorkaroundsEnabled`. Not unconditional.
- **Could opt_level=1 make it false?** No. The gate was `allReduceWorkaroundEnabled`, a distinct
  boolean option; `optimizationLevel` never controlled it. (And in the current build the pattern
  is gone entirely, so opt-level is irrelevant.)
- **Does the reshape-to-4D run in a different always-on block?** Yes — the CollectiveReshape /
  reshape-to-4D workaround runs directly under `decompositionWorkaroundsEnabled` (default true),
  not under the removed `allReduceWorkaroundEnabled`. That is why `all_reduce_4d` naming appears
  with no RS decomposition.

### VERDICT (Q2) — CONFIRMED

The two runs differ because **libTTMLIRCompiler was rebuilt across commit #8961**. The earlier
`devstral_test.log` was produced by a **pre-#8961** compiler (or with
`enable-all-reduce-workaround` on a pre-#8961 build) that decomposed
`all_reduce → reduce_scatter + all_gather` (hence `reduce_scatter=25`, `all_reduce=0`). The later
`devstral_dptp_test.log` was produced by a **post-#8961** compiler that emits a **native fused
`ttnn.all_reduce`** merely reshaped to 4D (hence `all_reduce=25` fused, `all_reduce_4d` naming,
`reduce_scatter=0`). It is **not** an opt-level or runtime-flag difference — the decomposition
pattern and its `enable-all-reduce-workaround` option were removed from the compiler.

I cannot from these artifacts alone prove *which* `.so` each log linked (no build manifest in the
logs), but: (i) the current tree is unambiguously post-#8961; (ii) post-#8961 has no decomposition
code, so `devstral_test.log`'s `reduce_scatter=25` is only explicable by a pre-#8961 build; hence a
rebuild/compiler-swap between the two runs is the only consistent explanation. To settle the exact
`.so`: diff the `pjrt_plugin_tt.so` / `libTTMLIRCompiler.so` build timestamps or the tt-mlir commit
each run linked (e.g. a `git rev-parse HEAD` recorded at build time), or re-run both configs against
the current build and confirm both now emit fused `all_reduce`.

---

## Consolidated label sheet

| Claim | Label | Key citation |
|-------|-------|--------------|
| Fused `ttnn.all_reduce` = RS+AG device ops at runtime (this shape) | CONFIRMED | all_reduce.cpp:45; all_reduce_async.cpp:273,312,366,410; reduce_scatter.cpp:86; all_gather.cpp:77 |
| The minimal `AllReduceAsyncDeviceOperation` is NOT on this path | CONFIRMED | all_reduce.cpp:45 (calls line-273 overload, not `ttnn::prim::all_reduce_async`) |
| Program hash is spec-only (buffer address excluded) | CONFIRMED | tensor.hpp:256-257; storage.hpp:204-205; operation.hpp:24-26 |
| RS/AG/all_reduce_async hashes contain no `buffer()->address()` | CONFIRMED (current tree) | reduce_scatter_device_operation.cpp:108-121; all_gather_device_operation.cpp:152-163 |
| Semaphores minted on miss and baked into runtime args | CONFIRMED | reduce_scatter_program_factory.cpp:41-49; ..._program.cpp:778-841 |
| Semaphores + all buffer addresses REFRESHED on every cache hit | CONFIRMED | reduce_scatter_minimal_async_program.cpp:936-966 (override helper) |
| No address baked-at-create-and-not-refreshed (only TensorAccessorArgs baked) | CONFIRMED | ..._program.cpp:622-624,674-675,1215 |
| Semaphores held alive in cache entry (address stays valid) | CONFIRMED | reduce_scatter_program_factory.cpp:140-143 |
| Graph-1 and graph-2 all_reduce byte-identical for hashing | CONFIRMED | devstral_dptp_test.log:6488 vs 13345; 8524 vs 15165 |
| Phase-1 all_reduces pass; first byte-identical phase-2 op hangs | CONFIRMED | devstral_dptp_test.log:8524-8995 OK, 15165 HANG + timeout tail |
| Byte-identical all_gather cache-hits post-trace and SUCCEEDS (counterexample) | CONFIRMED | devstral_dptp_test.log:15053 vs 8413/8720 (identical spec) |
| Earlier log decomposed (all_reduce=0, reduce_scatter=8) | CONFIRMED | devstral_test.log grep |
| D58 buffer-address-hash present in RS/AG at log time | UNVERIFIABLE from evidence (no stash, no commit) | git stash empty (all 3 repos); no buffer-addr-hash commit on RS device op |
| "GlobalSemaphore minted-on-miss, not-refreshed-on-hit → stale → deadlock" | REFUTED (this path) | override refreshes all addrs (936-966); all_gather counterexample |
| Cross-device intermediate symmetric-allocation divergence → deadlock | PLAUSIBLE (not proven) | needs per-device intermediate address dump; see Q1 "what would settle it" |
| Fused-vs-decomposed difference = compiler rebuild across #8961 | CONFIRMED | tt-mlir 1d91fcf556 (#8961); merge-base ancestor of HEAD; pre-#8961 gate at :686-687 |
| Difference is an opt-level / runtime-flag effect | REFUTED | gate was `allReduceWorkaroundEnabled`, not `optimizationLevel`; pattern removed entirely |
