# Bringing up fused MoE decode (`tt_moe_fused`) on a 4×8 Blackhole galaxy — status & handoff

This is the consolidation/handoff doc for getting
`tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy` (GPT-OSS-120B, fused
MoE decode, `TT_MOE_FUSED_BACKEND_NAME`) running on a 4×8 Blackhole galaxy mesh.

Adding the test turned into a genuine bringup: the fused `tt.moe_decode` path (dense bmm in
prefill; `all_to_all_dispatch` + `moe_compute` in decode) had not been exercised on a **2D
mesh where the EP cluster axis is smaller than the whole mesh** (here EP-8 on "model",
DP-4 on "batch"). It hits a *sequence* of independent gaps across tt-xla, tt-mlir, and
tt-metal — each revealed only after the previous is fixed.

## The test

- `test_gpt_oss_120b_tp_moe_fused_galaxy` — GPT-OSS-120B, `experts_implementation=tt_moe_fused`,
  4×8 mesh `("batch","model")`, `cluster_axis=1` (EP-8 on "model"), DP-4 on "batch",
  `bfp_bf8` weights, trace disabled, `optimization_level=1`.
- Validated at `--num-layers 2` (the idiomatic reduced-layer bringup used by the other
  heavy galaxy MoE tests). Full-model runs are additionally bottlenecked by slow
  single-threaded host-side weight packing (experts are replicated 4× across the "batch"
  axis, since the fused decode requires single-axis EP).
- Currently a **known failure** at issue #4 below. Marked `"skip": true` in
  `.github/workflows/perf-bench-matrix.json` with a pointer to this doc.

## Issue chain

| # | Symptom | Layer | Status | Detail |
|---|---|---|---|---|
| 1 | `tt.mark_argument` ui32 operand → i32 result; folded → reshape element-type mismatch → `Error code: 13` at compile | tt-xla frontend pass | ✅ fixed | `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md` |
| 2 | decode `input_ids` placeholder i32 vs runtime uint32 → `to_layout` device→device ROW_MAJOR typecast FATAL | tt-xla test / decode util | ✅ fixed | see below |
| 3 | `expert_mapping` first dim = cluster-axis (8) vs required mesh-devices (32) → `all_to_all_dispatch_metadata` FATAL | tt-mlir lowering | ✅ fixed | `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`, background: `MOE_ALL_TO_ALL_DISPATCH_EXPLAINER.md` |
| 4 | `typecast` rejects 8-byte sharded page (top-4 routing tensor) — needs 16-byte L1 alignment | tt-mlir layout decomposition | ✅ fixed | this doc, below |
| 5 | Decode collectives are 1D-fabric-only, but the galaxy was forced to `FABRIC_1D` (Linear); the a2a dispatch has no 2D-mesh path and `moe_compute` rejects the Mesh fabric default → deadlock/FATAL | tt-metal kernel + tt-mlir + tt-xla | ✅ fixed | this doc, below |
| 6 | a2a dispatch omits the `convert_2d_to_1d_topology` step its sibling CCL ops all apply → runs `Mesh` on the 2D fabric | tt-metal (`all_to_all_dispatch_metadata.cpp`) | ✅ fixed (necessary, not sufficient) | this doc, below |
| 7 | Even with a2a on `Linear`, the a2a dispatch collective does not complete on the galaxy → `moe_compute`'s first `finish` (draining it) deadlocks at `create_global_semaphore`/`read_completion_queue` | tt-metal fabric / device | ✅ resolved by config (see below) | this doc, below + "Current status" |
| 8 | Metadata drain-core SPLIT + reshard: a2a drains persistent metadata to `(0,0)` but `moe_compute` reads it on `(11,9)` → compiler inserts a `to_memory_config` reshard of the persistent metadata → collective deadlock | tt-mlir | ✅ fixed | "Current status" §hang #1 |
| 9 | `expert_mapping` **VALUE** is the axis-local owner `[0,4)`, but the a2a kernel needs the **global mesh coord** `[0,32)` → tokens misroute → 29/32-device moe_compute tilize `CWFW` starvation hang | tt-mlir | ✅ fixed (necessary, not yet sufficient) | `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` (Follow-up) |

Fixes #1–#6 are each validated as *necessary* (each cleared its blocker and advanced
execution). #7 was **resolved by a configuration change, not a kernel fix**: the galaxy was
switched from `FABRIC_2D` + `cluster_axis=1` (EP-8) to **`FABRIC_1D_RING` + `cluster_axis=0`
(EP-4)** — tt-metal's actually-tested MoE path — after which the a2a dispatch completes.
Two further device hangs then surfaced on the tested path: **#8** (the metadata drain-core
reshard, fixed) and **#9** (the expert-mapping value, fixed). See **Current status** below —
the decode still hangs *after* both fixes, at a later point that is not yet localized.

## Current status (updated 2026-07-06)

**Config resolution of #7.** Every tt-metal full-model MoE test (gpt_oss, deepseek_v3) uses
**(4,8) + `FABRIC_1D_RING` + `cluster_axis=0` (EP-4)**; `FABRIC_2D` + `cluster_axis=1` (EP-8) appears
only in 1D op-level tests and was never a validated full-mesh path. The #7 a2a NSW hang was specific
to that untested choice. tt-xla was switched to the tested path: `client_instance.cc`
(galaxy→`FABRIC_1D_RING`), `register_tt_moe_backend(cluster_axis=0)`, the shard spec EP-shards experts
along "batch" (axis 0), and `TTNNResolveComposites.cpp` sets moe_compute topology=`Ring`. On this path
the a2a dispatch completes (isolated + chained native repros pass). Two device hangs then surfaced:

**Hang #1 — metadata drain-core SPLIT + reshard (issue #8, FIXED, IR-verified).** tt-xla's a2a wrote its
PERSISTENT metadata to core `(0,0)` while `moe_compute` read it on `(11,9)` (`get_moe_tilize_drain_core`),
so the compiler inserted a `to_memory_config` reshard of the persistent metadata that deadlocks the
collective (proven by a native drain-core-split repro: split hangs, aligned passes). Three tt-mlir
changes eliminate the reshard:
- `TTNNWorkaroundsPass.cpp` `createMoeComputeOpOperandsWorkarounds` — pin moe_compute indices/scores
  operands to **L1 HeightSharded** (were RowMajor+dtype only → defaulted to DRAM → "Only L1 buffers can
  have an associated circular buffer" CB error once the reshard was removed).
- `TTNNAllocateDistributedOpBuffers.cpp` — align the a2a persistent-metadata drain core to
  moe_compute's `get_moe_tilize_drain_core` (stash via attr `ttnn.moe_metadata_drain_core`, read by
  `AllToAllDispatchMetadataOp::allocateBuffers` in `TTNNOps.cpp`); op_model-guarded.
- `TTNNDeduceMoEComputeLayouts.cpp` — remove `reshardTilizeInputToDrainCore`.
Verified via dry_run IR: decode metadata is L1 `(11,9)` end-to-end, zero `to_memory_config`.

**Hang #2 — expert-mapping VALUE axis-local vs global coord (issue #9, FIXED, IR-verified).** With the
reshard gone, the decode still hung: watcher on the native decode flatbuffer (via `ttrt`) showed
**29/32 devices stuck at the moe_compute `tilize_reader` `CWFW`** (circular-buffer wait) — most devices'
experts never got their dispatched tokens. Root cause: `synthesizeMoeDecodeExpertMapping`
(`StableHLOToTTIRPatterns.cpp`) emitted the **axis-local owner** `e/expertsPerDevice ∈ [0,4)` as the
mapping value, but the a2a dispatch kernel uses that value **directly as the global mesh coord**
`target_device ∈ [0,32)`. Fixed to emit the global coord of the owner replica in each source's own
cluster group (rows now differ per source). **Full details, kernel-contract evidence, A/B repro, and
the fix formula are in `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` → "Follow-up" section.**

**Residual: the decode STILL hangs after both fixes (open).** With #8 + #9 in place, the 1-layer run
compiles + executes the prefill, compiles + submits the fused decode, and reaches program-time ~969s,
then **deadlocks on-device**: a single `FDMeshCommandQueue::read_completion_queue` /
`completion_queue_wait_front` reader stuck in `Cluster::read_from_sysmem`. The old 32-thread
`completion_queue` signature is gone and it gets further, so #9 is real progress, but something in the
fused decode still doesn't complete. **Two hypotheses:** (a) the mapping fix cleared the tilize `CWFW`
and a downstream fused-decode collective (`selective_reduce_combine` / post-moe `all_gather`
`cluster_axis=0`) now hangs; (b) the tilize still stalls for another reason. **Next step:** run the
fixed decode flatbuffer (`modules/fb_*_1lyr_*_g1_*.ttnn`) under `ttrt` + waypoint watcher (the method
that localized hang #2) and compare the stuck kernel/core.

### Diagnostic notes (learned this cycle — avoid re-tripping)

- **A frozen log ≠ a hang here.** Each decode graph takes ~9 min to *compile* in tt-mlir (the fused
  120B lowering), so the log goes silent for minutes between graphs while the device sits idle. Confirm
  a real hang via gdb: `completion_queue_wait_front` / `read_from_sysmem` reader stuck = device hang;
  `clock_nanosleep` executors + advancing program-time = just slow compile. `%CPU` from `ps` is a
  *lifetime average* — a thread showing 95% may actually be parked in a futex; re-sample the live stack.
- **`kill -0 <pid>` returns success on a zombie.** A finished-but-unreaped build process (`ps` shows
  `ZN <defunct>`) will make a `while kill -0 PID` wait-loop hang forever. Check `ps -o stat` or just
  re-run `cmake --build build` (fast if already done).
- **The run compiles ~5 executables, not 2.** There are only 2 graph *types* (dense prefill-shaped and
  fused single-token decode), but the benchmark drives warmup + timed + PCC-reference passes and
  torch_xla lazily recompiles per distinct trace (KV-cache position/constants differ), so decode steps
  don't cache-hit — a separate *perf* issue (recompiling decode per step) worth fixing after correctness.

## Fixes landed (keep these regardless)

- **#1 — `tt.mark_argument` uint32 (tt-xla).**
  `pjrt_implementation/src/api/module_builder/frontend_passes/shlo_input_role_propagation.cc`:
  when folding `tt.mark_argument`, if the op's result type differs from its operand type,
  insert a `stablehlo.convert` instead of forwarding the operand raw (which produced
  ill-typed IR for uint32 inputs). Inert for all matched-type inputs.

- **#2 — decode token-id dtype (tt-xla).**
  `tests/benchmark/llm_utils/decode_utils.py`, `LLMSamplingWrapper.forward`: cast the
  device `argmax` token ids to **uint32** on the XLA device so the compiled decode graph's
  `input_ids` placeholder matches the dtype the runtime materializes (device-produced
  32-bit ints are uint32; a `.to(int32)` is elided at the buffer level). CPU keeps int64
  (torch embedding requires Long/Int). This depends on #1 (uint32 must round-trip through
  `mark_argument`).

- **#3 — `expert_mapping` row count (tt-mlir).**
  `third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`,
  `synthesizeMoeDecodeExpertMapping` + caller: the mapping is `[devices, experts]` indexed
  by the *global* linearized mesh coordinate, so its row count must be the **total mesh
  device count (32)**, not the cluster-axis device count (8). Decoupled the row count
  (total mesh devices) from the expert count (`experts_per_device × cluster_devices`);
  values unchanged. Realizes the previously-TODO 2D case.

- **#4 — uint16 memory-config bounce over-applied to (de)shards (tt-mlir).**
  `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`,
  `createToMemoryConfigOpIfNeeded`: the uint16→uint32→uint16 bounce that works around
  `ttnn.copy` lacking uint16 (tt-metal#41689) was applied to **every** uint16
  `to_memory_config`, including sharded↔interleaved (de)shards. `ttnn.copy` is only used for
  interleaved↔interleaved buffer changes; (de)shards go through the shard kernels, which
  handle uint16 directly. Worse, the leading ui16→ui32 `ttnn.typecast` ran on the sharded
  input and hit the 8-byte-page assert (below). Fix: gate the bounce on
  `dataType == UInt16 && !inputSharded && !outputSharded`, i.e. only the true `ttnn.copy`
  path. Verified in the decode-graph `ttnn` dump (the failing `%typecast(indices #86 sharded)`
  is gone) and at runtime (execution passes the former FATAL point into live decode). Two
  earlier attempts were rejected: a `getOperandsWorkarounds` on `TypecastOp` (wrong pass
  order, never fired); a `from_device`→`to_device` host round-trip (cleared the FATAL but
  **deadlocked** the persistent moe pipeline — the mid-pipeline device↔host read stalls the
  a2a/moe_compute semaphore sync).

## Issue #4 (FIXED): `typecast` rejects 8-byte sharded page

### Symptom
```
TT_FATAL: Typecast operation requires sharded input tensor page size (8 bytes) to be aligned to L1 (16 bytes)
  ttnn/cpp/ttnn/operations/copy/typecast/device/typecast_device_op.cpp:119
  (validate_on_program_cache_miss)
```
Reached during decode warmup, via
`tt::runtime::ttnn::operations::layout::run(TypecastOp)` → `ttnn::typecast`.

### The check
`typecast_device_op.cpp:112-120` — only for **sharded** inputs:
```cpp
if (input_tensor.is_sharded()) {
    const uint32_t l1_alignment = hal::get_l1_alignment();          // 16
    const uint32_t page_size_bytes = input_tensor.buffer()->page_size();  // 8
    TT_FATAL(page_size_bytes == input_tensor.buffer()->aligned_page_size(), ...);
}
```

### Why the page is 8 bytes
The MoE dispatch (`all_to_all_dispatch_metadata`) requires `expert_indices` in **uint16**
and `expert_scores` in **bfloat16** (op validate). GPT-OSS routes **top-4** experts, so a
per-token routing row has 4 elements = **4 × 2 = 8 bytes**. A `typecast` producing/consuming
that tensor while it is **sharded** has an 8-byte page, which is below the 16-byte L1
alignment the op enforces for sharded inputs. (The exact operand — indices vs scores, and
the cast direction — should be confirmed from the decode-graph `ttnn` dump when fixing.)

### The failing op (confirmed from the `ttnn` decode-graph dump)
The router `topk` and `all_to_all_dispatch_metadata` produce the top-4 routing tensors
`[1,64,4]` uint16 (`indices`) as L1 **`<height_sharded>`** with a **4×2 = 8-byte** page
(layout `memref<64x4xui16, #l1>`). The `ttnn.typecast` that converts those indices
(uint16↔uint32, for `moe_expert_token_remap` / the dispatch dtype juggling) receives that
sharded 8-byte-page input and trips the assert.

### Root cause (confirmed) and the fix
The `ttnn.typecast` is NOT in the layout analysis — it comes from
`TTNNDecomposeLayouts.cpp::createToMemoryConfigOpIfNeeded`, which wraps every uint16
`to_memory_config` in a ui16→ui32→ui16 bounce (working around `ttnn.copy` lacking uint16,
tt-metal#41689). For the a2a-output indices → moe_compute drain-core reshard, the decompose
splits sharded→sharded into deshard(→DRAM interleaved)+shard; the **deshard** step's leading
ui16→ui32 typecast runs on the still-sharded `#86` (`memref<64x4xui16, #l1>, height_sharded`,
8-byte page) and FATALs. The bounce is unnecessary there: sharded↔interleaved (de)shards use
the shard kernels (uint16-capable), not `ttnn.copy`. Fix = gate the bounce on
`!inputSharded && !outputSharded` (see "Fixes landed" #4). Confirmed cleared in IR + runtime.

### Rejected attempts (for the record)
1. `getOperandsWorkarounds` on `TypecastOp` — wrong pass order (workaround runs before the
   layout analysis shards the input), never fired.
2. `from_device`→`to_device` host round-trip inside `createToMemoryConfigOpIfNeeded` — cleared
   the typecast FATAL (execution passed the old failure point) but **deadlocked**: a
   device→host→device read in the middle of the persistent a2a/moe_compute pipeline stalls the
   cross-device semaphore sync (`FromDeviceOp` → `read_completion_queue` never returns).

## Issue #5 (FIXED): decode collectives are 1D-fabric-only; galaxy was forced to FABRIC_1D

### Symptom
With #1–#4 fixed, the 2-layer run compiled both graphs and entered live decode, then **hung**
in `operations::ccl::run(AllocateMoeComputeSemaphoreOp)` → `MeshCommandQueue::finish` →
`read_completion_queue` (stable across samples, host busy-polling → deadlock, not a crash).

### Root cause
The decode collectives (`all_to_all_dispatch_metadata`, `moe_compute` combine) are written for
**1D fabric** (Linear/Ring). The galaxy is force-overridden to plain `FABRIC_1D` →
`Topology::Linear` (`client_instance.cc`), because `computeMeshFabricConfig` classifies the 4×8
as both-axis ring → `FABRIC_1D_RING`, which tt-metal's TopologyMapper reads as a 2D torus
(`TORUS_XY`) and rejects (the galaxy has no physical both-axis wrap). But plain `FABRIC_1D`
gives only **1D** connectivity, whereas the decode dispatch multicasts **per-axis** using 2D
directional (N/S/E/W) fabric connections — so the EP-axis (E/W) sends index unopened connection
slots and deadlock (documented "infinite hang" in `moe_utils.hpp`). The right fabric is
`FABRIC_2D` → `Topology::Mesh`: 2D routing, **no wrap required**, matching the galaxy and
opening the directional connections. The per-axis multicast then runs its non-wrapping (Linear)
algorithm on each Mesh axis (endpoint-elided).

### Fix landed (tt-mlir + tt-xla; a2a topology handled by #6)
- **a2a topology (see Issue #6):** the a2a op has no `topology` attribute and its 1D per-axis
  kernels rejected the Mesh fabric default. The idiomatic fix is #6 — add
  `convert_2d_to_1d_topology` in `all_to_all_dispatch_metadata.cpp` so the a2a runs `Linear` on
  the 2D fabric. (An earlier exploratory approach relaxing the kernel `static_assert` in
  `moe_utils.hpp` / `writer_all_to_all_dispatch_metadata.cpp` to accept Mesh was **reverted** —
  superseded by #6.)
- **tt-mlir (`TTNNResolveComposites.cpp`):** `moe_compute` DOES have a `topology` attr but it
  was passed null → resolved to the fabric default (Mesh) → rejected by the combine
  (`moe_compute_device_operation.cpp:510` allows only Linear/Ring). Pin it to
  `ttcore::Topology::Linear` (the op's documented override for a Mesh/Torus fabric default).
- **tt-xla (`client_instance.cc`):** galaxy override now returns `FABRIC_2D` instead of
  `FABRIC_1D`.

Validated: with all three, device init uses FABRIC_2D (no torus rejection), the a2a kernel
builds and runs, and `moe_compute` passes the `:510` topology check — execution advances **into
the `moe_compute` combine**, past every prior #5 blocker, before hitting #6.

Note: both the a2a (via #6's `convert_2d_to_1d_topology`) and `moe_compute` (via its explicit
`topology` attr set to `Linear`) run the non-wrapping/line path on the galaxy axes — consistent,
just reached differently because the a2a op exposes no topology attribute while `moe_compute` does.

## Issue #6 (FIXED): a2a dispatch missing the 2D→1D topology collapse

### Root cause
`all_to_all_dispatch_metadata` is a per-axis (1D) CCL op. Its siblings (all_gather, all_reduce,
reduce_scatter, all_to_all_async_generic, …) all call
`::ttnn::ccl::convert_2d_to_1d_topology(topology_)` after `get_usable_topology(...)` to collapse
a 2D fabric to its 1D per-axis topology (Mesh→Linear, Torus→Ring). `all_to_all_dispatch_metadata`
was the **only** such op that omitted it (`all_to_all_dispatch_metadata.cpp:38`), so on the
FABRIC_2D galaxy it ran with `topology == Mesh` while its kernels are 1D per-axis.

### Fix landed
`ttnn/.../all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.cpp`: add
`topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);` after `get_usable_topology`,
matching the sibling CCL ops. This is a tt-metal **host** change (`_ttnncpp.so`); rebuild it with
`ninja -C <tt-metal>/build_Release _ttnncpp.so` and copy `build_Release/ttnn/_ttnncpp.so` over
`third_party/tt-mlir/install/lib/_ttnncpp.so` (the `build_Release/lib/` copy is stale — use the
`ttnn/` one). With this, the a2a runs `Linear` on the 2D fabric like moe_compute.

Note: this superseded the earlier exploratory tt-metal kernel edits (making the atomic-inc accept
Mesh) — those were **reverted**; the idiomatic fix is the op-level 2D→1D collapse, not per-kernel
Mesh support.

## Issue #7 (RESOLVED BY CONFIG — see "Current status"): a2a dispatch does not complete on the galaxy even as Linear

> **Resolution (2026-07-06):** #7 was specific to the untested `FABRIC_2D` + `cluster_axis=1` (EP-8)
> choice. Switching to tt-metal's tested `FABRIC_1D_RING` + `cluster_axis=0` (EP-4) path makes the a2a
> dispatch complete — see **Current status**. The analysis below is the original FABRIC_2D root-cause
> investigation, kept for the record.

### Symptom
With #6 fixed (a2a now `topology == Linear`, verified: the freshly-built `_ttnncpp.so` is the one
mmap'd by the process), execution still **hangs** at the *same* place — `moe_compute`'s first
`finish`:
```
ProgramExecutor::execute → ccl::run(MoeComputeOp) → ttnn::prim::moe_compute
 → create_global_semaphore → reset_semaphore_value
 → enqueue_write_shard_to_sub_grid → finish_nolock → read_completion_queue   (never completes)
```

### Minimal native repro (the watcher DOES work — the PJRT crash was PJRT-specific)
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/test_a2a_metadata_4x8_fabric2d_repro.py`
isolates the a2a metadata op on the exact scenario (native tt-metal, no PJRT/tt-xla, no 120B model,
~1-2 min/iter). Two cases: `test_a2a_metadata_4x8_fabric2d` (the #7 repro) and
`test_a2a_metadata_1x8_fabric1d_control` (**PASSES** — proves harness + watcher work). Run it from
the tt-metal `python_env` with a CLEAN PYTHONPATH (the tt-xla PYTHONPATH shadows torch/ttnn); see
the file header for the exact invocation. The full watcher overflows the ACTIVE_ETH kernel buffer,
but **waypoint-only watcher works** on the native run (the earlier "invalid watcher.enable" crash
was through PJRT only): `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_{RING_BUFFER,STACK_USAGE,
SANITIZE_NOC,ASSERT,PAUSE}=1`.

Getting the repro to build valid (4,8) inputs required fixing the shared runner
`tests/nightly/tg/ccl/moe/test_all_to_all_dispatch_metadata_6U.py`, which was 1D-only: (a)
`tokens_per_device = batch // dispatch_devices` (was `// devices`); (b) persistent output/metadata
buffer first-dim = `dispatch_devices` (was `devices`). On a 1D mesh `devices == dispatch_devices`
so these are inert; on 2D they over-shard the drain-core buffers. Also had to sync the fixed
`_ttnncpp.so` into `build_Release/lib/` (what native ttnn mmaps) in addition to `install/lib/`
(what PJRT loads).

### Root cause (watcher-confirmed): a2a dispatch semaphore never reaches target → NoC Semaphore Wait
Watcher capture saved to `issue7_watcher_capture.log`. In the last dump:
- The a2a dispatch core (worker running k_ids 584|583 = a2a writer|reader) is stuck at
  **`NTW, NSW`** = BRISC **NoC Transaction Wait** + NCRISC **NoC Semaphore Wait**.
- All other workers are `GW,W` (go-wait / idle) → the a2a's **local compute finished**.
- Fabric ethernet routers (`acteth`, `fabric_erisc_router.cpp`) are up (`FSCD`/`NSID`/`NWID`) but
  the cross-device signal never completes.

So the a2a dispatch core waits forever on its **NoC semaphore** — the dispatch-completion
semaphore never reaches its target count. The cross-device atomic-inc
(`fabric_multicast_bidirectional_atomic_inc_1d`, now running Linear-per-axis on the 2D galaxy)
does not deliver the expected count to the waiting core. This is the concrete `moe_compute`-finish
hang seen through PJRT (that finish drains the never-completing a2a).

### Where to look next (tt-metal fabric/kernel)
- The atomic-inc multicast **range/target-count math** for the 8-device EP row on the (4,8) galaxy:
  `fabric_multicast_bidirectional_atomic_inc_1d` (`moe_utils.hpp`) — do the Linear positive/negative
  ranges + the waiter's expected count agree on this device linearization?
- The a2a's semaphore wait target vs what the multicast actually increments per device.
- Device/neighbor mapping for the EP row on the galaxy's FABRIC_2D mesh (is the 8-wide row routed as
  a contiguous 1D line?).

### Operational note
A hard `kill -9` of a hung 32-device run can leave an ethernet core stuck
(`llrt.cpp:566: Timed out while waiting for active ethernet core ... Try resetting the board`) —
and a normal `tt-smi -r` may not clear it; a **power-cycle** can be required. Because diagnosing
this hang *requires* hanging the fabric and hard-killing, the board degrades each cycle — prefer
the fast native repro (one hang + watcher capture) over repeated full-model PJRT runs.

## Artifacts in this branch

- Test + CI entry: `tests/benchmark/test_llms.py` (`test_gpt_oss_120b_tp_moe_fused_galaxy`,
  `_gpt_oss_120b_moe_fused_galaxy_shard_spec_fn`), `.github/workflows/perf-bench-matrix.json`.
- Fixes: `shlo_input_role_propagation.cc` (#1), `decode_utils.py` (#2),
  `StableHLOToTTIRPatterns.cpp` (#3 row-count **and** #9 mapping-value), `TTNNDecomposeLayouts.cpp` (#4),
  `all_to_all_dispatch_metadata.cpp` (convert_2d_to_1d_topology) (#6).
  - **#7 config resolution:** `client_instance.cc` galaxy→**`FABRIC_1D_RING`** (was `FABRIC_2D`),
    `TTNNResolveComposites.cpp` moe_compute topology=**`Ring`** (was `Linear`), test →
    `register_tt_moe_backend(cluster_axis=0)` + EP-shard experts on "batch". (Supersedes #5's
    `FABRIC_2D`/`Linear` choice; #5's reverted `moe_utils.hpp`/`writer_*` kernel edits stay reverted.)
  - **#8 metadata reshard fix:** `TTNNWorkaroundsPass.cpp` (L1-HeightShard moe_compute metadata
    operands) + `TTNNAllocateDistributedOpBuffers.cpp` (drain-core alignment via
    `ttnn.moe_metadata_drain_core`, read in `TTNNOps.cpp`) + `TTNNDeduceMoEComputeLayouts.cpp`
    (removed `reshardTilizeInputToDrainCore`).
- Native repros (tt-metal `python_env`, clean `PYTHONPATH`; under
  `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/`):
  `test_a2a_metadata_4x8_fabric2d_repro.py` (a2a; `..._fabric1d_ring_axis0` = tested path),
  `test_moe_compute_4x8_fabric1dring_repro.py`, `test_moe_decode_chain_4x8_fabric1dring_repro.py`
  (real a2a→moe_compute chain), `test_a2a_metadata_4x8_ttxla_wrongmapping.py` (#9 A/B: injects the
  axis-local mapping). Plus the shared-runner (4,8) fixes in `test_moe_compute_6U.py` /
  `test_all_to_all_dispatch_metadata_6U.py`.
- Docs: `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md` (#1),
  `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` (#3 row-count + **#9 mapping-value follow-up**),
  `MOE_ALL_TO_ALL_DISPATCH_EXPLAINER.md` (background for #3), this doc (#4/#5/#6/#7/#8/#9 + chain +
  Current status + residual hang).

## How to reproduce / continue

```bash
source venv/activate
pytest -svv --num-layers 2 \
  tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
```
Needs a 32-device (4×8) Blackhole galaxy. Use `--num-layers 1` for the minimal decode repro
(the hang reproduces at 1 layer; faster than 2). After editing tt-mlir/tt-metal sources, rebuild
with `cmake --build build` — this cycle it *did* rebuild the tt-mlir ExternalProject and relink the
plugin (verify: `libTTMLIRCompiler.so` and `pjrt_plugin_tt.so` mtimes newer than your edit). If a
plugin `.so` mtime is stale, `rm build/pjrt_implementation/src/pjrt_plugin_tt.so` and rebuild.

**Continue diagnosing the residual decode hang (no full-model PJRT run needed):**

1. **Verify a mapping/IR change without the board** — dump the decode IR under `dry_run` (compiles,
   skips submit, cannot hang): `TTXLA_DRY_RUN=1 pytest -svv --num-layers 1
   tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`. IR lands in
   `modules/irs/ttnn_*_g1_*.mlir` (decode = the graph with `all_to_all_dispatch_metadata`); the
   expert-mapping constant is `main_const_eval_*` (`tensor<32x128xui16>`).
2. **Localize the on-device stuck point** — run the compiled decode flatbuffer natively under `ttrt`
   + the waypoint watcher (bypasses the broken PJRT watcher; `ttrt` was built manually via `pip wheel`
   with `TTMLIR_ENABLE_RUNTIME=ON TT_RUNTIME_ENABLE_TTNN=ON`):
   ```
   TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
   TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 \
   ttrt run modules/fb_*_1lyr_*_g1_*.ttnn --fabric-config FABRIC_1D_RING --program-index all
   ```
   Then read `generated/watcher/watcher.log`: `CWFW` on moe_compute `tilize_*` k_ids = tilize still
   starved; a stuck point in `selective_reduce_combine` / `all_gather` = the hang moved downstream.
   A hang here needs `kill -9` + likely a **power-cycle** (see Operational note).
