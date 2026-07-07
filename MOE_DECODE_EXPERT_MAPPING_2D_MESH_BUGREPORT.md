# `moe_decode` synthesizes a cluster-axis-sized expert_mapping; needs full-mesh rows on a 2D mesh

## TL;DR

> **Two distinct bugs, found in sequence.** This doc originally covered the mapping's **row
> count** (below). A follow-up (2026-07-06) found the mapping's **values** were *also* wrong on a
> 2D mesh — see **"Follow-up: the mapping VALUE was axis-local, not the global mesh coord"** near
> the end. The follow-up is the one that caused the observed 29/32-device decode hang; it is exactly
> the `cluster_axis=0` correctness gap this doc's own caveat flagged as "to be double-checked."

For the fused MoE decode path, tt-mlir synthesizes the `all_to_all_dispatch_metadata`
`expert_mapping` with **one row per cluster-axis device** (`meshShape[cluster_axis]`), but
the op's reader kernel indexes the mapping by the **global linearized mesh coordinate**,
so it requires **one row per mesh device**. On a 1×N dispatch ring these coincide; on a 2D
mesh (e.g. 4×8, `cluster_axis=1`) the mapping is `[8, experts]` while the op expects
`[32, experts]`, and execution aborts:

```
TT_FATAL: Expert mapping tensor first dimension must equal number of devices (32), got 8
  all_to_all_dispatch_metadata_device_operation.cpp:123
```

The op's `validate` check is **correct**; the bug is the mapping synthesis in tt-mlir,
which is self-documented as a TODO for the 4×8 case.

## Affected configuration

- Any fused `tt.moe_decode` (`experts_implementation="tt_moe_fused"`, decode step) on a
  **2D mesh where the EP cluster axis is smaller than the whole mesh** — i.e. EP on one
  axis with data-parallel replication on the other.
- First observed: `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`
  on a 4×8 Blackhole galaxy mesh (`("batch","model")`, `cluster_axis=1` → EP-8 on
  "model", DP-4 on "batch"). Also applies to the 2×4 20B `moe_fused` path (would be 4 vs
  8), so a 2×4 run hits the same assert.
- This reproduces only after the `tt.mark_argument` uint32 fix
  (`MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md`) is in place; before that, compilation
  fails earlier.

## Background: `cluster_axis`, dispatch extent, and the mapping's row count

`all_to_all_dispatch_metadata` (`ttnn/.../experimental/ccl/all_to_all_dispatch_metadata/`)
takes an `axis` attribute (= `cluster_axis`). On a 2D mesh it means (device op comment,
lines 148–153):

- **Experts are EP-sharded along `axis`**; each device on that axis owns a slice.
- **Tokens are data-parallel along `axis`** (dispatched among those devices) and
  **duplicated along the other axis** (each off-axis replica is independent).

There are **two distinct device counts**, and they must not be conflated:

1. **Dispatch extent** = `dispatch_devices` = `axis==0 ? num_rows : num_cols` (= 8 for
   4×8, `axis=1`). This drives output token count, neighbor routing, and the dispatch
   loop. Computed correctly and consistently in `compute_output_specs`
   (device op lines 155–160) and the program factory (lines 281–284).

2. **Mapping row count** = **total mesh devices** (= 32). The reader kernel reads exactly
   one mapping page — the source device's row — indexed by the *global* linearized mesh
   coordinate:

   ```cpp
   // reader_all_to_all_dispatch_metadata.cpp:101-107
   // Page index = linearized_mesh_coord (source device index)
   noc_async_read_page(linearized_mesh_coord, mapping_addr_gen, base_mapping_addr);
   ```

   with `linearized_mesh_coord = row*num_cols + col ∈ [0, num_mesh_devices)`. So the
   mapping must have `num_mesh_devices` rows or devices with `coord >= dispatch_devices`
   read out of bounds. `validate` (device op line 118) enforces exactly this:
   `expected_devices = mesh_view.num_devices()`.

The mapping value is the expert's **axis-local owner position**
(`owner(e) = e / experts_per_device ∈ [0, dispatch_devices)`), and it is a global fact, so
**every row is identical** (replicated across the non-cluster axis). The device reads its
own row only to obtain this replicated owner vector; routing to the owner is then done
along `axis` in the kernel.

## Root cause

tt-mlir's `synthesizeMoeDecodeExpertMapping`
(`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`) built the mapping with row
count = `numDevices = meshShape[cluster_axis]` (the dispatch extent, 8), not the total
mesh device count (32). Its own comment flagged the gap:

```cpp
// Contiguous owner layout (1xN dispatch ring); the replicated cluster_axis==0
// interleave (4x8) is a separate TODO.
```

Caller (same file):

```cpp
int64_t numDevices = meshShape[clusterAxis];              // 8  (dispatch extent)
...
int64_t expertsPerDevice = w0Ty.getShape()[1];            // 16
Value mapping = synthesizeMoeDecodeExpertMapping(rewriter, loc, numDevices, expertsPerDevice);
//                                                                 ^^^^^^^^^^  used as the row count -> [8, 128]
```

So a correct-shaped `[dispatch_devices, experts]` was produced for a 1×N ring but a
too-short `[8, 128]` for the 4×8 mesh.

## Fix (implemented)

Decouple the mapping's **row count** (total mesh devices) from the **expert count**
(`experts_per_device * cluster_devices`). The values are unchanged
(`owner(e) = e / experts_per_device`), and every row is identical, so the fix is a
row-count expansion — it realizes the previously-TODO replicated 2D case.

`synthesizeMoeDecodeExpertMapping` now takes the mesh device count and the cluster device
count separately:

```cpp
static Value synthesizeMoeDecodeExpertMapping(OpBuilder &builder, Location loc,
                                              int64_t numMeshDevices,     // rows = all mesh devices (32)
                                              int64_t clusterDevices,     // EP extent along axis (8)
                                              int64_t expertsPerDevice) { // 16
  int64_t numExperts = expertsPerDevice * clusterDevices;                // 128
  auto mappingTy = RankedTensorType::get({numMeshDevices, numExperts}, u16);
  for (d in numMeshDevices) for (e in numExperts)
    values.push_back(e / expertsPerDevice);                             // axis-local owner, replicated
  ...
}
```

Caller computes the total mesh device count and passes both:

```cpp
int64_t totalMeshDevices = 1;
for (int64_t d : meshShape) totalMeshDevices *= d;                       // 32
Value mapping = synthesizeMoeDecodeExpertMapping(
    rewriter, loc, totalMeshDevices, numDevices /*=cluster*/, expertsPerDevice);
```

`numDevices` (cluster extent, 8) is still used unchanged for the dispatch/decomposition
(all_gather over the cluster axis, global expert count `E_local * numDevices`, and the
`all_to_all_dispatch` `num_devices` attribute). Only the mapping's row count changes.

### Why this is sufficient (and its caveats)

- `compute_output_specs` and the program factory already derive `dispatch_devices`
  axis-awarely, so nothing else assumes a `[dispatch_devices, experts]` mapping. The only
  consumer that indexes rows — the reader kernel — uses the global coordinate, which the
  32-row mapping now satisfies.
- The owner value is the **axis-local** position (0–7), and dispatch routing walks the
  cluster axis via `get_neighbors(mesh_view, coord, topology, axis)`, so replicating rows
  is semantically correct for both `cluster_axis=0` and `cluster_axis=1`.
  > **⚠ This bullet is WRONG — corrected 2026-07-06.** The value must be the **global** mesh
  > coord, not the axis-local owner, and the rows must **differ** per source (not be replicated).
  > See the "Follow-up" section below; this was the actual decode-hang cause.
- **To validate on hardware:** confirm decode PCC is correct end-to-end (the fix makes the
  program *run*; correctness of the owner→fabric-destination mapping for `cluster_axis=0`
  should be double-checked, since only `cluster_axis=1` has been exercised here).
  > **Update 2026-07-06:** double-checking this (as flagged) is exactly what surfaced the
  > value bug — see the Follow-up section.

## Follow-up: the mapping VALUE was axis-local, not the global mesh coord (the decode-hang cause)

**Found 2026-07-06.** After the row-count fix above made the program *run*, the fused decode still
hung on the 4×8 galaxy (29/32 devices stuck; watcher showed the moe_compute **tilize_reader** at
`CWFW` — a circular-buffer wait — because most devices' experts never received their dispatched
tokens). Root cause: the mapping **value** was the **axis-local owner**, but the a2a dispatch kernel
requires the value to be the **global linearized mesh coordinate** of the owner device.

### The kernel contract (what the value must be)

`all_to_all_dispatch_metadata`'s dispatch writer uses `expert_mapping[expert]` **directly** as
`target_device`, and treats it as a *global* coord:

```cpp
// writer_all_to_all_dispatch_metadata.cpp (dispatch_tokens_to_devices)
uint16_t target_device = expert_mapping[expert_chosen];
if (target_device == LinearizedSrcMeshCoord) { /* expert is local */ }          // global-coord compare
else if (is_configured_target<LinearizedSrcMeshCoord, MeshRows, MeshCols, Axis>(target_device)) { ... }
send_preparation_buffer[(local_token - token_start_idx) * NumDevices + target_device] = 1;  // index in [0, NumDevices)
```

and `is_configured_target` (`moe_utils.hpp:302`) computes `dest / MeshCols` and `dest % MeshCols`.
So `target_device` **must be in `[0, NumDevices)` (a full mesh coord)**. The canonical references
agree: both deepseek `get_linearized_mesh_coord` and the 6U `gen_expert_mapping` emit a global coord
(comment: "each entry is the linearized mesh coordinate of the device that owns that expert").

### The bug

`synthesizeMoeDecodeExpertMapping` (`StableHLOToTTIRPatterns.cpp`) emitted `value = e /
expertsPerDevice` = the **axis-local owner** in `[0, clusterDevices)` (`[0,4)` for `cluster_axis=0`
on a 4×8 mesh), replicated identically on every row. Its own comment admitted the gap: *"the
replicated cluster_axis==0 interleave (4x8) is a separate TODO."* On a 2D mesh axis-local ≠ global,
so every dispatch misroutes (tokens go only to global devices 0–3), the true owner devices get no
tokens, and their moe_compute tilize waits forever.

### The fix (implemented)

Emit the **global coord of the owner replica in each source device's own cluster group** — this
makes the mapping rows *differ* per source (no longer identical). With `numCols = meshShape[1]`,
`row(d)=d/numCols`, `col(d)=d%numCols`, `a = e/expertsPerDevice`:

```cpp
if      (meshShape.size() < 2) target = a;                     // 1D: global coord == axis-local owner
else if (clusterAxis == 0)     target = a * numCols + col(d);  // owner varies along axis 0; same column as source
else                           target = row(d) * numCols + a;  // owner varies along axis 1; same row as source
```

The helper's signature changed from `(numMeshDevices, clusterDevices, expertsPerDevice)` to
`(meshShape, clusterAxis, expertsPerDevice)` so it can compute row/col.

### Validation

- **dry_run IR (no board):** the decode-graph mapping constant (`main_const_eval_13`,
  `tensor<32x128xui16>`) now has all 4096 entries = `a*8 + d%8`, values spanning `[0,31]` (was
  `[0,3]`), rows differing per source (row0 `{0,8,16,24}`, row1 `{1,9,17,25}`, … row7
  `{7,15,23,31}`, row8 cycles back). Confirmed by decoding the dense-hex constant.
- **A/B native a2a** (`test_a2a_metadata_4x8_ttxla_wrongmapping.py`, monkeypatches the runner's
  mapping to tt-xla's axis-local value): routing target differs (correct: expert28→dev7; wrong:
  expert17→dev**2**, a `[0,4)` value). NB: the a2a alone does **not** hang under either mapping — it
  completes and only *misroutes*; the hang is downstream in moe_compute, matching the watcher.

### Status: necessary but not yet sufficient

With the fix the full 1-layer decode gets much further (prefill executes; the fused decode compiles
and submits) and the old 32-thread `completion_queue` signature is gone, **but it still hangs on-device
later** (~program-time 969s; a single `FDMeshCommandQueue::read_completion_queue` /
`completion_queue_wait_front` reader stuck). Open question: whether the mapping fix cleared the tilize
`CWFW` and a *downstream* fused-decode collective (`selective_reduce_combine` / post-moe `all_gather`)
now hangs, or the tilize still stalls for another reason. **Next step:** run the fixed decode flatbuffer
(`modules/fb_*_1lyr_*_g1_*.ttnn`) under `ttrt` + waypoint watcher (same method that localized the
original tilize `CWFW`) to see if the stuck point moved. Keep this fix regardless.

## References

- `third_party/.../tt-metal/.../all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_device_operation.cpp:118`
  — `expected_devices = mesh_view.num_devices()` (the *correct* mapping-row check).
- `.../all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_program_factory.cpp:281`
  — axis-aware `dispatch_devices` (dispatch extent, distinct from mapping rows).
- `.../all_to_all_dispatch_metadata/device/kernels/dataflow/reader_all_to_all_dispatch_metadata.cpp:101`
  — reads `mapping[linearized_mesh_coord]` (global index → needs full-mesh rows).
- `third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
  — `synthesizeMoeDecodeExpertMapping` and its caller (the fix site).
- Related: `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md` (must be fixed first to reach this
  code path).
- Example test: `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`.
