# tt-metal: `FABRIC_1D_RING` silently drops wrap edges on non-UBB-galaxy boxes

**Status:** open tt-metal bug, worked around in tt-xla (see [Workaround](#workaround-in-tt-xla)).
**Found on:** `bh-rb-01-...` — a single-host Blackhole box, 8 × p150, `ClusterType::P150_X8`.
**Symptom:** every 2D-mesh tensor-parallel run aborts during engine init with

```
TT_FATAL @ tt_metal/fabric/fabric.cpp:161: forwarding_direction.has_value()
Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)
```

**Reproduces without any DSA / DeepSeek involvement.** Plain Qwen3-0.6B TP is enough:

```bash
# fails  (2D mesh -> (2,4))
pytest -q "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::\
test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-0.6B-True]"
# passes (1D mesh)
pytest -q "...test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-0.6B-False]"
```

tt-metal at `f1f4ff75579`.

---

## 1. The hardware really is a 2×4 torus

`build_Release/tools/umd/system_health` on this box reports 8 chips, 12 chip pairs,
2 cables each (24 links), all UP with `retrain: 0`, 6 of 12 eth channels used per chip:

```
chip 0 -> [1, 2, 4]      chip 4 -> [0, 5, 6]
chip 1 -> [0, 3, 5]      chip 5 -> [1, 4, 7]
chip 2 -> [0, 3, 6]      chip 6 -> [2, 4, 7]
chip 3 -> [1, 2, 7]      chip 7 -> [3, 5, 6]
```

That is a 2×2×2 hypercube (every neighbour differs by exactly one bit), and it is
**graph-isomorphic to a 2×4 torus-X** — both are 3-regular on 8 nodes with 12 edges.

The topology solver already picks a valid embedding, auto-exported to
`generated/fabric/physical_chip_mesh_coordinate_mapping_1_of_1.yaml`:

```
row 0:  chip7  chip6  chip4  chip5     <- fabric chip_ids 0,1,2,3
row 1:  chip3  chip2  chip0  chip1     <- fabric chip_ids 4,5,6,7
```

Checked against the live graph:

| | edges needed | present | unused cables |
|---|---|---|---|
| 2×4 mesh | 10 | ✅ | **2** — exactly `1↔3` and `5↔7` |
| 2×4 torus-X | 12 | ✅ | **0** |

So the two row wrap edges (`5↔7` for row 0, `1↔3` for row 1) are physically
cabled and link-up. Nothing here is a missing-hardware problem.

## 2. tt-mlir correctly detects the ring

`tt-mlir/runtime/lib/common/mesh_fabric_config.cpp`, `classifyLine()`:

```cpp
if (!areConnected(line.front(), line.back())) {
  return FabricConfig::FABRIC_1D;
}
return FabricConfig::FABRIC_1D_RING;      // front <-> back cabled
```

Both mesh rows wrap, so tt-mlir requests `FabricConfig::FABRIC_1D_RING`. Confirmed in
the log with `TT_METAL_LOGGER_LEVEL=Debug`:

```
Fabric config changed from FabricConfig::DISABLED to FabricConfig::FABRIC_1D_RING,
reinitializing control plane (metal_env.cpp:319)
```

This classification is **correct** for this box.

## 3. tt-metal then discards the ring

`tt_metal/fabric/fabric_host_utils.cpp:66`:

```cpp
FabricType get_fabric_type(tt::tt_fabric::FabricConfig fabric_config, bool is_ubb_galaxy) {
    switch (fabric_config) {
        case tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE:
        case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: {
            if (is_ubb_galaxy) {
                return FabricType::TORUS_XY;
            }
            return FabricType::MESH;      // <-- ring discarded here
        }
        ...
```

`P150_X8` is not a UBB galaxy, so a **ring** request resolves to `FabricType::MESH`.
`mesh_graph.cpp:399-416` then lets the requested type win outright:

```cpp
FabricType mgd_fabric_type = MeshGraphDescriptor::infer_fabric_type_from_dim_types(mesh_desc);
if (fabric_config.has_value()) {
    FabricType requested_fabric_type = get_fabric_type(*fabric_config, is_ubb_galaxy);
    if (requires_more_connectivity(requested_fabric_type, mgd_fabric_type, mesh_shape)) {
        TT_THROW("FabricConfig {} requests topology {} which requires more connectivity "
                 "than MGD provides {}. FabricConfig can only restrict topology "
                 "(e.g., torus->mesh), not create new connections.", ...);
    }
    effective_fabric_type = requested_fabric_type;   // <-- MESH wins
}
```

`get_valid_connections()` (`mesh_graph.cpp:196-213`) only emits wrap coordinates under
`TORUS_X` / `TORUS_Y`, so with `effective_fabric_type == MESH` **no wrap edge is ever
added to intra-mesh connectivity.**

## 4. …but the collectives still behave like a ring

The fabric config is still *named* `FABRIC_1D_RING`, and ttnn's CCLs act accordingly.
`ttnn::all_gather` over a tensor spanning a full mesh row builds a ring schedule and asks
fabric for a direct hop between the row endpoints — `D0 → D3` — which is the wrap. Fabric
has no such route, and `append_fabric_connection_rt_args` aborts at `fabric.cpp:161`.

Observed stack (abbreviated):

```
ttnn::all_gather
  -> ttnn::operations::ccl::AllGatherDeviceOperation::AllGatherProgram::create_at
    -> build_all_gather_async_minimal_default_program_artifacts
      -> tt::tt_fabric::FabricMuxConfig::get_fabric_mux_run_time_args
        -> tt::tt_fabric::append_fabric_connection_rt_args
          -> TT_FATAL forwarding_direction.has_value()
```

ttnn already knows this hazard exists — `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py:2137`:

> The op normally derives this from the fabric config + tensor coverage via
> `get_usable_topology()`, but that heuristic marks any tensor that spans a full mesh row
> as WRAP/Ring — incorrect for physically-line meshes (e.g. BH single Loudbox p150_x8).
> When set, this lets the caller force Linear and avoid forwarding requests across
> non-existent wrap edges.

That comment describes the same failure but assumes the box has no wrap cables. On this
box the cables **do** exist — the wrap edges are missing from fabric's routing tables
purely because `get_fabric_type()` downgraded the request.

### The inconsistency, in one line

`get_fabric_type()` decides ring-vs-mesh **connectivity** from `is_ubb_galaxy`, while the
CCL layer decides ring-vs-line **scheduling** from the fabric config name and tensor
coverage. On any non-UBB-galaxy box with real wrap cables the two disagree, and the
result is a hard abort rather than a clean fallback.

## 5. How tt-metal should fix it

In rough order of preference.

**(a) Gate on actual connectivity, not on `is_ubb_galaxy`.** The MGD already carries the
answer via `infer_fabric_type_from_dim_types()`. In `mesh_graph.cpp:402-416`, when the
requested config is a ring variant and the MGD provides torus connectivity, prefer the
MGD's type instead of restricting to `MESH`:

```cpp
// FABRIC_1D_RING asks for a ring. get_fabric_type() can only answer MESH or
// TORUS_XY (via is_ubb_galaxy), so honour what the MGD actually provides.
if (is_ring_fabric_config(*fabric_config) && mgd_fabric_type != FabricType::MESH) {
    effective_fabric_type = mgd_fabric_type;
} else {
    effective_fabric_type = requested_fabric_type;
}
```

This makes `is_ubb_galaxy` unnecessary and generalises to any box whose MGD declares
`dim_types` with a `RING`.

**(b) Register a torus MGD for `P150_X8`.** `mesh_graph.cpp:74-105` maps
`FabricType` → `ClusterType` → descriptor filename. `TORUS_X`/`TORUS_Y`/`TORUS_XY` only
have `GALAXY` and `BLACKHOLE_GALAXY` entries; **`P150_X8` has no torus variant at all**.
Its only entry is under `FabricType::MESH`:

```cpp
{tt::tt_metal::ClusterType::P150_X8, "p150_x8_mesh_graph_descriptor.textproto"},
```

and that descriptor understates the box:

```
device_topology { dims: [ 2, 4 ] }        # no dim_types -> both dims LINE
host_topology   { dims: [ 1, 1 ] }
channels { count: 2 policy: RELAXED }
```

It should declare `dim_types: [ LINE, RING ]`, matching
`tests/scale_out/4x_bh_quietbox/mesh_graph_descriptors/2x4_bh_torus_x_mesh_graph_descriptor.textproto`.
Note a `RING` on the 2-wide dimension would be degenerate (front and back are the same
adjacent pair), so only the 4-wide dimension should wrap.

**(c) Fail loudly instead of silently.** If a ring config is requested and the resolved
`FabricType` has no wrap edges, that should be a startup error naming both values — not
an abort thousands of lines later inside a CCL program build. Right now the only signal
is `fabric.cpp:161`, which names two fabric node IDs and no topology at all.

**(d) Make `TT_MESH_GRAPH_DESC_PATH` work in production.** `rtoptions.cpp:493-499`
parses it into `custom_fabric_mesh_graph_desc_path`, but the only consumers are under
`tests/` — the production `ControlPlane` path always resolves the MGD from
`cluster.get_cluster_type()`. Setting the env var appears to work and silently does
nothing, which cost real debugging time here.

## 6. Things that do *not* fix it

- **Editing `p150_x8_mesh_graph_descriptor.textproto` alone.** Adding
  `dim_types: [ LINE, RING ]` raises `mgd_fabric_type` to `TORUS_X`, but
  `mesh_graph.cpp:413` overwrites it with the requested `MESH`. Verified inert: identical
  failure with all three on-disk copies patched (source tree,
  `build_Release/libexec/tt-metalium/...`, and `tt-mlir/install/tt-metal/...`).
- **`TT_MESH_GRAPH_DESC_PATH`** — test-only, see (d) above.
- **`TT_VISIBLE_DEVICES` to reorder chips.** `ClusterDescriptor::get_target_chip_ids_from_visible_devices()`
  returns `std::unordered_set<ChipId>`, so it filters but cannot reorder. Mesh
  coordinates are assigned by the topology solver, not by this list.
- **Recabling.** Unnecessary — §1 shows the wrap edges are already present.

## Workaround in tt-xla

Two changes, both required together:

1. `third_party/tt-mlir/.../runtime/lib/common/mesh_fabric_config.cpp` —
   `computeMeshFabricConfig()` now requests `FABRIC_2D_TORUS_{X,Y,XY}` instead of
   `FABRIC_1D_RING` when the mesh is genuinely 2D and a wrapping axis is longer than 2
   elements. These configs map directly to the matching `FabricType`, so the wrap edges
   get built.
2. `third_party/.../tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto` —
   `device_topology { dims: [ 2, 4 ] dim_types: [ LINE, RING ] }`. Without this,
   `requires_more_connectivity(TORUS_X, MESH, [2,4])` throws, since a 4-wide wrap is not
   degenerate.

Caveat: change 1 moves this box from **1D** to **2D** fabric, which is a larger
behavioural change than just enabling the wrap — different EDM setup and CCL kernels.
If 2D fabric turns out to be unsupported or slower here, the conservative alternative is
to force `Topology::Linear` at
`tt-mlir/runtime/lib/ttnn/operations/ccl/all_gather.cpp:38`, which currently passes
`std::nullopt` and so defers to ttnn's `get_usable_topology()` heuristic. That leaves the
two wrap cables idle but avoids the ring path entirely.

Both changes live in vendored submodules and will be lost on submodule update. Upstreaming
(a) + (b) to tt-metal is the durable fix.

### Verified

With both changes applied and tt-xla rebuilt, on `bh-rb-01-...` (8 × p150):

| test | before | after |
|---|---|---|
| `test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-0.6B-True]` (2D mesh) | `fabric.cpp:161` abort | **passed**, 183 s |
| `test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-0.6B-False]` (1D mesh) | passed | passed (unaffected) |
| `test_tensor_parallel_generation_deepseek_v32_3l[sparse-topk128]` (`mesh_shape [2,4]`) | `fabric.cpp:161` abort | **passed**, 188 s |

The MGD change alone was confirmed inert (three runs, identical abort). The
`mesh_fabric_config.cpp` change is what takes effect; the MGD change only keeps
`requires_more_connectivity()` from rejecting it.

Note on the DeepSeek run: at the test's original `optimization_level: 0` it passes but the
DSA composites still inline their decompositions. Raising it to `optimization_level: 1`
promotes all three to `ttnn.indexer_score_dsa` / `ttnn.topk_large_indices` /
`ttnn.sparse_sdpa` (verified in the exported IR: 6 sites each, and `ttnn.relu`/`scatter`/
`softmax` drop from 12 each to 0). Composite promotion therefore requires the optimizer;
that is unrelated to fabric. On the kernel path the run then hits a separate
`indexer_score_dsa` sharding-assumption failure — see the DSA handoff notes.
