# DSA on Blackhole — tt-metal changes and upstream asks

Companion to [`dsa_blackhole_tt-mlir_changes.md`](./dsa_blackhole_tt-mlir_changes.md).
That doc covers the three tt-mlir commits (pushed to `hshah/all-dsa-ops`); this one covers
the **tt-metal** side: the single local edit, and the tt-metal-side issues found while
bringing DSA up on Blackhole that were *not* fixed and should be raised upstream.

Baseline: tt-metal `f1f4ff75579` (`v0.75.0-dev20260717-29`), the SHA tt-mlir's
`third_party/tt-metal` submodule pins. Checked out detached, as submodules are.
Hardware: `bh-rb-01-…`, single host, 8 × p150, `ClusterType::P150_X8`, mesh `[2, 4]`.

---

## 1. The one change — and it is NOT committed

`tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto`

```diff
 mesh_descriptors {
   name: "M0"
   arch: BLACKHOLE
-  device_topology { dims: [ 2, 4 ] }
+  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, RING ] }
   host_topology   { dims: [ 1, 1 ] }
   channels { count: 2 policy: RELAXED }
 }
```

> ⚠️ **State: uncommitted working-tree edit.** It exists only in this machine's
> CMake-managed tt-metal checkout and is lost by anything that re-checks-out tt-metal. It
> could not ride along with the tt-mlir commits: tt-metal is a separate repo, pinned by SHA
> as a submodule of tt-mlir, so landing it needs a branch/PR in tt-metal **plus** a
> submodule-pointer bump in tt-mlir. See [§3](#3-how-to-land-it).

### Why

Without `dim_types` both dimensions default to `LINE`, and tt-mlir's
`4c6f88a08e` ("Request a 2D torus fabric config when a 2D mesh's long axis wraps") is then
rejected by `requires_more_connectivity(TORUS_X, MESH, [2,4])` — a 4-wide wrap is not
degenerate, so the request asks for connectivity the descriptor does not advertise.

The descriptor understates the box. `build_Release/tools/umd/system_health` reports 8 chips,
12 chip pairs at 2 cables each (24 links), all UP with `retrain: 0`, 6 of 12 eth channels
used per chip:

```
chip 0 -> [1, 2, 4]      chip 4 -> [0, 5, 6]
chip 1 -> [0, 3, 5]      chip 5 -> [1, 4, 7]
chip 2 -> [0, 3, 6]      chip 6 -> [2, 4, 7]
chip 3 -> [1, 2, 7]      chip 7 -> [3, 5, 6]
```

That is a 2×2×2 hypercube, which is **graph-isomorphic to a 2×4 torus-X** (both 3-regular
on 8 nodes with 12 edges). Against the embedding the topology solver actually picks —
exported to `generated/fabric/physical_chip_mesh_coordinate_mapping_1_of_1.yaml` as
row 0 = chips 7,6,4,5 and row 1 = chips 3,2,0,1 — all 10 mesh edges are present **and** so
are both row wrap edges (`5↔7`, `1↔3`):

| | edges needed | present | cables unused |
|---|---|---|---|
| 2×4 mesh | 10 | ✅ | **2** — exactly `1↔3` and `5↔7` |
| 2×4 torus-X | 12 | ✅ | **0** |

So declaring `RING` is not claiming capability the hardware lacks; it describes what is
already cabled. Only the 4-wide dimension is declared `RING` — a `RING` on the 2-wide
dimension would be degenerate, since front and back are the same adjacent pair.

Compare `tests/scale_out/4x_bh_quietbox/mesh_graph_descriptors/2x4_bh_torus_x_mesh_graph_descriptor.textproto`,
which already uses `dim_types: [ LINE, RING ]` for the same 2×4 P150 shape (differing only
in `host_topology`).

### This edit does nothing on its own

Verified across three runs with all three on-disk copies patched: identical abort.
`mesh_graph.cpp:413` overwrites the MGD-derived type with the requested one, so the
descriptor only raises the **ceiling**; tt-mlir's `4c6f88a08e` is what takes effect. The two
are strictly a pair — either both or neither.

### Three copies on disk; the live one is in the tt-mlir install tree

The plugin loads `third_party/tt-mlir/install/lib/_ttnncpp.so`, and the runtime resolves
MGDs relative to `rtoptions.get_root_dir()`:

```
…/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/                       # source
…/tt-metal/src/tt-metal/build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/   # build
third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/           # LIVE
```

A rebuild reinstalls from the source copy, so patching the source keeps the change across
builds. Patching only the source copy without rebuilding has no effect — that cost real
debugging time here.

---

## 2. Upstream asks — tt-metal issues found but NOT fixed

Ordered roughly by value.

### 2.1 `FABRIC_1D_RING` silently drops wrap edges on non-UBB-galaxy boxes

`tt_metal/fabric/fabric_host_utils.cpp:66`:

```cpp
case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: {
    if (is_ubb_galaxy) { return FabricType::TORUS_XY; }
    return FabricType::MESH;      // ring discarded
}
```

`FabricType::MESH` means `get_valid_connections()` (`mesh_graph.cpp:196-213`) never emits
wrap coordinates, so no wrap edge enters intra-mesh connectivity. But the config is still
*named* ring, and the CCL layer builds a ring schedule accordingly, then asks fabric to
forward across the wrap:

```
TT_FATAL @ tt_metal/fabric/fabric.cpp:161: forwarding_direction.has_value()
Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)
```

Connectivity is decided from `is_ubb_galaxy`; scheduling is decided from the config name and
tensor coverage. On any non-UBB-galaxy box with real wrap cables the two disagree, and the
result is a hard abort rather than a clean fallback. This blocked **every** 2D-mesh
tensor-parallel run on this box, DSA or not — plain Qwen3-0.6B reproduces it.

**Suggested fix:** gate on actual connectivity, not board class. The MGD already answers the
question via `infer_fabric_type_from_dim_types()`, so in `mesh_graph.cpp:402-416`, when the
request is a ring variant and the MGD provides torus connectivity, prefer the MGD's type
instead of restricting to `MESH`:

```cpp
if (is_ring_fabric_config(*fabric_config) && mgd_fabric_type != FabricType::MESH) {
    effective_fabric_type = mgd_fabric_type;
} else {
    effective_fabric_type = requested_fabric_type;
}
```

That makes `is_ubb_galaxy` unnecessary and generalises to any box whose MGD declares a
`RING` dim. It would also make §1 sufficient on its own, retiring the tt-mlir workaround.

### 2.2 No torus MGD registered for `P150_X8`

`mesh_graph.cpp:74-105` maps `FabricType` → `ClusterType` → descriptor filename.
`TORUS_X` / `TORUS_Y` / `TORUS_XY` have entries **only** for `GALAXY` and
`BLACKHOLE_GALAXY`. `P150_X8` appears solely under `FabricType::MESH`, so even a correct
torus request has no descriptor to resolve to. Registering a `p150_x8_torus_x_…textproto`
(or accepting §1) closes this.

### 2.3 Failure surfaces far from its cause

A ring config resolving to a wrap-less `FabricType` is knowable at control-plane init, but
the only symptom is `fabric.cpp:161` thousands of lines later inside a CCL program build,
naming two fabric node IDs and no topology. A startup check that names both the requested
config and the resolved `FabricType` would have saved most of this investigation.

### 2.4 `TT_MESH_GRAPH_DESC_PATH` is parsed but test-only

`rtoptions.cpp:493-499` parses it into `custom_fabric_mesh_graph_desc_path`, but the only
consumers are under `tests/` — the production `ControlPlane` path always resolves the MGD
from `cluster.get_cluster_type()`. Setting the env var appears to work and silently does
nothing. Either wire it into the production path or reject it with a warning.

### 2.5 `indexer_score_dsa` cannot express a sequence-replicated query

`indexer_score_device_operation.cpp` (`max_chunk_start` / `validate_chunk_start`) models the
query as sequence-parallel: it derives a per-device rank from `q`'s device coords and
requires `T >= (max_rank + 1) * Sq`. A sequence-**replicated** query on N devices reports
rank N-1 with `Sq == T` and aborts:

```
TT_FATAL @ indexer_score_device_operation.cpp:134: max_cs + Sq <= T
fullest-device chunk window [1792, 1792+256) exceeds T=256 (base=0, per-rank stride Sq=256)
```

There is no way to say "the sequence is not sharded". `max_linearized_rank` returns 0 only
for a single-device tensor; with `cluster_axis` set it returns `physical_coord[axis]`, so the
only escape is naming a **size-1 mesh axis** — impossible on `[2,4]`. Note
`a07201edbbf` ("consolidate seq-shard axes into a single `seq_shard_axes` arg") does **not**
help: `seq_shard_axes = []` means *flat linear device order*, not replicated, and
`split_seq_shard_axes()` decomposes straight back into the old two fields.

Prefill works around this by genuinely sharding the sequence. **Decode cannot** — there is
one query row, nothing to split — so this is a hard blocker for sparse decode on any mesh
without a size-1 axis, including the `[2,4]` and `[8,4]` shapes DSA targets.

**Suggested fix:** a sentinel or flag meaning "sequence not sharded" that forces `ring = 1`
regardless of device count.

### 2.6 `Sq % TILE_HEIGHT == 0` makes decode fall back silently

`indexer_score_device_operation.cpp:287`:

```cpp
Sq % tt::constants::TILE_HEIGHT == 0 && T % tt::constants::TILE_WIDTH == 0 && ...
    "Sq {}, T {}, D {} must be tile-aligned"
```

Decode has `Sq = 1` per user, so the op is illegal there and the composite **silently
inlines its decomposition** — no error, no warning, just the slow path. This is why the
sparse decode graph shows `ttnn.relu` and no `ttnn.indexer_score_dsa`.

Padding the decode query to one tile is a viable caller-side fix (cheap: the score grows
from `[1,1,1,T]` to `[1,1,32,T]`), but the silence is the real hazard — a promotion veto
that produces different performance and, for `topk_large_indices`, different semantics
should not be invisible.

### 2.7 Minor: `TT_VISIBLE_DEVICES` cannot reorder chips

`ClusterDescriptor::get_target_chip_ids_from_visible_devices()` returns
`std::unordered_set<ChipId>`, so the variable filters which chips are used but cannot
influence mesh-coordinate assignment. Worth documenting, since the comma-separated list
reads like an ordering.

### 2.8 `sparse_sdpa` cannot read a paged KV cache, so DSA decode must gather

This is the **highest-value performance ask** here, and it supersedes what the tt-xla
handoff doc calls "G1".

DSA decode selects `top-k` keys, so the attention arithmetic is `O(top-k)`. But
`sparse_sdpa` cannot be pointed at vLLM's paged latent KV cache, so
`TTMLAAttentionBackendImpl._forward_decode_sparse` gathers each user's whole context
into a dense buffer every step:

```python
blocks = torch.index_select(kv_cache, 0, page_table[u].to(torch.int64))
kv_u = blocks.reshape(1, 1, max_seq_len, head_dim)     # O(context) per layer per step
```

Memory traffic is therefore `O(context)` and grows with sequence length, which is what
makes sparse decode *slower* than dense decode today. Two constraints cause it:

**(a) `cache_batch_idx` assumes a batch-contiguous cache.** The header advertises it as
selecting a batch slot in a shared `[B, 1, T, K_DIM]` cache, and the implementation is
a flat offset:

```cpp
const uint32_t kv_batch_page_offset = attrs.cache_batch_idx.value() * T;
```

Slot `b` must occupy rows `[b*T, (b+1)*T)`. A paged cache scatters a request's blocks
by design, so no per-slot offset can describe it. The parameter is genuinely useful for
a batch-contiguous deployment; it just does not fit paging.

**(b) A layout conflict between `sparse_sdpa` and the cache-write ops.**

```cpp
sparse_sdpa_device_operation.cpp: TT_FATAL(kv.layout() == Layout::ROW_MAJOR,
                                          "sparse_sdpa kv must be ROW_MAJOR");
update_cache_device_operation.cpp:32: TT_FATAL(input.layout() == TILE && cache.layout() == TILE,
                                              "Inputs to update_cache must be tilized");
```

The same tensor would have to be ROW_MAJOR for `sparse_sdpa` and TILE for
`paged_update_cache` / `paged_fill_cache`. Converting per step is `O(cache)` --
strictly worse than the gather it replaces.

**The ask, in preference order:**

1. **Give `sparse_sdpa` a `page_table` operand**, so it resolves logical key positions
   through a page table the way its dense sibling already does. The precedent is
   in-tree: `paged_flash_multi_latent_attention_decode` is paged-aware, which is
   exactly why *dense* MLA decode reads the cache directly and sparse cannot.
   `sparse_sdpa` is the outlier among the MLA decode ops, and this would close the gap
   for any paged deployment, not just DSA.
2. **Failing that, accept a TILE `kv`** (constraint b alone). That plus the
   index-translation trick below is sufficient, and is narrower kernel work.

**The `O(top-k)` half is solvable caller-side once (b) is resolved** and needs no
`cache_batch_idx` at all: translate each top-k index from a logical position to a
physical cache position through the page table --
`physical = page_table[u][p / block_size] * block_size + (p % block_size)` -- and pass
the whole cache as `kv` with `B == 1`. That is an `O(top-k)` gather over the page table
rather than an `O(context)` copy of the cache, and `0xFFFFFFFF` sentinels pass through
untranslated. Worth noting so the request is understood as "let us address the cache in
place", not "add another indexing mode".

---

## 3. How to land it

1. Branch tt-metal off `f1f4ff75579`, apply §1, PR it. Ideally raise §2.1 at the same time —
   if §2.1 lands, §1 becomes sufficient by itself and tt-mlir's `4c6f88a08e` can be reverted.
2. Bump tt-mlir's `third_party/tt-metal` submodule pointer to that commit on
   `hshah/all-dsa-ops`, and push.
3. Nothing in tt-xla needs to change — it pins tt-mlir by branch
   (`TT_MLIR_VERSION "hshah/all-dsa-ops"`), so a tt-mlir push updates every build.

**Until step 2, 2D-mesh TP on an 8 × p150 box does not work from a fresh checkout.** This
machine works only because of the local edit in §1. The DSA prefill kernel path itself
(tt-mlir `4f5bc20e78` + `4949466d43`) is unaffected and is fully landed.
