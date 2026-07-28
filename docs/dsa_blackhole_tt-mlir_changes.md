# DSA on Blackhole — tt-mlir / tt-metal submodule changes

Every change made below the tt-xla source tree to get DeepSeek Sparse Attention running
on **real Blackhole TTNN kernels** on `bh-rb-01-...` (single host, 8 × p150,
`ClusterType::P150_X8`, mesh `[2, 4]`).

| # | repo | file | kind | state |
|---|---|---|---|---|
| [1](#1-fabric_1d_ring-drops-wrap-edges) | tt-mlir | `runtime/lib/common/mesh_fabric_config.cpp` | workaround for a tt-metal bug | committed `4c6f88a08e` |
| [2](#2-p150_x8-mesh-graph-descriptor-understates-the-box) | tt-metal | `tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto` | descriptor fix (prereq for 1) | ⚠️ **uncommitted** |
| [3](#3-sdyall_slice-is-a-delta-not-an-absolute-sharding) | tt-mlir | `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` | **upstream bug fix** | committed `4f5bc20e78` |
| [4](#4-indexer_score_dsa-forbade-the-only-layout-its-kernel-accepts) | tt-mlir | `lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp` | **upstream bug fix** | committed `4949466d43` |

Changes 1, 3 and 4 are committed on tt-mlir branch `hshah/all-dsa-ops`, which is what
tt-xla's `third_party/CMakeLists.txt` pins (`TT_MLIR_VERSION "hshah/all-dsa-ops"` — a
branch, not a SHA, so a push updates every build of this tt-xla revision). **They are not
pushed yet.** Changes 3 and 4 are independent upstream bug fixes and are worth separate
PRs to tt-mlir `main`.

Change 2 is still an **uncommitted working-tree edit** in the tt-metal checkout and will be
lost by anything that re-checks-out tt-metal. It cannot ride along with the tt-mlir commits:
tt-metal is a separate repo (a submodule of tt-mlir, pinned by SHA), so landing it means a
branch/PR in tt-metal plus a submodule-pointer bump in tt-mlir. Until then, change 1 has no
effect on a fresh build, because `requires_more_connectivity()` rejects a TORUS_X request
against a descriptor that declares no `dim_types` — **so the 2D-mesh fabric fix is only half
landed.** See §2 for the three on-disk copies and which one the runtime reads.

Baselines: tt-mlir at `121f69fa34` (branch `hshah/all-dsa-ops`; that commit is the earlier
`sparse_sdpa` scatter-decomposition work and is **already committed**, not part of this
diff). tt-metal at `f1f4ff75579` (`v0.75.0-dev20260717-29`).

```
tt-mlir  git diff --stat
 .../StableHLOLegalizeCompositePass.cpp        | 59 ++++++++++++++++
 .../Transforms/RegisterCustomShardingRule.cpp | 29 ++++++---
 runtime/lib/common/mesh_fabric_config.cpp     | 34 ++++++++++
 3 files changed, 115 insertions(+), 7 deletions(-)

tt-metal git diff --stat
 .../p150_x8_mesh_graph_descriptor.textproto   |  2 +-
```

---

## Result

`test_tensor_parallel_generation_deepseek_v32_3l[sparse-topk128]` on mesh `[2,4]`,
`optimization_level: 1`:

```
1 passed in 93.99s
token_ids: [119103, 13484, 122095, 122095, ...]  'hurican/questions/questions...'
```

Those token ids are **identical** to the Wormhole decomposition reference in
`dsa-blackhole-handoff.md` §5.2 — real Blackhole kernels reproduce the validated
decomposition path token-for-token.

Exported IR, sparse prefill graph (`ttnn_dsa_3l_topk128_g0_g1`):

| op | count |
|---|---|
| `ttnn.indexer_score_dsa` | 3 (one per layer) |
| `ttnn.topk_large_indices` | 3 |
| `ttnn.sparse_sdpa` | 3 |
| `ttnn.relu` / `ttnn.scatter` / `ttnn.softmax` | **0** |
| `ttcore.composite` | 0 |

and the query operand is `tensor<1x64x32x128xbf16>` — `Sq = 32 = 256/8`, i.e. genuinely
sequence-sharded across all 8 devices.

⚠️ **Prefill only.** In the sparse *decode* graph (`g0_g6`) `topk_large_indices` and
`sparse_sdpa` promote but `indexer_score_dsa` **silently decomposes** (`relu = 3`).
See [Open issues](#open-issues).

### Correctness gate: the A/B numerics test passes on the kernel path

`test_tensor_parallel_generation_deepseek_v32_dsa[mesh_shape0]` (mesh `[2,4]`,
`index_topk == padded prefill length`, so top-k covers every causally visible key and
sparse **must** equal dense):

```
1 passed in 100.24s
dsa_mode=auto: [30587, 122095, 122095, 122095, 122095, 122095, 122095, 122095]
dsa_mode=off:  [30587, 122095, 122095, 122095, 122095, 122095, 122095, 122095]
```

Token-for-token identical. Promotion verified independently from the exported IR, because
the test's own `DSA_TTNN_OPS` assertion is gated on `mesh_shape == [1, 4]` and so does not
fire on this box:

| | `indexer_score_dsa` | `topk_large_indices` | `sparse_sdpa` | `relu` |
|---|---|---|---|---|
| `ir_auto` (sparse) | 6 | 6 | 6 | **0** |
| `ir_off` (dense) | 0 | 0 | 0 | 0 |

So sparse ran entirely on real kernels with no decomposition, dense emitted no DSA op, and
the two agree exactly. This is the gate that would catch a mis-masked causal window from
change 4, and it is clean. (`indexer_score_dsa` is 6 rather than the split seen in the
`..._3l` run because decode is dense here — `dsa_decode_uses_sparse(256, 256)` is false —
so only the sparse prefill graph exists.)

Two test-side changes were needed to run it, neither of them a compiler change:

* `_run_dsa` used `optimization_level: 0`, which would have compared the *decomposition*
  against dense and proven nothing about the kernels. Raised to `1`.
* `_run_dsa` never shut its engine down. `vllm.LLM` falling out of scope is not enough —
  the EngineCore child process holds all 16 `/dev/tenstorrent` fds until told to stop, so
  the second engine in the test blocked forever on devices the first still owned (observed:
  both engines alive, 16 fds each, the second parked in futex waits, log frozen 23 min).
  Fixed with `finally: llm.llm_engine.engine_core.shutdown()` — the same call the conftest
  makes, but that only runs on the *failure* path (`pytest_runtest_makereport` with
  `report.failed`), so it never fired between the two engines of a passing test.

  **This appears to be the real cause of the skip this test carried**, which blamed "all
  three DSA ops inside one compiled model graph under TP sharding" and a host parked in
  `FDMeshCommandQueue::read_completion_queue`. The sparse run in fact completes and emits
  tokens; it is the *second* engine that hangs, whichever mode it is. That also explains the
  skip's "reproduced at DSA_MODEL_LEN 256 AND 1024, so it is shape-independent" (device
  contention is shape-independent) and why `dsa_mode='off'` alone finished in ~3 min (single
  engine). The skip has been removed.

---

## 1. `FABRIC_1D_RING` drops wrap edges

`runtime/lib/common/mesh_fabric_config.cpp`

**Symptom.** Every 2D-mesh TP run — DSA or not, plain Qwen3-0.6B reproduces it — aborted
during engine init:

```
TT_FATAL @ tt_metal/fabric/fabric.cpp:161: forwarding_direction.has_value()
Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)
```

**Cause.** `classifyLine()` correctly detects that each mesh row wraps (this box is a
2×2×2 hypercube, and `system_health` confirms the row wrap cables `5↔7` and `1↔3` are
present and link-up), so tt-mlir requests `FabricConfig::FABRIC_1D_RING`. tt-metal then
throws the ring away — `fabric_host_utils.cpp:66`:

```cpp
case tt::tt_fabric::FabricConfig::FABRIC_1D_RING: {
    if (is_ubb_galaxy) { return FabricType::TORUS_XY; }
    return FabricType::MESH;      // <-- P150_X8 is not a UBB galaxy
}
```

`mesh_graph.cpp:413` then lets the requested type win (`effective_fabric_type =
requested_fabric_type`), and `get_valid_connections()` only emits wrap coordinates under
`TORUS_X`/`TORUS_Y`. So no wrap edge is ever added to intra-mesh connectivity — while the
CCL layer, seeing a config *named* ring, still builds a ring schedule and asks fabric to
forward across the wrap. The two halves disagree and it aborts.

**Change.** Request a 2D torus config instead of `FABRIC_1D_RING` when the mesh is
genuinely 2D and a wrapping axis is longer than 2 elements. `FABRIC_2D_TORUS_{X,Y,XY}` map
straight to the matching `FabricType`, so the wrap edges actually get built:

```cpp
  bool rowAxisIsTorus = numCols > 2 && rowAxisConfig == FabricConfig::FABRIC_1D_RING;
  bool colAxisIsTorus = numRows > 2 && colAxisConfig == FabricConfig::FABRIC_1D_RING;
  if (numRows > 1 && numCols > 1 && (rowAxisIsTorus || colAxisIsTorus)) {
    FabricConfig torusConfig = FabricConfig::FABRIC_2D_TORUS_XY;
    if (!colAxisIsTorus)      { torusConfig = FabricConfig::FABRIC_2D_TORUS_X; }
    else if (!rowAxisIsTorus) { torusConfig = FabricConfig::FABRIC_2D_TORUS_Y; }
    return {torusConfig, perAxisConfig};
  }
```

The `> 2` guards matter: a wrap on a 2-element axis is degenerate (front and back are the
same adjacent pair), so `classifyLine` reports RING for *any* connected pair and a naive
check would claim a torus on the 2-wide dimension.

Axis mapping follows tt-metal's `get_valid_connections`: `TORUS_X` wraps `mesh_shape[1]`
(E/W), which is the axis a *row* line runs along; `TORUS_Y` wraps `mesh_shape[0]`.

⚠️ **Caveat.** This moves the box from **1D** to **2D** fabric — a larger behavioural
change than just enabling a wrap (different EDM setup and CCL kernels). It is verified for
these workloads, not broadly. The conservative alternative is to force `Topology::Linear`
at `runtime/lib/ttnn/operations/ccl/all_gather.cpp:38` (which currently passes
`std::nullopt` and so defers to ttnn's `get_usable_topology()` heuristic); that leaves the
two wrap cables idle and does not fix the underlying inconsistency for other CCLs.

Full analysis, including four suggested tt-metal-side fixes and the things that do *not*
work (`TT_MESH_GRAPH_DESC_PATH` is test-only; `TT_VISIBLE_DEVICES` cannot reorder chips):
[`fabric-1d-ring-torus-mismatch.md`](./fabric-1d-ring-torus-mismatch.md).

## 2. `p150_x8` mesh graph descriptor understates the box

`tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto`

```diff
-  device_topology { dims: [ 2, 4 ] }
+  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, RING ] }
```

Without `dim_types` both dims default to LINE, and change 1 then fails
`requires_more_connectivity(TORUS_X, MESH, [2,4])` — a 4-wide wrap is not degenerate, so
the request is rejected. Only the 4-wide dimension is declared RING; `RING` on the 2-wide
dimension would be degenerate.

**This edit is inert on its own** — verified across three runs with all three on-disk
copies patched, producing an identical abort. `mesh_graph.cpp:413` overwrites
`mgd_fabric_type` with the requested type, so the descriptor only raises the *ceiling*;
change 1 is what takes effect.

There are **three** copies on disk and the live one is the tt-mlir install tree (the plugin
loads `third_party/tt-mlir/install/lib/_ttnncpp.so`):

```
third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/   # source
  ...same.../build_Release/libexec/tt-metalium/tt_metal/fabric/mesh_graph_descriptors/                      # build
third_party/tt-mlir/install/tt-metal/tt_metal/fabric/mesh_graph_descriptors/                                # LIVE
```

A rebuild reinstalls from the source copy, so patching source keeps it after builds.

Upstream should register a torus variant for `P150_X8`: `mesh_graph.cpp:94-105` maps
`FabricType` → `ClusterType` → filename, and `TORUS_X`/`TORUS_Y`/`TORUS_XY` currently have
entries **only** for `GALAXY` and `BLACKHOLE_GALAXY` — `P150_X8` has no torus variant at all.

## 3. `sdy.all_slice` is a delta, not an absolute sharding

`lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp`,
`ShardyAllSliceToTTIRMeshPartitionConversionPattern`

**Symptom.**

```
error: failed to legalize unresolved materialization
  from ('tensor<16x2112xbf16>') to ('tensor<32x2112xbf16>') that remained live after conversion
module_builder.cc:839  Failed to convert from SHLO to TTIR module
```

**Cause.** Shardy emits `all_slice` as a **delta** from the operand's existing sharding.
Here the operand is the output of a `reduce_scatter` that has already scattered `_axis_0`:

```
%303 = stablehlo.dot_general(...)                                          -> 256x2112
%304 = stablehlo.reduce_scatter(%303) replica_groups=[[0,4],[1,5],[2,6],[3,7]]
                                                                           -> 128x2112
%305 = stablehlo.composite "sdy.all_slice" %304
         {out_sharding = #sdy.sharding<@mesh, [{"_axis_0","_axis_1"}, {}]>} -> 32x2112
```

(Those replica groups are exactly the `_axis_0` pairs on a `[2,4]` mesh, and Shardy's own
decomposition body confirms the intent — `@sdy.all_slice13(128x2112)` reshapes to
`4x32x2112`, a single division by 4.)

The pattern instead rebuilt the sharding from `out_sharding` as if the operand were fully
replicated, emitting one `ttir.mesh_partition` per axis: `128/2 = 64`, then `64/4 = 16`.
It then `replaceOp`'d a composite whose declared result is `32x2112` with a `16x2112`
value — the live unresolved materialization.

**Change.** Treat the composite's declared result type as authoritative: compute the
outstanding divisor per tensor dim as `operandDim / resultDim`, then consume axes
**minor-most first** (Shardy lists them major → minor, and the already-applied ones are
the major ones), skipping axes that are already reflected in the operand.

```cpp
    llvm::SmallVector<int64_t> pendingDivisor(operandType.getRank(), 1);
    for (int32_t dim : tensorDims) { /* ... */ pendingDivisor[dim] = inDim / outDim; }
    llvm::SmallVector<bool> applyAxis(tensorDims.size(), false);
    for (int i = (int)tensorDims.size() - 1; i >= 0; --i) {
      int64_t axisSize = meshShape[clusterAxes[i]];
      int64_t &pending = pendingDivisor[tensorDims[i]];
      if (pending > 1 && axisSize > 1 && pending % axisSize == 0) {
        applyAxis[i] = true; pending /= axisSize;
      }
    }
    if (llvm::any_of(pendingDivisor, [](int64_t d) { return d != 1; })) {
      return rewriter.notifyMatchFailure(srcOp, "Could not reconcile ...");
    }
```
plus `if (!applyAxis[i]) { continue; }` in the emit loop, and new failure paths for
rank mismatch / non-integral shard shapes.

**This is not DSA-specific.** It fires for any multi-axis `out_sharding` downstream of a TP
`reduce_scatter`, so it is the most broadly valuable fix here and worth upstreaming on its
own.

## 4. `indexer_score_dsa` forbade the only layout its kernel accepts

`lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp`,
`getIndexerScoreDsaShardingRule`

```diff
-  // Query sequence (query/weights dim 2, out dim 2): kNeedReplication.
+  // Query sequence (query/weights dim 2, out dim 2): kPassThrough.
   builder.addFactor({2, sdy::kNullDim, 2}, {2}, Sq,
-                    sdy::FactorType::kNeedReplication);
+                    sdy::FactorType::kPassThrough);
```
(plus a rewritten factor-design comment.)

**Cause — a direct contradiction between tt-mlir and tt-metal.** The kernel models the
query as sequence-parallel: `max_chunk_start()` reads a per-device rank off q's device
coords and `validate_chunk_start()` requires `T >= (max_rank + 1) * Sq`. A sequence-**
replicated** q on N devices therefore reports rank N-1 with `Sq == T`:

```
TT_FATAL @ indexer_score_device_operation.cpp:134: max_cs + Sq <= T
fullest-device chunk window [1792, 1792+256) exceeds T=256 (base=0, per-rank stride Sq=256)
```

(`1792 = 7 × 256`.) But the sharding rule marked `Sq` as `kNeedReplication`, whose old
comment read *"Cannot shard: the causal mask uses absolute query positions the fused op
recomputes locally, so a sharded Sq would mis-mask."* So the compiler **guaranteed the one
layout the kernel rejects**, and no amount of frontend annotation could survive: Shardy
dutifully gathered the sharding away around the op —

```
%207 = all_gather(%199) dim=2, groups=[[0,4],[1,5],[2,6],[3,7]]
%208 = all_gather(%207) dim=2, groups=[[0,1,2,3],[4,5,6,7]]
%213 = custom_call @tt.indexer_score_dsa(%208, ...)      : (tensor<1x64x256x128xbf16>, ...)
%214 = composite "sdy.all_slice" %213 {out_sharding = [..dim2 over both axes..]}
```

— ran the op replicated, then re-sliced the *output*. Net effect: `Sq = 256` at the kernel,
plus two pointless all-gathers.

The old comment's premise is what `chunk_start_idx` exists to handle. Each device's causal
window starts at `chunk_start_idx + rank * Sq`, so a **contiguous row-major split with
`chunk_start_idx == 0`** masks correctly on every device — device *r* holds absolute rows
`[r·Sq, (r+1)·Sq)` and the op computes exactly that offset. tt-mlir already passes
`chunk_start_idx = 0` (`runtime/lib/ttnn/operations/transformer/indexer_score_dsa.cpp:24`),
so no extra plumbing was needed.

`Key seq (T)` and `Head dim (D)` stay `kNeedReplication`: every query row scores against
all `T` keys, and `D` is contracted internally. `topk_large_indices` needed no change — it
already treats leading row dims as `kPassThrough`.

⚠️ **This factor is now permissive, not prescriptive.** Shardy will shard `Sq` only if a
sharding is present, so single-device and unsharded paths are unchanged. But it also means
**an incorrect split is no longer rejected by the compiler**. The op is only correct for a
contiguous row-major split over the whole mesh with `chunk_start_idx == 0`; a strided,
rotated, or partial-axis split would mis-mask silently. The tt-xla side that produces the
right layout is `TTIndexer._prefill_seq_shard_spec` in
`integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py`, which shards seq over the
compound `("batch","model")` axis — row-major over all 8 devices, matching the op's
`cluster_axis = None` flat linearization — and guards `seq_len % devices == 0` and
`Sq % 32 == 0`.

A stricter alternative worth considering upstream: keep the factor shardable but have the
runtime pass `cluster_axis` / `seq_shard_axes` explicitly so the op validates the layout
rather than inferring it. tt-metal `a07201edbbf` consolidates `cluster_axis` +
`seq_subshard_axis` into one `seq_shard_axes` argument; note that `[]` there means "flat
linear device order", **not** "replicated", so that uplift alone does not change any of
this.

---

## Open issues

1. **Decode's `indexer_score_dsa` silently decomposes.** `g0_g6` shows `relu = 3` with
   `idx = 0`, while `topk`/`sparse_sdpa` promote. Decode has `Sq = 1`, replicated, so it
   likely trips a TTNN verifier constraint (`Sq` is not a multiple of 32) and falls back
   with no diagnostic — the silent-fallback hazard of handoff-doc trap #1.

   **Do not "fix" this by forcing promotion.** With a replicated `Sq = 1` on 8 devices the
   deduction gives `base = T − ring·Sq = 248`, so device *r* would believe its single
   token sits at absolute position `248 + r` — eight devices, eight different causal
   windows, one correct. That is silently wrong output, strictly worse than the current
   decomposition. Decode needs an explicitly pinned `chunk_start_idx` (and probably
   `kv_len`) per user before it can go on the kernel path.

2. **`optimization_level: 1` is required for promotion.** At `0` all three composites
   inline their decompositions even on Blackhole with a correct `blackhole` system desc.
   The DSA test was changed from `0` to `1`; `_run_dsa` (the §9.2 A/B test) still uses `0`
   and would therefore exercise decompositions.

3. ~~The §9.2 A/B numerics test has not been run.~~ **Run and passing** on the kernel path
   — see [above](#correctness-gate-the-ab-numerics-test-passes-on-the-kernel-path). Note it
   validates **sparse prefill** only: decode is dense on both sides of that A/B
   (`index_topk == max_seq_len`), so nothing here exercises sparse decode on kernels, which
   is issue 1. The test's own promotion assertion is also still gated on
   `mesh_shape == [1, 4]`; making it arch-based (`get_torch_device_arch() == TTArch.BLACKHOLE`,
   per handoff-doc §9.4) would let it assert promotion on any Blackhole mesh instead of
   silently skipping that check, as it does on `[2,4]`.

4. **Change 2 (the tt-metal descriptor) is still uncommitted**, so on a fresh build the
   fabric half of this work does not apply — change 1 alone throws in
   `requires_more_connectivity()`. Landing it needs a tt-metal branch plus a
   submodule-pointer bump in tt-mlir. Changes 1/3/4 are committed on
   `hshah/all-dsa-ops` but **not pushed**.

5. **Changes 3 and 4 deserve upstream PRs to tt-mlir `main`**, independently of DSA:
   change 3 fires for any multi-axis sharding downstream of a TP `reduce_scatter`, and
   change 4 is a correctness fix for a rule that made its op unusable on multi-device
   meshes. Change 1 is a workaround; its real fix belongs in tt-metal (gate ring-vs-mesh
   connectivity on what the MGD provides rather than on `is_ubb_galaxy`, and fail at
   startup rather than inside a CCL program build).
