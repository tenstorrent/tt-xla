# Handoff: porting GPT-OSS-120B fused-MoE decode from a Blackhole galaxy to a **Wormhole galaxy**

**Audience:** a fresh Claude (Opus 4.8) instance that will try to get `test_gpt_oss_120b_tp_moe_fused_galaxy`
running on a **Wormhole galaxy** (6U / TG, 32 chips). You have no prior context — this doc is self-contained.
**Branch:** `hshah/moe-compute-path` (tt-xla). **Written:** 2026-07-07 (from the Blackhole-galaxy bring-up).
**tt-metal pin:** `v0.74.0-dev20260621-14-g3a5f80334c1` (`3a5f80334c1506af81cfe8fb62fe62bf781d3074`).
**tt-mlir pin:** `6db2e09eb`. **tt-xla HEAD:** `2bcaa3b1d "debugging on bh galaxy"`.

> **Companion docs in the repo root (read for depth, but this doc supersedes their "next action"):**
> `MOE_FUSED_DECODE_BH_GALAXY_TTMETAL_ESCALATION.md` (the fabric-hang escalation — the real conclusion),
> `MOE_FUSED_DECODE_CONFIG_COMPARISON.md` (per-op config vs tt-metal's tested configs),
> `MOE_FUSED_DECODE_HANDOFF.md` (the prior Blackhole handoff), `MOE_FUSED_DECODE_GALAXY_BRINGUP.md`
> (full issue chain #1–#10), `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` (#9),
> `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md` (#4), `MOE_ALL_TO_ALL_DISPATCH_EXPLAINER.md`,
> `ISSUE7_A2A_DISPATCH_SEMAPHORE_HANG_RCA.md` (#7). Auto-memory: `moe-fused-galaxy-issue7-progress.md`,
> `ttmetal-moe-ops-galaxy-config.md`.

---

## 0. TL;DR — why you are on Wormhole, and what to do first

**The mission:** make `test_gpt_oss_120b_tp_moe_fused_galaxy` (in `tests/benchmark/test_llms.py`) run the
**fused** MoE decode path end-to-end on a 32-device galaxy. The fused decode pipeline is:

```
tt.moe_decode composite  →  all_to_all_dispatch_metadata (persistent mode, "the a2a")  →  moe_compute (tilize → matmul → selective_reduce_combine)
```

**Hard constraint from the user (do NOT violate):** keep the newer **`moe_compute`** op. Do **not** retarget to
the older `moe_gpt` path. (`moe_gpt` was only ever a cross-check; the deliverable is `moe_compute`.)

**Why you are on Wormhole now:** on the **Blackhole galaxy**, the fused decode **hangs**, and the hang was
run to ground and **escalated to the tt-metal fabric/CCL team** as a Blackhole-galaxy fabric defect. The
decisive evidence that this is Blackhole-specific:

- **tt-metal's own `test_moe_gpt_e2e` hangs on the Blackhole galaxy** (no tt-xla involved), and it
  **hardcodes `num_links=4`** — a Wormhole assumption (WH has 4 ethernet channels per axis hop; the BH
  galaxy has only **2**), which FATALs on BH before the run even starts.
- Every standard tt-metal MoE suite runs on **Wormhole galaxy (6U / TG) with `num_links=4`**.
- The gpt-oss **reference model gates the fused throughput path OFF on Blackhole**:
  `throughput_experts_supported_on_arch()` returns `not is_blackhole()`.

⟹ **The fused MoE decode is validated on Wormhole, not Blackhole.** Wormhole is the platform where this is
expected to work. That is the whole point of this port.

**Do this, in order (details in §5, §6):**

1. **First, prove the fabric on your WH galaxy with tt-metal's OWN test — no tt-xla, no compiler.**
   Run `test_moe_gpt_e2e.py::test_moe_gpt_e2e` **with `num_links` back at 4** (revert the BH `num_links=2`
   edits — see §4.2). If it PASSES on WH, the fabric/token-delivery layer works and the tt-xla path should
   follow. If it still hangs on WH, you have a tt-metal fabric bug independent of tt-xla — escalate with the
   §6 analysis.
2. **Reset the config knobs from the Blackhole *experiment* state to the reference path (§3).** The working
   tree is currently mid-experiment on Blackhole (`FABRIC_1D`/Linear, `(8,4)` mesh, EP-8). For Wormhole use
   the tt-metal-validated reference: **`FABRIC_1D_RING`, combine `Ring`, mesh `(4,8)`, `cluster_axis=0`,
   EP-4, `num_links=4`, `SPARSE_UNICAST`**.
3. **Rebuild** (`cmake --build build`, §5.2), **then run the 1-layer decode** (§5.3).

**Keep every correctness fix (§3, §4) — they are architecture-independent and IR-verified.** Only the
`num_links` values and the three Blackhole *experiment* knobs (fabric, combine topology, mesh/EP) are
BH-specific and must be set to the WH reference.

---

## 1. Wormhole vs Blackhole — the differences that matter

| Aspect | Blackhole galaxy (where it hung) | Wormhole galaxy (your target) | Consequence |
|---|---|---|---|
| **Ethernet channels / axis hop** | **2** | **4** | `num_links=4` FATALs on BH ("link index 2 out of bounds, 2 channels available", `fabric.cpp:163`) but is the **native, validated** value on WH. The BH hang may itself be a 2-channel fabric backpressure/credit issue that simply does not exist at 4 channels. |
| **tt-metal MoE test coverage** | Only 2026 *repro* tests + `moe_gpt_e2e` (which hangs) | **Standard nightly suites** (`test_moe_compute_6U.py`, `test_all_to_all_dispatch_metadata_6U.py`, `test_selective_combine_6U.py`, `test_all_to_all_dispatch_6U.py`) all run here, `num_links=4` | WH is the tested envelope; BH is not. |
| **gpt-oss reference model fused path** | **Disabled** (`throughput_experts_supported_on_arch() = not is_blackhole()`; demo `pytest.skip`s it) | **Enabled** | The model authors expect this to run on WH, not BH. |
| **Physical topology / torus** | 32 chips physical **8×4**, 2 eth/hop, **no both-axis wrap** (`computeMeshFabricConfig` deducing RING+RING → TORUS_XY was rejected) | 6U/TG 32 chips, typically **8×4**; the `_6U` tests are "**torus-descriptor gated**" → WH 6U may present a real both-axis torus | On WH a genuine `FABRIC_1D_RING`/torus wrap may be available where BH had none. Verify the WH system descriptor's wrap before assuming. |
| **`num_links` in tt-xla runtime** | `std::nullopt` → `get_num_links()` auto (see §4.1 note) | same auto path; should resolve to a WH-valid value | Confirm the auto value on WH (should be ≤4). |

**The single most important takeaway:** the Blackhole hang was `a2a-semaphores-complete-but-token-payloads-
never-land` at the fabric layer, with the ethernet routers stuck at `NWID` (NoC-write blocked). This is
exactly the kind of failure that a too-narrow (2-channel) fabric under `num_links` pressure would produce and
that a 4-channel WH fabric may not. **Test the fabric first (§0 step 1).**

---

## 2. What the fused decode does (so you can reason about the config)

- **`tt.moe_decode` composite** (emitted by the tt_torch `tt_moe_fused` backend for the single-token decode
  step; prefill stays on the dense bmm path) lowers in tt-mlir to:
- **`all_to_all_dispatch_metadata`** (persistent mode) — the "a2a": each source device sends its selected
  tokens to the device that owns each selected expert, along **one mesh axis** (the `cluster_axis`). It emits
  the dispatched tokens + per-token expert indices/scores as **persistent L1 height-sharded** buffers on a
  drain core. This op's per-axis multicast uses the **`tt::tt_fabric::linear::` (1D-fabric) API** → the
  dispatch axis **must be a 1D fabric** (Linear or Ring), never a 2D Mesh fabric.
- **`moe_compute`** — tilize (indices/scores/tokens) → grouped matmul over the local experts → clamped-SwiGLU
  activation + bias → **`selective_reduce_combine`** (Ring) that reduces each token's top-k expert outputs
  back to the source device.

Mesh `(rows, cols)` = `("batch","model")` by tt-xla convention. `cluster_axis` picks the EP/dispatch axis:
`cluster_axis=0` → dispatch along axis-0 (rows); `cluster_axis=1` → along axis-1 (cols). Experts are
**EP-sharded along the cluster axis and replicated along the other axis**; attention is **TP-sharded along
the other axis**.

---

## 3. The reference config to set for Wormhole (and how it differs from the current BH-experiment state)

tt-metal's validated full-model MoE config (`test_moe_gpt_e2e.py`, `models/demos/gpt_oss/...`,
`models/demos/deepseek_v3/...`) is: **mesh `(4,8)`, `cluster_axis=0` (EP-4 along the size-4 axis),
`FABRIC_1D_RING`, combine `Topology::Ring`, `SPARSE_UNICAST`, `num_links=4` (WH), global expert-mapping,
`DispatchCoreAxis.ROW`**, GPT-OSS shape (128 experts, hidden=intermediate=2880, top-4, swiglu+bias,
`output_height_shard_dim=4`, mux `(1,1)-(3,3)`).

**The working tree is currently NOT at the reference** — it is mid-Blackhole-experiment. Set each knob:

| Knob | File / location | **Current (BH experiment) value** | **Set to (Wormhole reference)** | Keep/why |
|---|---|---|---|---|
| Galaxy fabric | `pjrt_implementation/src/api/client_instance.cc` `computeFabricConfig`, `m_devices.size()==32` block (~L563) | `FabricConfig::FABRIC_1D` (2026-07-07 EXPERIMENT: drop the ring wraparound edge) | **`FabricConfig::FABRIC_1D_RING`** | Reference. The file's own comment says "Revert to FABRIC_1D_RING for the reference path." |
| Combine topology | `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp` (~L358) | `ttcore::Topology::Linear` (matches the FABRIC_1D experiment) | **`ttcore::Topology::Ring`** | Must match the fabric; the sibling a2a auto-derives Ring on FABRIC_1D_RING. |
| Mesh shape | `tests/benchmark/test_llms.py` `_galaxy_mesh_config_fn` (~L2011) | `return (8, 4), ("batch","model")` (EP-8 diagnostic) | **`return (4, 8), ("batch","model")`** | Reference EP-4 along the size-4 cluster axis. (Confirm your WH galaxy's physical row/col ordering — see note below.) |
| `cluster_axis` | `test_llms.py` `test_gpt_oss_120b_tp_moe_fused_galaxy` `register_tt_moe_backend(...)` (~L2238) | `cluster_axis=0` (with `(8,4)` mesh → EP-8) | **`cluster_axis=0`** (with `(4,8)` mesh → **EP-4**) | With `(4,8)`, axis-0 is the size-4 EP axis = reference. |
| Expert shard | `test_llms.py` `_gpt_oss_120b_moe_fused_galaxy_shard_spec_fn` (~L2173) | experts on `("batch", None, None)` (axis-0) | **experts on `("batch", None, None)`** (axis-0, now size-4) | Same spec; with `(4,8)` it is EP-4. Attention stays TP on `"model"` (size-8). |
| Dispatch algorithm | `third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/operations/ccl/all_to_all_dispatch_metadata.cpp` (~L56) | `SPARSE_UNICAST` | **`SPARSE_UNICAST`** (keep) | #10 fix — avoids the documented `SHORTEST_PATH` cluster_axis=0 hop-distance bug. |
| `num_links` (tt-metal tests) | all `num_links=` in the tt-metal test files (§4.2) | `2` (BH has 2 eth channels) | **`4`** (WH has 4) | Revert the BH workaround. This is likely material to the hang. |

> **Note on `(4,8)` vs `(8,4)` on your WH galaxy:** the physical device list is row-major, so the mesh shape
> chooses how the 32-chip list folds. The **reference** is `(4,8)`/`cluster_axis=0` = EP-4 along the size-4
> axis, matching `test_moe_gpt_e2e`. The BH experiment used `(8,4)`/`cluster_axis=0` = EP-8 along the size-8
> ring purely to test whether a longer dispatch ring changed the (BH) fabric failure — it did not. On WH,
> start from the reference `(4,8)`/EP-4. `test_llms.py` also carries a comment (~L2155) noting you can instead
> keep `(8,4)` and swap the axis *names* to `("model","batch")` + `cluster_axis=1` to get the same EP-4/TP-8 —
> either expression is fine; pick `(4,8)`/`cluster_axis=0` for the least surprise.

**Global expert-mapping (#9) is already in the reference form and must stay** — see §4.1.

---

## 4. ALL changes under `third_party/` (the required inventory)

Two submodules are modified. **None is committed** — all are working-tree edits on top of the pins above.
Full diffs: `git -C third_party/tt-mlir/src/tt-mlir diff` and
`git -C third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal diff`.

Tag legend: **[KEEP]** architecture-independent correctness fix — keep verbatim on WH.
**[WH-ADJUST]** BH-specific value to change for WH. **[EXPERIMENT]** a BH diagnostic knob to set to the §3
reference.

### 4.1 `third_party/tt-mlir/src/tt-mlir` — 8 files

1. **`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`** — `synthesizeMoeDecodeExpertMapping`
   (~L9382) + caller (~L9761). **[KEEP] — the headline correctness fix (#3 row-count + #9 value).**
   - The a2a reader reads **one mapping page per *mesh* device** indexed by the **global linearized mesh
     coordinate** (`row*num_cols + col`), so the mapping tensor must have **`numMeshDevices` rows** (32 on a
     4×8), not `clusterDevices` rows. (This was #3.)
   - The mapping **VALUE** must be the **global** mesh coordinate of the expert-owner, **not** the axis-local
     owner `e/expertsPerDevice ∈ [0,clusterDevices)`. The dispatch kernel uses the value *directly* as
     `target_device` (compared to `LinearizedSrcMeshCoord`, indexed into
     `send_preparation_buffer[...*NumDevices + target_device]`, fed to `is_configured_target()` which does
     `dest/MeshCols`, `dest%MeshCols`). An axis-local value misroutes every dispatch on a 2D mesh → owner
     devices' `moe_compute` tilize starve forever (this was the 29/32-device `CWFW` hang, #9).
   - The fix emits, for source device `d` and axis-local owner `a=e/expertsPerDevice`:
     `cluster_axis==0 → a*numCols + col(d)`; `cluster_axis==1 → row(d)*numCols + a`; 1D → `a`.
     (`numCols=meshShape[1]`, `row(d)=d/numCols`, `col(d)=d%numCols`.) Signature is now
     `(meshShape, clusterAxis, expertsPerDevice)`. Verified in dry_run IR (`main_const_eval_*`,
     `tensor<32x128xui16>`, values span [0,31], rows differ per source). **This is correct for any arch and
     any 2D mesh — keep it exactly.** Full write-up: `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`.

2. **`lib/Dialect/TTNN/IR/TTNNOps.cpp`** — `AllToAllDispatchMetadataOp::allocateBuffers` (~L3246).
   **[KEEP] — part of #8 (metadata drain-core reshard deadlock).** Reads an attribute
   `ttnn.moe_metadata_drain_core` (stashed by the pass in file 5) and drains the **persistent** metadata to
   the *same* core `moe_compute` reads it from (default `(0,0)` if absent). Prevents an inserted
   `to_memory_config` reshard of the persistent buffers, which deadlocks the collective (host spins in
   `completion_queue_wait_front`).

3. **`lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`** — `createMoeComputeOpOperandsWorkarounds` (~L1597).
   **[KEEP] — part of #8.** Pins `moe_compute`'s expert-indices/scores operands to **L1 HeightSharded**
   (`l1ShardedRmU16` / `l1ShardedRmBf16`). Without L1 they default to DRAM → "Only L1 buffers can have an
   associated circular buffer". With L1-sharded, layout propagation accepts the a2a's L1 metadata **in place**
   — no `to_memory_config`, matching the working native path.

4. **`lib/Dialect/TTNN/Transforms/OptimizerPasses/TTNNDeduceMoEComputeLayouts.cpp`** (~L69, ~L126).
   **[KEEP] — part of #8.** **Removed** the old `reshardTilizeInputToDrainCore` calls (the function is kept
   `[[maybe_unused]]` for reference). That cross-core reshard of persistent metadata was the deadlock; the
   in-place approach (files 2/3/5) replaces it.

5. **`lib/Dialect/TTNN/Transforms/TTNNAllocateDistributedOpBuffers.cpp`** (~L33, under
   `#ifdef TTMLIR_ENABLE_OPMODEL`). **[KEEP] — part of #8.** Before allocating buffers, walks each
   `MoeComputeOp`, computes `getMoeTilizeDrainCoreRangeSet`, and stashes it on the producer a2a
   (`mc->getOperand(0)`) as attr `ttnn.moe_metadata_drain_core` (via `ScopedSingletonDeviceGuard`). This is
   what file 2 reads. Doing it in the same pass as `allocateBuffers` keeps the stashed core from being dropped
   by an intervening pass.

6. **`lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp`** — `createToMemoryConfigOpIfNeeded` (~L411).
   **[KEEP] — #4 (ui16 sharded typecast FATAL).** The ui16→ui32 bounce that `to_memory_config` uses for
   `ttnn.copy` (tt-metal#41689) applies **only** to interleaved↔interleaved changes. Restricting it with
   `!currentLayout.hasShardedTensorMemoryLayout() && !info.output.isSharded()` avoids running `ttnn.typecast`
   on a sharded ui16 input whose page (`[..,4]` ui16 = 8B) is below the 16B L1 alignment (would FATAL). Arch-
   independent. Full write-up: `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md`.

7. **`lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp`** — `moe_compute` creation (~L358).
   **[EXPERIMENT → set Ring for WH].** Currently `ttcore::Topology::Linear` (matched to the BH `FABRIC_1D`
   experiment). moe_compute's combine kernel supports only Linear/Ring. **For the reference/WH path set
   `ttcore::Topology::Ring`** (matches `FABRIC_1D_RING` and the a2a's auto-derived Ring). See §3.

8. **`runtime/lib/ttnn/operations/ccl/all_to_all_dispatch_metadata.cpp`** — the runtime wrapper (~L56).
   **[KEEP] — #10.** Hardcodes `dispatch_algorithm = SPARSE_UNICAST` (the op default is
   `SPARSE_MCAST_SHORTEST_PATH`, which has a **tt-metal-documented** bug for `cluster_axis=0`: it computes
   ring hop distances from the global linearized device id instead of the intra-ring column position → targets
   the wrong device; see `test_moe_gpt_e2e.py:2087`). `SPARSE_UNICAST` routes per-target via the correct
   `get_route()`, matching tt-metal's gpt_oss reference (`fused_decode.py:147`). There is **no MLIR attr** for
   this — it is a runtime hardcode. Keep on WH. (Longer term: expose a per-op `dispatch_algorithm` MLIR attr,
   or fix SHORTEST_PATH in tt-metal.)
   > NB: `ttrt run … --init randn` earlier produced spurious "hangs" under *all* dispatch algorithms — those
   > were **garbage-input artifacts** (`ttrt` randn fills int token-id/index tensors with `randint(0,2^31)`,
   > wildly out of the 201088 vocab range), **not** a routing bug. Use realistic captured inputs (§5.4).

### 4.2 `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal` — 5 modified + 5 untracked

**Product code (1 file) — [KEEP]:**

- **`ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.cpp`**
  (~L38) — **#6.** Adds `topology_ = ::ttnn::ccl::convert_2d_to_1d_topology(topology_);` after
  `get_usable_topology(...)`. Collapses a 2D fabric to its 1D per-axis topology (**Mesh→Linear, Torus→Ring**),
  matching the other 1D CCL ops (all_gather, all_reduce, reduce_scatter). Without it, a `FABRIC_2D` mesh
  leaves `topology_==Mesh` and the 1D per-axis dispatch never completes. **On `FABRIC_1D_RING` this is a
  no-op (Ring→Ring) but harmless — keep it.** This is a HOST file compiled into `_ttnncpp.so` (see §5.2 build
  note). It should also be upstreamed to tt-metal.

**Test / harness code (4 files):**

- **`tests/nightly/tg/ccl/moe/test_all_to_all_dispatch_metadata_6U.py`** — **[KEEP] 2D-mesh golden fixes.**
  `tokens_per_device = batch // mesh_shape[cluster_axis]` (not `//devices`); persistent output/metadata first
  dim = `mesh_shape[cluster_axis]` (not `devices`). Correct for any 2D mesh (identical on 1D). Needed to run
  the `(4,8)` a2a repro.
- **`tests/nightly/tg/ccl/moe/test_moe_compute_6U.py`** — **[KEEP] 2D-mesh golden fixes.** The reference
  now places each token on the **linearized mesh coord** (`get_linearized_mesh_coord`) not
  `expert_id//experts_per_device`; total unique experts span **all** `num_devices` (not just the cluster
  axis); token count is derived from the input tensor. All identical on a 1D mesh; required so the golden is
  correct on a 2D mesh.
- **`models/demos/deepseek_v3/tests/test_optimized_moe_decode_block_tg.py`** — the a2a→moe_compute chain-
  repro driver. **[WH-ADJUST] `num_links`:** currently `num_links=2` for the a2a and `num_links=1` for the
  `reduce_scatter` ("BH has 2 eth channels; 4 is out of bounds"). **On WH set both back toward `4`** (the
  upstream values). Also carries a benign comment noting drain placement is not the hang cause.
- **`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py`** — tt-metal's own E2E.
  **[WH-ADJUST] `num_links`:** every `num_links=4` was lowered to **`num_links=2`** for BH. **Revert all of
  them to `4` on WH** (this is the very first thing to run — §0 step 1). The `dispatch_algorithm=SPARSE_UNICAST`
  and `topology=Ring` in this file are upstream (leave them).

**Untracked repro tests (5 files, under
`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/`) — [BH artifacts, optional for WH]:**
`test_a2a_metadata_4x8_fabric2d_repro.py` (the issue-#7 hang repro + the `_fabric1d_ring_axis0` variant),
`test_moe_compute_4x8_fabric1dring_repro.py`, `test_moe_decode_chain_4x8_fabric1dring_repro.py`,
`test_a2a_metadata_4x8_ttxla_wrongmapping.py` (the #9 A/B), `test_all_gather_4x8_ca1_repro.py`. These isolate
each op on a `(4,8)` galaxy and run under the tt-metal watcher (the PJRT watcher is broken — §7). They live in
a `blackhole_CI` path; for WH you would place equivalents under a WH galaxy path, but they are only needed if
you have to localize a WH-specific hang. **The `num_links` inside them is set for BH (1/2) — raise for WH.**

### 4.3 tt-xla-side changes (context; these are committed on the branch, not under `third_party`)

Committed in `2bcaa3b1d`/`ba2a316bf`. Relevant files: `pjrt_implementation/src/api/client_instance.cc`
(galaxy fabric override — §3), `tests/benchmark/test_llms.py` (`test_gpt_oss_120b_tp_moe_fused_galaxy` +
`_galaxy_mesh_config_fn` + `_gpt_oss_120b_moe_fused_galaxy_shard_spec_fn` — §3),
`pjrt_implementation/.../shlo_input_role_propagation.cc` (#1, input-role propagation for the composite),
`tt_torch/.../decode_utils.py` (#2, the `tt_moe_fused` backend / `tt.moe_decode` emission),
`tests/benchmark/benchmarks/llm_benchmark.py` (`TTXLA_DRY_RUN` → `dry_run=True` compile option, for IR dumps),
`.github/workflows/perf-bench-matrix.json` (`gpt_oss_120b_tp_moe_fused_galaxy`, `runs-on: galaxy-bh`,
`"skip": true` — **update `runs-on` for your WH runner and unskip once green**).

---

## 5. Environment, build, run

### 5.1 tt-xla env
```bash
cd /home/ubuntu/hshah/tt-xla && source venv/activate
# sanity: TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain, TTXLA_ENV_ACTIVATED=1
python -c "import jax; print(len(jax.devices('tt')))"   # expect 32 on the galaxy
```

### 5.2 Rebuild after editing tt-mlir / tt-metal sources
```bash
cmake --build build     # rebuilds the tt-mlir ExternalProject (+ its tt-metal) and relinks the plugin
```
- One `cmake --build build` handles all three layers: it recompiles the changed tt-mlir `.cpp`, reinstalls
  tt-mlir into `third_party/tt-mlir/install/`, and relinks `pjrt_implementation/src/pjrt_plugin_tt.so`.
  Confirm via `stat -c '%y'` that `pjrt_plugin_tt.so` and `third_party/tt-mlir/install/lib/libTTMLIR*.so`
  are newer than your edit (a stale `ls` mtime once misled us — trust `stat`, and grep the build log for
  the file you changed compiling + the final "Linking … pjrt_plugin_tt.so").
- **`_ttnncpp.so` gotcha (only if you edit a tt-metal HOST `.cpp` like #6):** native `ttnn` mmaps
  `…/tt-metal/build_Release/lib/_ttnncpp.so` while **PJRT loads `…/tt-metal/install/lib/_ttnncpp.so`**. After
  editing a tt-metal host file, rebuild it (`ninja -C …/tt-metal/build_Release _ttnncpp.so`) and copy the
  result to **both** `build_Release/lib/` and `install/lib/`. (Kernel `.cpp` files are JIT-compiled on device,
  no host rebuild.)
- Build-wait gotcha: `kill -0 <pid>` succeeds on a finished-but-unreaped ZOMBIE, so a `while kill -0` loop
  hangs — check `ps -o stat` (`Z`=done) or just re-run `cmake --build build` (fast if already built).

### 5.3 Run the test (and IR-only dry run)
```bash
# 1 layer reproduces the decode collective and is fastest:
pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
# IR-only (compiles, no device submit — safe even on a stranded board), dumps to modules/irs/:
TTXLA_DRY_RUN=1 pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
```
Decode IR = `modules/irs/ttnn_*_g1_*.mlir` (contains `all_to_all_dispatch_metadata`); flatbuffers =
`modules/fb_*_g1_*.ttnn`; the mapping constant is `main_const_eval_*` (`32x128xui16`). (Dry run ends with
"PCC … denominator is zero" — expected, no real output.) `required_pcc=0.90` in the test.

### 5.4 Native tt-metal repros / watcher (bypass PJRT — the watcher works here)
```bash
TTM=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
cd $TTM
env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD python_env/bin/python -m pytest -svv <repro>
# watcher (waypoint-only; full watcher overflows the ACTIVE_ETH kernel config buffer 30432>25600):
#   TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
#   TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1
```
Use the tt-metal `python_env` with a **CLEAN PYTHONPATH** — the tt-xla venv's `ttnn`/`torch` are stubs.
**FIRST WH run:** `test_moe_gpt_e2e.py::test_moe_gpt_e2e` (revert `num_links`→4 first).

### 5.5 ttrt (native flatbuffer runner; watcher works)
`ttrt run <fb.ttnn> --fabric-config FABRIC_1D_RING --program-index all --init randn --seed 42` executes a
flatbuffer natively. Built manually from `third_party/tt-mlir/src/tt-mlir` (its target is gated behind
`TTMLIR_ENABLE_BINDINGS_PYTHON=OFF`; `pip wheel tools/ttrt` with `TTMLIR_ENABLE_RUNTIME=ON
TT_RUNTIME_ENABLE_TTNN=ON …` then `pip install`). **Capture-replay of the real decode:** `TTXLA_EXPORT_TENSORS=1`
on the PJRT run dumps `modules/tensors/argN.tensorbin` (snapshot **all 27** g1 inputs); then
`TTRT_REPLAY_DIR=<dir> ttrt run <g1.ttnn> --fabric-config FABRIC_1D_RING` + watcher replays with faithful
per-device sharding (the only way we got a truthful watcher capture, since the PJRT watcher is broken).

---

## 6. The Blackhole residual hang — full analysis (what you are hoping WH avoids)

**Symptom (device-side, watcher-confirmed):** the a2a dispatch **completes** (output readable, completion
semaphores fire — **no `NSW`**), but the downstream compute's **`tilize` stalls forever at `cb_wait_front`
(`CWFW`)** waiting for dispatched tokens that never fully arrive, while the **active-ethernet fabric routers
sit at `NWID`** (NoC-write blocked). ⟹ **a2a semaphores complete, but some token payloads never land**, so
the consumer's tilize CB never fills.

**Progress before the wall (both KEEP-worthy):** with #8 (drain-core in-place) + #9 (global expert-mapping),
the decode flatbuffer **PASSES under `ttrt --init zeros`**, and with realistic routing the stuck-device count
dropped **29/32 → 8/32** — a clean **2×4 block** (`cols 4-7`, structural; the batch row is data-dependent).

**The hang is INVARIANT to every knob above the fabric** (this is why it was escalated):

| Knob | Values tried on BH | Result |
|---|---|---|
| Mesh shape | `(4,8)` and `(8,4)` | both hang, byte-identical stuck set (same physical chips) |
| `cluster_axis` | `0` and `1` | both hang |
| EP factor / ring size | EP-4 (size-4 axis) **and** EP-8 (size-8 ring) | **both hang** |
| Fabric | `FABRIC_2D` (#7), `FABRIC_1D_RING` (Ring/TORUS_XY), `FABRIC_1D` (Linear/LINE) | all hang. FABRIC_2D couldn't route the `linear::` multicast at all (a2a NSW, #7); both 1D fabrics route it (a2a completes) but payloads still don't all land. `FABRIC_1D` drops the wraparound edge and still hangs → the wrap link is not the culprit. |
| `dispatch_algorithm` | SHORTEST_PATH, SPARSE_UNICAST, SPARSE_MCAST_LINEAR | all hang (UNICAST still needed for the separate #10 correctness bug) |
| Compute op | `moe_gpt` (tt-metal's own) and `moe_compute` | both hang at `tilize` |
| `num_links` | 1, 2 (4 FATALs on BH) | hang |

**Reproduces in tt-metal's OWN `test_moe_gpt_e2e`** (no tt-xla) once `num_links=2` — see §0. **Ruled out at the
tt-xla/tt-mlir layer:** a2a alone PASSES; moe_compute alone PASSES; a2a→moe_compute chain (epd=2) PASSES;
all_gather PASSES; the whole decode with `--init zeros` PASSES. The deadlock needs realistic multi-expert
routing to manifest, and survives every mesh/axis/algo/op choice ⟹ the defect is in the **a2a token-delivery /
fabric layer on the 32-device Blackhole galaxy**.

**Why WH is expected to differ:** the failure signature (payloads wedged in the fabric at `NWID`, only under
real multi-expert traffic, on a 2-eth-channel galaxy where `num_links=4` is impossible) is consistent with a
Blackhole-galaxy fabric backpressure/credit/routing limitation. WH galaxy has **4 eth channels** and is the
**validated** platform for these ops. **If `test_moe_gpt_e2e` (num_links=4) passes on your WH galaxy, the wall
is gone.** If it does *not*, capture a watcher dump (§5.4/§5.5) and escalate to the tt-metal CCL team with the
`MOE_FUSED_DECODE_BH_GALAXY_TTMETAL_ESCALATION.md` analysis (the three "Asks" there apply to WH too).

---

## 7. Gotchas & operational notes

- **PJRT watcher is broken:** `TT_METAL_WATCHER=…` on a PJRT run aborts ("Watcher read invalid watcher.enable
  … `watcher_device_reader.cpp:593`") — PJRT device-init doesn't init all cores' watcher mailboxes. **Use
  `ttrt` (native) for the watcher** (§5.5).
- **A frozen log is usually NOT a hang:** each decode graph takes ~9 min to **compile** in tt-mlir, so the log
  goes silent while the device is idle. Confirm a real hang via `gdb`: a stuck
  `completion_queue_wait_front`/`read_from_sysmem` reader = device hang; `clock_nanosleep` executors +
  advancing program-time = slow compile. `ps %CPU` is a lifetime average — re-sample the live stack.
- **~5 graphs, not 2:** 2 graph *types* (dense prefill-shaped, fused single-token decode) but warmup + timed +
  PCC-reference passes and per-step torch_xla recompiles (KV position/constants differ) → decode steps don't
  cache-hit. That recompile-per-step is a separate **perf** bug to fix after correctness.
- **Board fragility (was severe on BH; verify on WH):** a `kill -9` of a hung 32-device fabric job strands an
  ethernet core (`llrt.cpp:566 … Try resetting the board`); `tt-smi -r` often failed → **power-cycle**. Prefer
  **one** native repro + watcher capture over repeated full PJRT runs. Ask the user to reset between hangs.
- **Hang host signature = collective deadlock:** host spins in `completion_queue_wait_front` (PJRT: many
  `NumaAwareExecutor` threads; ttrt/native: 1 `FDMeshCommandQueue` thread).
- **Watcher waypoints:** `GW`=go/idle wait, `CWFW`=CB-wait-front (consumer waiting on a CB — the tilize starve),
  `NSW`=NoC semaphore wait (the #7 signature, now gone), `NTW`=NoC transaction wait, `NWID`=NoC-write in
  flight/blocked (the fabric-payload wedge), `K`=kernel running. `PWW/PSW/DAPW/UAPW` are fast-dispatch infra,
  not the bug. Resolve `k_id[N]` via the legend at the top of `watcher.log`. `get_moe_tilize_drain_core` on the
  `(4,8)` config = core `(11,9)`.

---

## 8. First-day checklist for the Wormhole attempt

1. [ ] `git log --oneline -3` on tt-xla (expect `2bcaa3b1d`), and confirm the two submodule diffs match §4.
2. [ ] **Prove the fabric:** revert `num_links`→4 in `test_moe_gpt_e2e.py` (§4.2) and run
   `test_moe_gpt_e2e::test_moe_gpt_e2e` natively on the WH galaxy (§5.4). PASS ⟹ proceed; HANG ⟹ watcher +
   escalate (§6).
3. [ ] Set the §3 reference knobs: `client_instance.cc`→`FABRIC_1D_RING`; `TTNNResolveComposites.cpp`→`Ring`;
   `_galaxy_mesh_config_fn`→`(4,8)`; confirm `register_tt_moe_backend(cluster_axis=0)` + experts on `"batch"`;
   confirm runtime `SPARSE_UNICAST` (keep); set WH `num_links` where pinned.
4. [ ] Keep ALL **[KEEP]** fixes (§4.1 files 1–6, 8; §4.2 product #6 + 2D-mesh goldens).
5. [ ] `cmake --build build` (§5.2); verify plugin + `_ttnncpp.so` (if you touched a tt-metal host file) are
   fresh in BOTH `build_Release/lib` and `install/lib`.
6. [ ] `TTXLA_DRY_RUN=1 pytest … --num-layers 1 …` — sanity-check the IR (32-row global mapping, no
   `to_memory_config` on the persistent metadata).
7. [ ] `pytest … --num-layers 1 …` on device. Completes + PCC≥0.90 ⟹ decode is up; scale layers. Hangs ⟹
   ttrt+watcher localize (§5.4/5.5), compare the stuck-set to the BH §6 signature.
8. [ ] Update the auto-memory (`moe-fused-galaxy-issue7-progress.md`) and `perf-bench-matrix.json`
   (`runs-on` → WH runner, unskip) once green.

**Do not** retarget to `moe_gpt`. **Do not** drop any **[KEEP]** fix. The open question is entirely: *does the
WH galaxy's 4-channel fabric deliver the a2a token payloads that the BH galaxy's 2-channel fabric could not?*
