# Handoff: GPT-OSS-120B fused MoE decode bring-up on a 4×8 Blackhole galaxy

**Audience:** a fresh Claude (Opus 4.8) instance continuing this work with no prior context.
**Branch:** `hshah/moe-compute-path` (tt-xla). **Last updated:** 2026-07-06.
**Companion docs (read after this):** `MOE_FUSED_DECODE_GALAXY_BRINGUP.md` (full issue chain +
"Current status"), `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` (the mapping bug, incl. the
"Follow-up" section that is the key fix this cycle), `MOE_ALL_TO_ALL_DISPATCH_EXPLAINER.md`,
`ISSUE7_A2A_DISPATCH_SEMAPHORE_HANG_RCA.md`. Auto-memory: `moe-fused-galaxy-issue7-progress.md`.

---

## 1. Mission & current state (read this first)

**Goal:** make `test_gpt_oss_120b_tp_moe_fused_galaxy` (in `tests/benchmark/test_llms.py`) run the
**fused** MoE decode path (`TT_MOE_FUSED_BACKEND_NAME`, the `tt.moe_decode` composite → `moe_compute`
op) end-to-end on a 32-device (4×8) Blackhole galaxy. **Constraint from the user: keep the newer
`moe_compute` op — do NOT retarget to the older `moe_gpt` path.**

**Fused decode pipeline:** `tt.moe_decode` composite → `all_to_all_dispatch_metadata` (persistent
mode; the "a2a") → `moe_compute` (tilize → matmul → `selective_reduce_combine`).

**Where we are:** issues #1–#6 fixed previously. This cycle root-caused and fixed **two more device
hangs** (#8 metadata reshard, #9 expert-mapping value), both verified on-device. **Proof they work:** the
fixed decode flatbuffer PASSES under `ttrt` with `--init zeros`, and with realistic `--init randn` routing
the stuck-device count dropped 29→8. **The decode still hangs** with multi-expert routing, but the residual
is now localized (§4) to a fabric ring-dispatch stall on a 2×4 device block — a different, narrower bug.

**Config that matters:** the galaxy runs `FABRIC_1D_RING` + `cluster_axis=0` (EP-4 along "batch",
the 4-device axis), matching tt-metal's only tested full-model MoE path. (The original `FABRIC_2D` +
`cluster_axis=1` EP-8 choice was never validated and caused the original #7 hang.)

---

## 2. IMMEDIATE NEXT ACTION

Root cause is found (§4, issue #10: a documented tt-metal SHORTEST_PATH hop-distance bug for cluster_axis=0)
and the **fix is already applied** (runtime `dispatch_algorithm` `SHORTEST_PATH → SPARSE_UNICAST`, matching
the tt-metal gpt_oss reference). The next job is to build and verify it.

1. **Rebuild:** `cd /home/ubuntu/hshah/tt-xla && source venv/activate && cmake --build build` (~30 min;
   rebuilds tt-mlir runtime + relinks plugin/ttrt). Verify `libTTMLIRRuntime.so`/`ttrt` mtimes updated.
   NOTE: `ttrt run` uses the tt-metal `build_Release/lib/libtt_metal.so` runtime — the LINEAR change is in
   tt-mlir's runtime wrapper (`all_to_all_dispatch_metadata.cpp`), which ttrt links; confirm the rebuilt
   ttrt picks it up (reinstall the ttrt wheel if needed — see §5.5).
2. **Power-cycle the board** (the last ttrt run was `kill -9`'d → eth core likely stranded). Verify `ttrt query`.
3. **Verify the fix** with the seed that reproduced the hang:
   ```bash
   FB=$(ls -t modules/fb_gpt_oss_120b_tp_moe_fused_galaxy_1lyr_*_g1_*.ttnn | head -1)
   TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
   TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 \
   ttrt run "$FB" --fabric-config FABRIC_1D_RING --program-index all --init randn --seed 42
   ```
   Expect **PASS** (no `completion_queue` hang). Then spot-check a couple more seeds. NB the flatbuffer is
   unchanged (the fix is runtime-side), so the existing `run8573_g1` fb is fine — no recompile of the fb.
4. **If seed 42 passes:** re-run the **PJRT full 1-layer decode** (real tokens, the actual target):
   `pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`.
   If it completes → the decode is up. If it still hangs, watcher-localize via ttrt on the fresh fb.
5. **If seed 42 still hangs:** SPARSE_UNICAST wasn't sufficient → try `SPARSE_MCAST_LINEAR` or
   `SPARSE_MCAST_SPLIT_BW`, and/or escalate the SHORTEST_PATH hop-distance bug to the tt-metal CCL team
   with the §4 analysis (they documented it but the default is still buggy).

**To revert the candidate fix:** change `SPARSE_MCAST_LINEAR` back to `SPARSE_MCAST_SHORTEST_PATH` at the
same line and rebuild.

---

## 3. Fixes made THIS cycle (keep all of them — verified correct)

### #9 — expert-mapping VALUE: axis-local → global mesh coord  ← the headline fix
- **File:** `third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`,
  `synthesizeMoeDecodeExpertMapping` (line ~9414) + caller (~9770).
- **Bug:** emitted `value = e/expertsPerDevice` = axis-local owner ∈ [0,4). The a2a dispatch kernel
  uses that value **directly as a global mesh coord** ∈ [0,32) (`target_device`, compared to
  `LinearizedSrcMeshCoord`, indexed into `send_preparation_buffer[...*NumDevices + target_device]`,
  fed to `is_configured_target` which does `dest/MeshCols`, `dest%MeshCols`). On a 2D mesh axis-local
  ≠ global → tokens misroute → 29/32 devices' moe_compute tilize starve at `CWFW`.
- **Fix:** emit the global coord of the owner replica in the source's own cluster group (rows now
  differ per source): `cluster_axis==0 → a*numCols + col(d)`; `cluster_axis==1 → row(d)*numCols + a`;
  1D → `a`. (`a=e/expertsPerDevice`, `numCols=meshShape[1]`, `row(d)=d/numCols`, `col(d)=d%numCols`.)
  Signature changed to take `(meshShape, clusterAxis, expertsPerDevice)`.
- **Verified:** dry_run IR — the decode mapping constant `main_const_eval_*` (`tensor<32x128xui16>`)
  now has values spanning [0,31] matching `a*8 + d%8`, rows differ per source. Full write-up:
  `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md` → "Follow-up" section.

### #8 — metadata drain-core SPLIT + reshard hang
The a2a wrote persistent metadata to core `(0,0)` but `moe_compute` read it on `(11,9)`
(`get_moe_tilize_drain_core`), so the compiler inserted a `to_memory_config` reshard of the PERSISTENT
metadata → deadlock. Three tt-mlir changes eliminate the reshard (all in
`third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/`):
- `IR/TTNNWorkaroundsPass.cpp` `createMoeComputeOpOperandsWorkarounds` (~1590): pin moe_compute
  indices/scores operands to **L1 HeightSharded** (`l1ShardedRmU16`/`l1ShardedRmBf16`, ~1609). Were
  RowMajor+dtype only → defaulted to DRAM → "Only L1 buffers can have an associated circular buffer".
- `Transforms/TTNNAllocateDistributedOpBuffers.cpp` (~47): under `#ifdef TTMLIR_ENABLE_OPMODEL`, walk
  MoeComputeOps, compute `getMoeTilizeDrainCoreRangeSet`, and stash it on the producer a2a as attr
  `ttnn.moe_metadata_drain_core` (via `ScopedSingletonDeviceGuard`).
- `IR/TTNNOps.cpp` `AllToAllDispatchMetadataOp::allocateBuffers` (~3240): read that attr and drain the
  persistent metadata to that core (default (0,0) if absent).
- `Transforms/OptimizerPasses/TTNNDeduceMoEComputeLayouts.cpp` (~74/130): removed the
  `reshardTilizeInputToDrainCore` call (function kept `[[maybe_unused]]`).
- **Verified:** dry_run IR — decode metadata L1 `(11,9)` end-to-end, **zero `to_memory_config`**.

### #7 — config resolution (not a kernel fix)
Switched the galaxy from `FABRIC_2D`+`cluster_axis=1` to the tested `FABRIC_1D_RING`+`cluster_axis=0`:
- `pjrt_implementation/src/api/client_instance.cc` (~558): `m_devices.size()==32` →
  `FabricConfig::FABRIC_1D_RING`.
- `third_party/tt-mlir/.../TTNNResolveComposites.cpp` (~350): moe_compute combine topology = `Ring`.
- `tests/benchmark/test_llms.py` `test_gpt_oss_120b_tp_moe_fused_galaxy`:
  `register_tt_moe_backend(cluster_axis=0, use_interleaved=True, moe_decode_activation="swiglu", …)`;
  `_gpt_oss_120b_moe_fused_galaxy_shard_spec_fn` EP-shards experts on "batch" (axis 0).

### Earlier fixes #1–#6 (keep) — see `MOE_FUSED_DECODE_GALAXY_BRINGUP.md` "Fixes landed".
Files: `shlo_input_role_propagation.cc` (#1), `decode_utils.py` (#2), `StableHLOToTTIRPatterns.cpp`
row-count (#3), `TTNNDecomposeLayouts.cpp` (#4), `all_to_all_dispatch_metadata.cpp` (#6).

### Helper wired for dry_run IR dumps (keep)
`tests/benchmark/benchmarks/llm_benchmark.py` (~473): `**({"dry_run": True} if
os.environ.get("TTXLA_DRY_RUN") else {})` in the compile-options dict.

---

## 4. The residual hang — LOCALIZED (2026-07-06 via ttrt+watcher)

**It's a data-dependent fabric ring-dispatch issue, still at the moe_compute tilize, on a 2×4 device block.**

Localized by running the fixed decode flatbuffer under `ttrt` + watcher with different inputs:
- `ttrt run <fb_g1> --init zeros` → **PASSES** (`run_results.json` result=pass, outputs read back). Zero
  inputs make the router topk pick a trivial single-expert pattern → the multi-expert dispatch is barely
  exercised.
- `ttrt run <fb_g1> --init randn --seed 42` → **HANGS** (single `FDMeshCommandQueue`/
  `completion_queue_wait_front` reader stuck; watcher captured it).

**Watcher dump (randn hang), vs the pre-fix hang #2:**
- Pre-fix hang #2: **29/32** devices stuck at tilize `CWFW`.
- Post-fix randn: **8/32** stuck — a clean **2×4 block: rows 2-3 × cols 4-7** (devices 20,21,22,23,28,29,30,31).
  The other 24 devices went idle (`GW`) and advanced to the next program (`h_id` 294 vs the stuck 283).
- Stuck cores are STILL moe_compute `tilize_reader/writer/compute` at `CWFW` (waiting for dispatched tokens).
- **No `NSW` anywhere** → the a2a completion semaphores are fine (NOT the original #7 signature).
- Stuck devices' **eth fabric routers are at `NWID`** (NoC-write in-progress/blocked; idle devices' eth
  = `NSID`) → the fabric is trying to move tokens to these devices but the transfer is blocked.

### ROOT CAUSE (issue #10) — a KNOWN, tt-metal-DOCUMENTED bug in SPARSE_MCAST_SHORTEST_PATH

The seed sweep proved it's **data-dependent** (`ttrt --init randn` seeds 0/1/7 PASS; seed 42 HANGS). The
exact bug is **documented verbatim by tt-metal** in their GPT-OSS E2E test
(`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py:2087-2091`):

> "SPARSE_MCAST_SHORTEST_PATH has a bug for cluster_axis=0 where it computes hop distances using
> linearized device IDs (0-31) instead of column ring positions (0-3). For device 0→8, it gets distance=8
> (bit 7 in hop_mask) instead of 1 (bit 0), causing the sparse multicast to target the wrong device.
> SPARSE_UNICAST uses get_route() which correctly returns SOUTH (1 hop) for column-adjacent devices."

So the a2a's default dispatch algorithm (`SPARSE_MCAST_SHORTEST_PATH`, the bidirectional sparse multicast,
`dispatch_token_sparse_multicast_bidirectional`) mis-computes ring hops on cluster_axis=0 using the global
linearized mesh id instead of the intra-ring position → the multicast **targets the wrong device**, so the
real expert-owner never receives its tokens and its moe_compute tilize starves (`CWFW`). It's data-dependent
because the mis-targeting depends on which experts/devices the tokens select. (An earlier code-analysis
theory here — an "antipode stateful tie-breaker deadlock" — was plausible but WRONG; the documented
global-id-hop bug is the real cause. It happens to also implicate the antipode devices, matching the
watcher: rows 2,3 of the 4-ring stuck, eth `NWID`, no `NSW`.) Evidence:
`randn_hang_watcher_dump_rows23_cols47.txt`, `randn_hang_kid_legend.txt`, `seed0/1/7.log` (repo root).

### FIX (applied, needs rebuild + verify) — match the tt-metal gpt_oss reference: SPARSE_UNICAST

`third_party/tt-mlir/src/tt-mlir/runtime/lib/ttnn/operations/ccl/all_to_all_dispatch_metadata.cpp` (~line 56)
**hardcodes** the dispatch algorithm to the op's C++ default `SPARSE_MCAST_SHORTEST_PATH` (there is no MLIR
attr for it). Changed it to **`SPARSE_UNICAST`** — per-target point-to-point via the correct `get_route()`.
This matches tt-metal's own GPT-OSS reference, which uses `SPARSE_UNICAST` on exactly this (4,8) /
cluster_axis=0 / 4-device-ring config:
`models/demos/gpt_oss/tt/experts_throughput/fused_decode.py:147` and every `dispatch_algorithm=` in
`test_moe_gpt_e2e.py`. (The deepseek + 6U op-tests use the default SHORTEST_PATH but on configs/patterns
that don't trip the bug.) **Verify:** `cmake --build build`, then `ttrt run <fb_g1> --init randn --seed 42`
should PASS (flatbuffer unchanged — fix is runtime-side, no fb recompile); then a couple more seeds; then
the PJRT 1-layer decode. **Revert:** change `SPARSE_UNICAST` back to `SPARSE_MCAST_SHORTEST_PATH`. Longer
term, the SHORTEST_PATH multicast hop-distance bug should be fixed in tt-metal (or tt-xla should expose a
per-op `dispatch_algorithm` MLIR attr instead of the runtime hardcode).

**Ruled out (all PASS individually):** a2a alone; moe_compute alone; a2a→moe_compute chain (epd=2);
`all_gather cluster_axis=1`; multi-layer; the whole fused decode with `--init zeros`. The deadlock needs
realistic multi-expert routing (`--init randn`) to manifest.

---

## 5. Environment, build, and how to run

### 5.1 tt-xla env
```bash
cd /home/ubuntu/hshah/tt-xla && source venv/activate
# sanity: TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain, TTXLA_ENV_ACTIVATED=1
```

### 5.2 Rebuild after editing tt-mlir/tt-metal sources
```bash
cmake --build build     # rebuilds the tt-mlir ExternalProject + relinks the plugin (~30-45 min cold)
```
Verify the fix propagated: `libTTMLIRCompiler.so` and `pjrt_plugin_tt.so` (under
`third_party/tt-mlir/install/lib/` and `pjrt_implementation/src/`) mtimes must be newer than your edit.
The plugin loads `install/lib/`. **GOTCHA:** `kill -0 <pid>` returns success on a finished-but-unreaped
ZOMBIE, so a `while kill -0 PID` build-wait loop hangs forever — check `ps -o stat` (`ZN`=done) or just
re-run `cmake --build build` (fast if already built).

### 5.3 Full test / dry_run IR (dry_run = no board, cannot hang)
```bash
# minimal repro is 1 layer (hang reproduces at 1 layer, faster than 2):
pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
# IR-only (compiles, skips device submit) — dumps to modules/irs/, safe on a stranded board:
TTXLA_DRY_RUN=1 pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
```
Decode IR = `modules/irs/ttnn_*_g1_*.mlir` (the graph containing `all_to_all_dispatch_metadata`).
Flatbuffers = `modules/fb_*_g1_*.ttnn`. The mapping constant is `main_const_eval_*` (`32x128xui16`).
(dry_run fails at the end with "PCC … denominator is zero" — expected, no real output.)

### 5.4 Native tt-metal repros (bypass PJRT; the watcher works here)
Env: use tt-metal's `python_env` with a CLEAN PYTHONPATH (tt-xla's `ttnn`/`torch` are stubs):
```bash
TTM=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
cd $TTM
env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD python_env/bin/python -m pytest -svv <repro>
```
Repros (under `$TTM/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/`):
- `test_a2a_metadata_4x8_fabric2d_repro.py::test_a2a_metadata_4x8_fabric1d_ring_axis0` — a2a, tested path.
- `test_moe_compute_4x8_fabric1dring_repro.py` — moe_compute alone.
- `test_moe_decode_chain_4x8_fabric1dring_repro.py` — real a2a→moe_compute chain (epd=2; **scaling to
  epd=32 hits a harness weight-prep hardcode** in `models/demos/deepseek_v3/tests/test_optimized_moe_decode_block_tg.py`:
  the loop `for e in range(0, experts, 2)` + `cat([e, e+1])` hardcodes 2 experts/device → must generalize
  to `experts_per_device` to run epd=32).
- `test_a2a_metadata_4x8_ttxla_wrongmapping.py` — the #9 A/B (injects tt-xla's axis-local mapping).
Shared-runner (4,8) fixes live in `test_moe_compute_6U.py` / `test_all_to_all_dispatch_metadata_6U.py`.

### 5.5 ttrt (already built)
`ttrt` was built manually (its CMake target is gated behind `TTMLIR_ENABLE_BINDINGS_PYTHON`=OFF): from
`third_party/tt-mlir/src/tt-mlir`, `pip wheel` the `tools/ttrt` package with `TTMLIR_ENABLE_RUNTIME=ON
TT_RUNTIME_ENABLE_TTNN=ON TT_METAL_RUNTIME_ROOT=… TTMLIR_BINARY_DIR=… SOURCE_ROOT=…`, then `pip install`
the wheel into `venv`. `ttrt run <fb.ttnn>` executes a flatbuffer natively (watcher works, unlike PJRT).

---

## 6. Known-broken / gotchas

- **PJRT watcher is broken:** `TT_METAL_WATCHER=…` on the tt-xla PJRT run aborts with "Watcher read
  invalid watcher.enable on Device 0 worker core (0,0)…" (`watcher_device_reader.cpp:593`) — PJRT
  device-init doesn't init all cores' watcher mailboxes. **Use `ttrt` (native) for the watcher.**
- **A frozen log is usually NOT a hang here:** each decode graph takes ~9 min to *compile* in tt-mlir,
  so the log goes silent while the device is idle. Confirm a real hang via gdb: a stuck
  `completion_queue_wait_front`/`read_from_sysmem` reader = device hang; `clock_nanosleep` executors +
  advancing program-time = slow compile. `ps` `%CPU` is a lifetime average — a "95%" thread may be
  parked in a futex; re-sample the live stack.
- **~5 graphs, not 2:** only 2 graph *types* (dense prefill-shaped, fused single-token decode) but the
  benchmark runs warmup + timed + PCC-reference passes and torch_xla recompiles per distinct trace
  (KV-cache position/constants differ) → decode steps don't cache-hit. That recompile-per-step is a
  separate *perf* bug to fix after correctness.
- **Board degrades per hang:** a `kill -9` of a hung 32-device run strands an ethernet core
  (`llrt.cpp:566 … Try resetting the board`); `tt-smi -r` often fails → **power-cycle** (the user does
  this on request). Prefer one native repro + watcher capture over repeated full PJRT runs.
- **Hang host signature = collective deadlock:** host spins in `completion_queue_wait_front`
  (PJRT: many `NumaAwareExecutor` threads; ttrt/native: 1 `FDMeshCommandQueue` thread).

---

## 7. Watcher-dump reading playbook (from the hang-#2 localization)

Waypoints: `GW`=go/idle wait, `CWFW`=CB-wait-front (consumer waiting on a circular buffer),
`NSW`=NoC semaphore wait, `NTW`=NoC transaction wait, `K`=kernel running, `X`=n/a; `PWW/PSW/DAPW/UAPW`
are fast-dispatch (`cq_dispatch`/`cq_prefetch`) states = infra parked on the workers, not the bug.
Use waypoint-only flags (the `TT_METAL_WATCHER_DISABLE_*` set in §2) to avoid ACTIVE_ETH config-buffer
overflow. Resolve k_ids via the `k_id[N]: <path>` legend at the top of `watcher.log`. moe_compute
kernels: `tilize_reader/writer/compute`, `dm0/dm1/compute`, `selective_reduce_combine reader/writer`,
`tt_fabric_mux`. `get_moe_tilize_drain_core` on this config = core `(11,9)`. Tally state-per-device to
spot cross-device desync (last time: 29/32 devices stuck at tilize `CWFW`, 3 idle).

---

## 8. Repo state (as of 2026-07-06)

- **HEADs:** tt-xla `ba2a316bf` (branch `hshah/moe-compute-path`), tt-mlir `6db2e09eb`,
  tt-metal `3a5f80334c1`. All fixes are **uncommitted working-tree edits** (not yet committed).
- **tt-xla modified:** `client_instance.cc`, `shlo_input_role_propagation.cc`, `test_llms.py`,
  `decode_utils.py`, `llm_benchmark.py`, `.github/workflows/perf-bench-matrix.json`.
- **tt-mlir modified:** `StableHLOToTTIRPatterns.cpp`, `TTNNOps.cpp`, `TTNNWorkaroundsPass.cpp`,
  `TTNNDeduceMoEComputeLayouts.cpp`, `TTNNAllocateDistributedOpBuffers.cpp`, `TTNNDecomposeLayouts.cpp`,
  `TTNNResolveComposites.cpp`.
- **tt-metal modified:** `all_to_all_dispatch_metadata.cpp` (#6), the two `*_6U.py` runners (2D-mesh
  fixes), `test_optimized_moe_decode_block_tg.py` (chain-repro driver; num_links a2a=2/reduce_scatter=1
  for BH — the SPLIT-test edits were reverted, it's back to the aligned config). Plus 5 new untracked
  repro `.py` files (§5.4).
- Nothing is committed yet — when the bring-up lands, split into logical commits per issue and drop the
  working `.md` docs / `.log` files / repro tests as appropriate.

---

## 8b. REAL hang faithfully localized (2026-07-06) — capture-replay

Overcame the broken PJRT watcher: captured the 27 real g1 decode inputs via the plugin's `export_tensors`
(dumped before the hang), then replayed the g1 flatbuffer under `ttrt`+watcher with `load_tensor` (faithful
per-device sharding; run.py `TTRT_REPLAY_DIR` patch). It reproduces the REAL hang. Last watcher dump:
**8 devices stuck = rows 0-1 × cols 4-7, all at moe_compute `tilize` (`CWFW`); no `NSW`; eth `NWID`
(fabric write blocked).** The garbage seed-42 hang was rows 2-3 × cols 4-7 → **cols 4-7 (right half of the
model axis) is STRUCTURAL; the batch row is data-dependent.** With cluster_axis=0 each model column is an
independent 4-device dispatch ring; cols 0-3 complete, cols 4-7 don't receive their dispatched tokens →
tilize starves. Likely a tt-metal fabric topology/routing issue at the col-4 (right-half) boundary of the
(4,8) BH galaxy. Evidence: `real_hang_watcher_dump_rows01_cols47.txt`, `real_hang_kid_legend.txt`. Capture/
replay how-to: `TTXLA_EXPORT_TENSORS=1` on the PJRT run dumps `modules/tensors/argN.tensorbin`; snapshot ALL
of g1's inputs (27, not 20!); `TTRT_REPLAY_DIR=<dir> ttrt run <g1.ttnn> --fabric-config FABRIC_1D_RING` +
watcher. (ttrt `load_tensor` was copied in from the 20:26 build-tree module.)

## 9. TL;DR for the impatient
1. #8 + #9 fixes work: decode fb PASSES under `ttrt --init zeros`; `--init randn` stuck devices 29→8, and
   3/4 random seeds pass entirely → the residual is data-dependent. Keep #7/#8/#9.
2. **Root cause of residual (issue #10):** a KNOWN, tt-metal-documented bug —
   `SPARSE_MCAST_SHORTEST_PATH` computes ring hop distances with global device ids (0-31) instead of
   column-ring positions (0-3) on cluster_axis=0 → the sparse multicast **targets the wrong device** →
   real owner's tilize starves (`CWFW`). Documented in `test_moe_gpt_e2e.py:2087`.
3. **Fix APPLIED:** runtime `all_to_all_dispatch_metadata.cpp` dispatch_algorithm
   `SHORTEST_PATH → SPARSE_UNICAST` — matches tt-metal's gpt_oss reference (`fused_decode.py:147`,
   `test_moe_gpt_e2e.py`), which uses SPARSE_UNICAST on exactly this (4,8)/cluster_axis=0 config.
4. **Next:** `cmake --build build` → power-cycle → `ttrt run <fb_g1> --init randn --seed 42` (expect PASS)
   → PJRT 1-layer decode. If still hangs: try `SPARSE_MCAST_LINEAR`/`SPLIT_BW`, escalate to tt-metal CCL.
5. Evidence: `randn_hang_watcher_dump_rows23_cols47.txt`, `randn_hang_kid_legend.txt`, `seed{0,1,7}.log`.
