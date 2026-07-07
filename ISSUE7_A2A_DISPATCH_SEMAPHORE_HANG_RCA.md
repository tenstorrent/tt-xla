# Issue #7 RCA: `all_to_all_dispatch_metadata` hangs on a 4×8 Blackhole galaxy (dispatch semaphore never reaches target)

Standalone root-cause analysis + minimal repro for issue #7 of the fused-MoE-decode galaxy
bringup (`MOE_FUSED_DECODE_GALAXY_BRINGUP.md`). #1–#6 are fixed; #7 is the remaining blocker.

## TL;DR

Running GPT-OSS-120B fused MoE decode (`TT_MOE_FUSED_BACKEND_NAME`) on a 32-device Blackhole
galaxy (2D `(4,8)` mesh, FABRIC_2D, EP-8 along cluster_axis=1) **hangs** in the decode step.
Watcher on a minimal native repro shows the a2a dispatch core stuck in **`NSW` (NoC Semaphore
Wait)**: the a2a's dispatch-completion **semaphore never reaches its target count**. Local expert
compute has finished (all other workers idle); the fabric routers are up. So the cross-device
atomic-inc that signals dispatch completion
(`fabric_multicast_bidirectional_atomic_inc_1d`, running Linear-per-axis on the 8-device EP row)
**does not deliver the count the waiter expects** on this mesh. This is a tt-metal fabric/kernel
bug, isolated to the a2a's multicast atomic-inc count math for a per-axis 1D dispatch on a 2D mesh.

## Scenario

- Model/test: `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`, decode step.
- Mesh: `(4,8)` = `(batch, model)`; EP-8 experts along cluster_axis=1 ("model", the 8-wide row),
  DP-4 along axis 0. Fabric: `FABRIC_2D` → `Topology::Mesh` (no wrap; the galaxy is not a torus).
- Decode lowers `tt.moe_decode` → `all_to_all_dispatch_metadata` (persistent mode) → `moe_compute`.
- The a2a dispatches each token to the device(s) holding its top-4 experts, *within the
  8-device cluster row*, and signals completion via a cross-device global semaphore.

## Symptom

Through PJRT (full model), the host hangs in the first `finish` of `moe_compute`, which drains
the (never-completing) a2a:
```
ProgramExecutor::execute → ccl::run(MoeComputeOp) → ttnn::prim::moe_compute
 → create_global_semaphore → reset_semaphore_value
 → enqueue_write_shard_to_sub_grid → finish_nolock → read_completion_queue   (never returns)
```
Prelude `create_global_semaphore`/`allocate_moe_compute_semaphore` complete (they each `finish`
internally), so the semaphore-write machinery works — the stuck `finish` is the first one *after*
the a2a, i.e. it is waiting on the a2a's device work.

## Root cause (watcher-confirmed)

Watcher capture: `issue7_watcher_capture.log`. Last dump, on the dispatch device:
- a2a dispatch core (worker running `k_ids 584|583` = a2a writer|reader):
  **`NTW, NSW, W, W, W`** — BRISC **NoC Transaction Wait**, NCRISC **NoC Semaphore Wait**.
- all other workers: `GW,W` (go-wait / idle) ⇒ **local expert compute finished**.
- fabric ethernet routers (`acteth`, `fabric_erisc_router.cpp`): up (`FSCD`/`NSID`/`NWID`) — the
  routers are alive, but the cross-device increment never satisfies the waiter.

Conclusion: the a2a dispatch core waits forever on its **dispatch-completion NoC semaphore** — it
never reaches the target count. The cross-device atomic-inc multicast does not deliver the count
the waiter expects on the `(4,8)` galaxy.

Note topology is *not* the differentiator anymore: with #6's `convert_2d_to_1d_topology` the a2a
runs `Topology::Linear` (identical to the working n300-llmbox 2×4 20B case), yet it still hangs on
the 32-device galaxy. So the bug is in the Linear per-axis atomic-inc **count/target math or the
EP-row device mapping** on this mesh — not topology selection.

## Minimal native repro (fast, watcher-usable)

`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/test_a2a_metadata_4x8_fabric2d_repro.py`

Isolates the a2a metadata op — native tt-metal, no PJRT, no 120B model, ~1-2 min/iter:
- `test_a2a_metadata_4x8_fabric2d` — the repro: `(4,8)` / FABRIC_2D / cluster_axis=1 → **hangs**.
- `test_a2a_metadata_1x8_fabric1d_control` — `(1,8)` / FABRIC_1D → **passes** (proves harness+watcher).

Why this scenario was uncovered: every existing a2a(-metadata) test uses 1D sub-meshes
(`(1,8)`/`(1,16)`) on FABRIC_1D[_RING]; **none** exercises the full 2D `(4,8)` on FABRIC_2D.

### Environment (one-time)
The tt-xla venv's `ttnn`/`torch` are stubs; use the tt-metal `python_env` with a **clean
PYTHONPATH** (the tt-xla PYTHONPATH shadows torch/ttnn). `create_venv.sh` fails on `mmcv`
(a dev-only dep); minimal install works:
```
TTM=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal   # from tt-xla root
cd $TTM && source python_env/bin/activate && \
  uv pip install -e . torch==2.11.0 pytest pytz \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```
The a2a fix (#6, `convert_2d_to_1d_topology` in `all_to_all_dispatch_metadata.cpp`) must be built
into the `_ttnncpp.so` that **native ttnn mmaps** — `build_Release/lib/_ttnncpp.so` (not just
`build_Release/ttnn/` or `install/lib/`): `ninja -C build_Release _ttnncpp.so && cp
build_Release/ttnn/_ttnncpp.so build_Release/lib/_ttnncpp.so`.

### Runner fixes required (shared runner was 1D-only)
`tests/nightly/tg/ccl/moe/test_all_to_all_dispatch_metadata_6U.py` assumed `devices ==
dispatch_devices` (true only on 1D). For `(4,8)` these must use the dispatch axis:
- `tokens_per_device = batch // dispatch_devices` (was `// devices`).
- persistent output/metadata buffer first-dim = `dispatch_devices` (was `devices`).
Inert on 1D; correct on 2D. (Worth upstreaming as `(4,8)`/FABRIC_2D coverage.)

### Run
```
cd $TTM
REPRO=tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/nightly/test_a2a_metadata_4x8_fabric2d_repro.py
# control (expect PASS):
env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD python_env/bin/python -m pytest -svv \
    "$REPRO::test_a2a_metadata_1x8_fabric1d_control"
# repro (expect HANG) under waypoint watcher (full watcher overflows the ACTIVE_ETH kernel buffer):
env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH=$PWD \
    TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
    TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 \
    python_env/bin/python -m pytest -svv "$REPRO::test_a2a_metadata_4x8_fabric2d"
# inspect generated/watcher/watcher.log: the a2a worker core sits at NTW,NSW.
```
Operational note: a hang requires `kill -9`, which can strand an ethernet core
(`llrt.cpp:566 ... Try resetting the board`); a normal `tt-smi -r` may not clear it — a
power-cycle can be needed.

## Root cause: the a2a uses the 1D-fabric (`linear::`) multicast API on a FABRIC_2D fabric

The count math is **correct and internally consistent**, so this is a *transport/delivery* bug,
not a count bug:
- Sender (`writer_all_to_all_dispatch_metadata.cpp:1279`, `fabric_multicast_bidirectional_atomic_inc_1d`
  in `moe_utils.hpp`): for a source at `axis_position` p on a `dispatch_devices = N` line, Linear
  ranges are `positive(EAST) = (N-1) - p`, `negative(WEST) = p` → each device increments the other
  `N-1` devices exactly once. Ring uses `DoubleAntipodalAtomicInc` → `N`.
- Waiter (`reader_all_to_all_dispatch_metadata.cpp:180-183`):
  `expected_dispatch_device_inc = Linear ? (N-1) : N`; `noc_semaphore_wait(sem, expected)`.
- N-1 sent, N-1 awaited (Linear). Provably matches for a contiguous N-device row. Yet the waiter
  hangs at NSW ⇒ some increments never **arrive**.

Why they don't arrive: `fabric_multicast_bidirectional_atomic_inc_1d` issues the multicast via
**`tt::tt_fabric::linear::experimental::fabric_multicast_noc_unicast_atomic_inc`**
(`moe_utils.hpp:925/935`) — the **1D-fabric** API (single direction from `fabric_connections[dir]`,
scalar hop `range`, 1D packet-header routing). But the galaxy runs **FABRIC_2D** (Topology::Mesh),
whose routing is 2D. tt-fabric provides a distinct **`tt::tt_fabric::mesh::` API**
(`tt_metal/fabric/hw/inc/mesh/api.h:1335`) whose `fabric_multicast_noc_unicast_atomic_inc` takes
`dst_dev_id`, `dst_mesh_id`, and a `MeshMcastRange{e,w,n,s}` and calls `fabric_set_mcast_route(...)`
for 2D routing. A 1D-fabric multicast on a 2D-routing fabric does not route the increments to the
EP-row peers, so the dispatch-completion semaphore never reaches `N-1`.

The same applies to the token/metadata scatter-write path
(`fabric_multicast_bidirectional_scatter_write_ring_1d_async`, `writer:1264`), which also uses the
`linear::` multicast — it would mis-route on FABRIC_2D too.

## The fix (tt-metal fabric-kernel)

Make the a2a's per-axis multicasts use the **mesh (2D) fabric API** when running on a 2D fabric
(Mesh/Torus), instead of the `linear::` API:
- In `fabric_multicast_bidirectional_atomic_inc_1d` (and the scatter-write analog), when
  `is_2d_topology<Topology>()`, emit via `tt::tt_fabric::mesh::...::fabric_multicast_noc_unicast_atomic_inc`
  with `MeshMcastRange` = `{e = positive_range, w = negative_range, n = 0, s = 0}` for ROWS
  (`{n = positive_range, s = negative_range, e = 0, w = 0}` for COLS), and the target dev/mesh id
  for the axis peers. Keep the `linear::` path for genuine 1D fabrics.
- This requires the mesh fabric connection setup (`RoutingPlaneConnectionManager` / mesh sender)
  rather than the 1D `fabric_connections[4]` array — a real kernel change, and it must be validated
  on-device with the repro above.

Caveats to confirm during the fix: the mesh API's hop-count/route semantics vs the linear ranges
(is a "range" of R the same R hops?), whether the galaxy's EP-row is contiguous in mesh dev-id
space, and the interaction with `moe_compute`'s own combine (which also runs Linear on FABRIC_2D
and is currently unreached, so untested — it may need the same mesh-API treatment next).

### Why there's no config-only workaround
The a2a's per-axis multicast uses 2D directional sends (EAST/WEST for a row / NORTH/SOUTH for a
column), which only exist on a **2D** fabric (FABRIC_2D). But the multicast is issued through the
**1D** (`linear::`) API, which only routes on a **1D** fabric. So:
- FABRIC_1D / FABRIC_1D_RING (1D): the EAST/WEST directions don't exist → misroute (this was the
  original #5 hang on forced FABRIC_1D).
- FABRIC_2D / FABRIC_2D_TORUS_* (2D): directions exist, but the `linear::` transport doesn't route
  on a 2D fabric → increments don't arrive (#7, this doc).
No fabric-config choice satisfies both. The fix must change the *transport API* (linear → mesh),
not the fabric config. And there is currently **no existing kernel that uses the `mesh::` multicast
atomic-inc** (broadcast/reduce_to_root all use `linear::`), so this is a from-scratch fabric-kernel
implementation that needs on-device iteration (fast now via the repro above, but each hang requires
a `kill -9` that can strand an ethernet core → board reset/power-cycle between iterations).

## Artifacts
- Repro: `test_a2a_metadata_4x8_fabric2d_repro.py` (+ runner fixes in `test_all_to_all_dispatch_metadata_6U.py`).
- Watcher capture: `issue7_watcher_capture.log`.
- Chain + fixes #1–#6: `MOE_FUSED_DECODE_GALAXY_BRINGUP.md`.
