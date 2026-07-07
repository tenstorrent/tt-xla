# Escalation: fused-MoE decode (`all_to_all_dispatch_metadata` + `moe_gpt`/`moe_compute`) hangs on the single Blackhole galaxy

**To:** tt-metal fabric / CCL team
**From:** tt-xla GPT-OSS-120B fused-MoE-decode bring-up (branch `hshah/moe-compute-path`)
**Date:** 2026-07-06
**tt-metal:** `v0.74.0-dev20260621-14-g3a5f80334c1` (commit `3a5f80334c1506af81cfe8fb62fe62bf781d3074`)
**Hardware:** single Blackhole galaxy, 32 devices (physical topology **8×4**; 2 ethernet channels per axis hop)

---

## TL;DR

The fused-MoE decode collective — `ttnn.experimental.all_to_all_dispatch_metadata` feeding
`moe_gpt` / `moe_compute` — **hangs on the single Blackhole galaxy**. The a2a dispatch "completes"
(its output is readable and its completion semaphores fire — no `NSW`), but the downstream compute's
**`tilize` stalls forever at `cb_wait_front` (`CWFW`)** waiting for dispatched tokens that never fully
arrive, while the **ethernet fabric routers sit at `NWID`** (NoC write blocked). This is reproducible
**with tt-metal's own `test_moe_gpt_e2e`** (no tt-xla involved) once `num_links` is set to a value the
BH galaxy can satisfy. The hang is **invariant** to mesh shape, cluster axis, dispatch algorithm, and
compute op — pointing at the fabric / a2a token-delivery layer on this galaxy.

---

## Cleanest repro — tt-metal's OWN test (no tt-xla)

`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py::test_moe_gpt_e2e`
(mesh `(4,8)`, `FABRIC_1D_RING`, `dispatch_core_axis=ROW`, `cluster_axis=0`, `moe_gpt`).

Two problems, in order:

1. **The test aborts immediately on BH** because it hardcodes `num_links=4`:
   ```
   TT_FATAL: Requested link index 2 is out of bounds. 2 ethernet channels available
             to forward b/w src (M0, D0) and dst (M0, D3)   (fabric.cpp:163)
   ```
   `num_links=4` is a Wormhole-galaxy assumption (WH has 4 eth channels); the BH galaxy has **2**.
   ⟹ **The gpt_oss fused-MoE E2E was evidently never run/validated on the Blackhole galaxy.**

2. **With `num_links=2`, the test HANGS** (the same on the BH galaxy):
   ```
   INFO  run_test_moe_gpt_e2e:2092 - Running all_to_all_dispatch_metadata...
   INFO  run_test_moe_gpt_e2e:2110 - Dispatch complete, reading output for reference...   # a2a OK
   INFO  run_test_moe_gpt_e2e:2147 - Running moe_gpt...                                    # <- hangs here
   ```
   Host spins in `FDMeshCommandQueue::read_completion_queue` → `completion_queue_wait_front`
   → `Cluster::read_from_sysmem` (device never signals completion).

To reproduce:
```bash
cd <tt-metal>
# set every num_links=4 -> num_links=2 in test_moe_gpt_e2e.py (BH has 2 eth channels)
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 TT_METAL_WATCHER_DISABLE_STACK_USAGE=1 \
TT_METAL_WATCHER_DISABLE_SANITIZE_NOC=1 TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 \
pytest -svv "tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py::test_moe_gpt_e2e"
```

---

## Device-side symptom (waypoint watcher)

Captured on the hung run (both `moe_gpt` above and the tt-xla `moe_compute` decode show it):

- The stuck worker cores are the compute's **`tilize`** triple — `tilize_reader.cpp` / `tilize_writer.cpp`
  / `tilize_compute.cpp` (moe_gpt: `.../experimental/ccl/moe_gpt/device/kernels/`; moe_compute:
  `.../experimental/ccl/moe_compute/device/kernels/`) — all at **`CWFW`** (`cb_wait_front`, i.e. waiting
  for their input circular buffer = the dispatched tokens/metadata that never fully arrive).
- **No core is at `NSW`** → the a2a's completion/atomic-inc semaphores completed. The stall is in the
  **data path**, not the semaphore path.
- The **active-ethernet fabric routers on the stuck devices are at `NWID`** (NoC-write in flight/blocked)
  → token payload writes are wedged in the fabric.
- The a2a dispatch program itself has already finished (its output is read for the golden); the hang is
  in the *following* compute program, at its very first stage (tilize), starved for the dispatched data.

So: **a2a semaphores complete, but some dispatched token payloads never land, so the consumer's tilize
`cb_wait_front` never returns.**

Watcher dumps attached (tt-xla `moe_compute` decode, real captured inputs):
`real_hang_watcher_dump_rows01_cols47.txt` + `real_hang_kid_legend.txt`.

---

## The hang is INVARIANT to every knob we can turn from above the fabric

| Knob | Values tried | Result |
|---|---|---|
| Mesh shape | `(4,8)` and `(8,4)` (matching the physical 8×4) | both hang, same point/signature |
| `cluster_axis` | `0` and `1` | both hang |
| **EP factor / dispatch-ring size** | **EP-4 (dispatch along the physical size-4 axis) and EP-8 (dispatch along the physical size-8 ring, `(8,4)`/`cluster_axis=0`)** | **both hang** (EP-8 at the same `completion_queue` deadlock, 64 `NumaAwareExecutor` threads in `completion_queue_wait_front`) |
| **Fabric topology** | **`FABRIC_2D` (issue #7), `FABRIC_1D_RING` (Ring / TORUS_XY), `FABRIC_1D` (Linear / LINE)** | **all hang.** `FABRIC_2D` couldn't route the `linear::` multicast at all (a2a semaphore never completed). Both 1D fabrics route the multicast (a2a semaphores complete) but the token payloads still don't all land — `FABRIC_1D_RING` and `FABRIC_1D` hang identically. Notably, `FABRIC_1D` (Line) **drops the ring wraparound edge** and still hangs (it advanced slightly further — ~699s vs ~636s — then the same `completion_queue` deadlock), so the wraparound link is *not* the culprit. |
| `dispatch_algorithm` | `SPARSE_MCAST_SHORTEST_PATH`, `SPARSE_UNICAST`, `SPARSE_MCAST_LINEAR` | all hang |
| Compute op | `moe_gpt` (tt-metal's own) and `moe_compute` | both hang at `tilize` |
| `num_links` | 1 / 2 (4 is out of range on BH) | hang (2); 4 FATALs |

The **EP-8** case is worth calling out: every other `FABRIC_1D_RING` config dispatched EP-4 along the physical size-4 axis, so "invariant to cluster_axis" only meant "invariant to the axis *label* at EP-4". EP-8 dispatches along the **size-8 physical ring** — a genuinely different fabric route, and the closest we get on `FABRIC_1D_RING` to the old issue-#7 EP-8 (which was on the broken `FABRIC_2D`). It **still hangs**, so the a2a token-delivery failure is invariant to the dispatch-ring size too, not just the axis label.

Because it survives all of these, the defect is at the **a2a token-delivery / fabric layer on the
32-device Blackhole galaxy**, not in the choice of mesh/axis/algorithm/compute op.

### Device-level proof that the mesh shape is irrelevant

We replayed the **same fixed decode graph** with the **same routing** (`ttrt … --init randn --seed 42`)
under both mesh configs and captured the watcher stuck set:

| Config | a2a ring = | Stuck devices (tilize `CWFW`) |
|---|---|---|
| mesh `(4,8)`, `cluster_axis=0` | columns `{c,8+c,16+c,24+c}` | `20 21 22 23 28 29 30 31` |
| mesh `(8,4)`, `cluster_axis=1` | rows `{4r … 4r+3}` | `20 21 22 23 28 29 30 31` |

**Byte-identical stuck set.** Reshaping `(4,8)→(8,4)` is a row-major reshape over the same ordered
physical device list, so device ID *N* is the same physical chip in both. Under `(8,4)`/`cluster_axis=1`
the stuck devices `{20,21,22,23}` and `{28,29,30,31}` are **two complete, contiguous, axis-aligned
EP-4 rings** (rows 5 and 7); under `(4,8)`/`cluster_axis=0` those same chips were only a partial subset
of the column-rings. **Aligning the EP ring to a physically contiguous row did not move or clear the
hang** — the identical physical chips still failed to receive their dispatched tokens. The failure is
pinned to the physical fabric, independent of the logical mesh layout and cluster axis.

Data dependence: with random-but-in-range routing the *set* of stuck devices varies with the routing
(real captured inputs stalled a different pair of rows than the seed-42 garbage), but the failure mode
is always the same (tilize `CWFW` + eth `NWID`), and for a *given* routing it is invariant to mesh shape
(above). Trivial routing (all tokens to one expert, e.g. zero inputs) does **not** hang — it needs a
realistic multi-expert dispatch to manifest.

---

## Two additional tt-metal-side observations found along the way

1. **`test_moe_gpt_e2e` hardcodes `num_links=4`** (Wormhole assumption) → FATALs on the BH galaxy
   (2 eth channels). Please gate `num_links` on the arch/available channels so the BH path is runnable.

2. **`SPARSE_MCAST_SHORTEST_PATH` has a documented correctness bug for `cluster_axis=0`** — your own
   comment (`test_moe_gpt_e2e.py:2087`): *"computes hop distances using linearized device IDs (0-31)
   instead of column ring positions (0-3) … causing the sparse multicast to target the wrong device."*
   You work around it with `SPARSE_UNICAST`. We hit this too; note it is **separate** from the hang above
   (the hang persists under `SPARSE_UNICAST`). Worth fixing the default dispatch path regardless.

---

## What is NOT the cause (ruled out at the tt-xla / tt-mlir layer)

The fused decode reaches this point only after two tt-xla/tt-mlir compiler fixes (both IR-verified);
neither is related to this fabric hang, but they sit *above* it and are prerequisites to reach it:
- **Metadata drain-core reshard** — the a2a persistent metadata was resharded (`to_memory_config`)
  between the a2a drain core and `moe_compute`'s tilize drain core, deadlocking the collective. Fixed by
  aligning the drain cores (no reshard).
- **`expert_mapping` value** — must be the **global** linearized mesh coordinate; a compiler bug emitted
  the axis-local owner. Fixed. (This is the tt-xla side of the same class as observation #2 above.)

With both in place, the a2a runs and *completes*; the residual hang is entirely in the fabric/dispatch
data delivery described here.

---

## Asks

1. Is the fused-MoE a2a dispatch (`all_to_all_dispatch_metadata` → `moe_gpt`/`moe_compute`) **supported
   and validated on the single Blackhole galaxy**? Evidence suggests it has only been run on Wormhole TG
   (the `num_links=4` hardcode; your MoE nightly configs).
2. Why do some dispatched token payloads never land (eth `NWID`) while the a2a semaphores complete
   (`no NSW`) — is this a fabric credit/routing/backpressure issue specific to the BH galaxy's
   `FABRIC_1D_RING`/`TORUS_XY` on `(8,4)` with `num_links=2`?
3. A known-good BH-galaxy config (mesh shape, cluster_axis, num_links, dispatch algorithm) for the fused
   MoE decode, if one exists.

## Artifacts (available on request)
- Watcher dumps + k_id legend: `real_hang_watcher_dump_rows01_cols47.txt`, `real_hang_kid_legend.txt`.
- Mesh-invariance proof (same graph/seed, both mesh shapes → identical stuck set):
  `randn_hang_watcher_dump_rows23_cols47.txt` (mesh `(4,8)`), `hang_8x4_seed42_watcher.log` (mesh `(8,4)`).
- Faithful native repro of the tt-xla decode via captured real inputs (`ttrt run` + `load_tensor` of the
  exact dispatched-graph inputs) — reproduces the identical hang off the full model.
- Full bring-up narrative: `MOE_FUSED_DECODE_GALAXY_BRINGUP.md`, `MOE_FUSED_DECODE_HANDOFF.md`.
