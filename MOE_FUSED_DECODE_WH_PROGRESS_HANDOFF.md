# Resume/handoff: GPT-OSS fused MoE decode on the Wormhole galaxy (tt-xla)

**Audience:** a future Claude (or engineer) continuing this work with little/no prior context.
**Written:** 2026-07-08. **Repo:** `/home/hshah/tt-xla` (local NVMe), branch `hshah/moe-compute-path`,
HEAD `7ff776f62`. **Board:** Wormhole (WH) 32-chip galaxy (TG/6U). **Original brief:**
`MOE_FUSED_DECODE_WORMHOLE_HANDOFF.md` (the doc that started this).

---

## ⚠️ UPDATE 2026-07-08 (~13:00 UTC): full 24-layer 20B decode DEADLOCKS — depth-dependent (mechanism not yet pinned)

The full-depth run tracked in §6 item 1 did **NOT** pass — it **hung** (distributed deadlock, all 32 chips parked
on a fabric barrier, in the main decode graph). This is a **new blocker** on top of the weight-prep perf issue; the
"proven on WH" claim below holds **only at depth = 1**.

**PROVEN: the hang is depth-dependent** (1 layer passes, 24 hangs; everything else identical), which rules out all
the config-level suspects (see below). **LEADING HYPOTHESIS:** tt-xla compiles all 24 layers into ONE program and
runs the 24 ring-collectives **back-to-back on device with no host synchronization between layers** — so a later
layer's collective likely starts before the previous one has drained the fabric ring on all 32 chips → deadlock.
The mechanism is **not yet pinned** — needs a depth-bisect.

**CORRECTION (an earlier overclaim, walked back):** I first attributed this to "48 never-freed global semaphores
exceeding device capacity / aliasing." That is **wrong as a mechanism**: a global semaphore is just an **L1 buffer,
4 bytes/core** (`tt_metal/impl/buffers/global_semaphore.cpp:49-65`) — no fixed count cap; 48 is kilobytes (trivial
vs ~1.5 MB L1); L1 OOM would **`TT_FATAL` (crash), not hang**, and the allocator gives distinct addresses (no silent
aliasing). The 48-semaphore pile-up is a real but **harmless symptom** of the all-in-one-program structure, not the trigger.

### Why depth-dependent (rules out the config hypotheses)
1-layer and 24-layer tests are identical except layer count (same (4,8) mesh, cluster_axis=0, FABRIC_1D_RING,
fused `moe_compute`, default descriptor). So the cause MUST scale with depth — which rules out the mesh/fabric
config (a missing torus descriptor / open ring would hang at 1 layer too), the fused-vs-unfused op choice (fused
ran fine at 1 layer), and `layer_id` (verified: only a compile-time weight-buffer offset, `dm0.cpp:185`, correct
at L=1 per op). What scales with depth: the number of back-to-back on-device ring-collectives in one program (and,
as a symptom, the per-layer global semaphores — see the correction above; they are not the trigger).

### Evidence (from `modules/irs/…g1….mlir`, the compiled decode graph)
- `@main` allocates **24 `allocate_moe_compute_semaphore` + 24 `create_global_semaphore` = 48 global semaphores**,
  ALL emitted in the prelude (lines 5383–5430) **before** the first collective (`all_to_all_dispatch_metadata` @5571),
  and held live through all 24 layers (…@8347). **Zero semaphore deallocations** in the graph.
- Live global semaphores scale directly with depth (1 layer → 2; 24 → 48) — but this is a symptom that tracks the
  layer count, NOT the demonstrated trigger (see the correction above); the number of back-to-back collectives scales too.
- The native `test_moe_compute_6U` runs `num_layers=5` but as a **host loop of separate per-layer dispatches**, so
  only ~2 semaphores are ever live at once — it recycles per layer and never hits this. So its multi-layer PASS
  does NOT cover the "many collectives in one program" case; tt-xla is the only path that does.

### Semaphore-emission detail (a symptom; sharing is a cleanup, probably NOT the fix)
- Per-op materialization: **`MoeComputeOp::allocateSemaphores`** (`lib/Dialect/TTNN/IR/TTNNOps.cpp:3167`) and
  **`AllToAllDispatchMetadataOp::allocateSemaphores`** (`TTNNOps.cpp:3301`) each mint a FRESH semaphore per op,
  inserted "in the prelude (after GetDeviceOp) so it is trace-hoistable" (`:3181`), guarded only by per-op
  `hasUnboundSemaphores()` — no pooling, no dealloc. Driver: **`TTNNAllocateDistributedOpSemaphores.cpp`** walks
  every DistributedOpInterface op (no cross-op coordination). This is WHY the IR has 48; it's an inelegance, not the bug.
- Sharing one semaphore across the same-config ops would be a reasonable cleanup, but since each collective waits on
  its own distinct, correctly-addressed semaphore, it likely will NOT fix the hang. Don't chase it before the bisect.

### BISECT DONE (2026-07-08 ~13:30) — hangs at N=2, the MINIMUM
- Re-confirmed 1-layer baseline PASS (prefill 0.999905 / decode 0.999237, 7:16, bit-identical to original; mesh healthy).
- **N=2 HANGS** (42 min vs 7 min baseline; hot thread pegged 99.9 % R, `voluntary_ctxt_switches` frozen at 155, log
  silent ~35 min — the same busy-poll fingerprint as N=24). SIGTERM released the mesh cleanly (34 W→23 W, jax.devices=32).
- ⟹ It hangs at the **minimum** multi-collective depth. The trigger is the **1→2 consecutive-collective transition**
  (insufficient synchronization / fabric-ring quiescence between back-to-back on-device fused-MoE collectives), **not**
  accumulation — N=2 has only ~4 semaphores / KB of L1, so all count/pressure theories are dead. No need to test 4/8/16.
- **Minimal reproducer: `--num-layers 2`** (~10 min to reach the hang). Debug here.
- **Detection:** key on **hot-thread `voluntary_ctxt_switches` frozen while pegged ~100 % R**. Device power 34 W just
  means "mesh held by a live process" (23 W = released) — it is NOT a hang discriminator (I over-weighted it earlier).

### 2-LAYER IR ANALYSIS (2026-07-08) — the missing barrier, located
On the minimal repro (`modules/irs/…2lyr…g1….mlir` `@main`):
- The two collectives ARE data-ordered via the residual stream: L0 `%97 = moe_compute` → `to_layout` (807) →
  `multiply` (813) → …L1 attn/norm/router… → `%141` (910) → L1 `all_to_all_dispatch_metadata` (926) → L1 `moe_compute` (933).
- **But that data dep is the ONLY thing ordering them.** `@main`'s op inventory has **no barrier / wait / semaphore-wait /
  fabric-drain** between the collectives, and `moe_compute` returns only a data tensor (`-> tensor<4x8x2880xbf16>`, no
  completion token).
- **Mechanism (hypothesis):** `moe_compute` runs an internal cross-device ring combine; "local output tensor ready"
  (which satisfies the SSA dep) ≠ "ring quiesced on all 32 chips / fabric drained." With no barrier, L1's dispatch injects
  into the SAME ring while L0's combine tail is in flight → ring collision → deadlock. Explains 1-pass / 2-hang / native-
  host-loop-pass (host blocks on full completion between dispatches = implicit fabric barrier the fused path lacks).
- **Secondary:** `%31` is a SINGLE shared `ttnn.empty(4x8x2880)` (line 657) used as the combine-output buffer by BOTH
  layers (L0 `%97` aliases it → `deallocate(%97,force=false)` @808 → L1 reuses `%31` @933) — same "ring finished?" assumption.

### COMBINE COMPLETION SEMANTICS TRACED (2026-07-08) — kernel-level lead RETRACTED
- **Runtime handler** (`runtime/lib/ttnn/operations/ccl/moe_compute.cpp`) calls `::ttnn::experimental::moe_compute(...,
  compute_only=false)` and returns — no post-op synchronize.
- **Combine kernel** (`dm1.cpp`) full path ends with `noc_semaphore_wait` + `noc_semaphore_set(0)` +
  `noc_async_posted_writes_flushed` (departure-only), vs `compute_only` which uses `noc_async_write_barrier` (completion).
- **RETRACTION:** I initially called the posted-flush the "smoking gun" and proposed adding an ACK barrier there. The
  author comment (`dm1.cpp:51-57`) shows this is **deliberate and safe**: in the full path the **matmul↔combine
  `combine_semaphore` handshake provides receiver-side ordering, so posted writes are safe + faster**; `compute_only`
  needs the ACK barrier ONLY because it has no consumer/handshake (host reads L1 directly → NaN/Inf without it).
  So the posted-flush is doing its intended job — **NOT a bug, and the ACK-barrier kernel edit is the WRONG experiment.**
- ⟹ The kernel-level mechanism of the back-to-back hang is **NOT pinned.** If the `combine_semaphore` handshake fully
  orders/quiesces the ring per-op, the "fabric undrained on return" story is doubtful. The real trigger is more likely
  something *between/across* the two collectives that single/host-looped usage never exercised (per-op semaphore
  reset/reuse, fabric/EDM connection state, shared `%31` output buffer, a2a-dispatch↔prior-combine interaction) — unconfirmed.

### STILL SOLID vs OPEN
- SOLID: depth-dependent; N=2 hangs (minimal repro); 1 passes; NO barrier/CCL between the two collectives in tt-xla's
  `@main` (only a data dep); reference `tt_moe_decode` is single-layer/eager and never chains back-to-back in one graph.
- OPEN: the exact kernel/fabric resource that deadlocks when two fused-MoE collectives run back-to-back in one program.

### FIX / NEXT (revised)
- **Do NOT** do the `dm1.cpp` ACK-barrier edit (targets a safe-by-design path).
- **Right experiment:** insert a **lowering-level full device-sync / barrier between the two collectives** in `@main`
  (replicate the host-loop's implicit program-boundary quiesce), re-run N=2. Clears ⟹ structural hypothesis confirmed,
  fix lives in the lowering (barrier between fused-MoE ops), agnostic to the exact resource. Doesn't clear ⟹ instrument
  the two-collective sequence directly.
- **NEXT:** empirically confirm — add a drain/barrier between the 2 layers, re-run N=2; hang clears ⟹ confirmed. Then
  decide op-level vs lowering-level for the real fix.

### BARRIER TEST RUN (2026-07-08 16:xx) — DID NOT FIX; only partly conclusive
- Implemented as an **env-gated per-op mesh `distributed::Synchronize`** in `runtime/lib/ttnn/program_executor.cpp`
  (gated on `TTXLA_SYNC_AFTER_OP=1`, no-op otherwise), rebuilt `TTMLIRRuntime`, installed to `third_party/tt-mlir/install/lib`.
- Re-ran N=2 with `TTXLA_SYNC_AFTER_OP=1` → **STILL HANGS identically** (busy-poll R thread, `voluntary_ctxt_switches`
  frozen at 1435, ~48 min, log frozen post-prefill; SIGTERM 143; mesh released clean, 14 W).
- **Calibration (over-billed this as "decisive"):** `distributed::Synchronize` drains the **command queue** (kernel
  completion) but likely does NOT drain the combine's **posted-write fabric tail** (posted = no completion ACK → CQ can
  report done while writes are still in flight). So a CQ-sync is NOT a guaranteed fabric drain. Hence:
  - **Definitive:** a per-op CQ `Synchronize` does NOT fix the hang → the simplest "sync between collectives" fix is insufficient.
  - **Not refuted:** the fabric-contention hypothesis (my barrier may not have actually drained the fabric).
  - Also: hang is at **layer-1's** collective *even after* layer-0 ops were CQ-sync'd → not a simple CQ-ordering race.
- Live interpretations: **(A)** persistent device/fabric state a CQ-sync can't reset (→ EDM connection trace);
  **(B)** need a genuine fabric-level drain (bigger edit than a CQ sync).
- **CLEANUP: DONE** — the env-gated `program_executor.cpp` barrier edit was reverted (git diff clean) and the runtime
  rebuilt/reinstalled (gate string confirmed gone from the installed lib).

### EDM / FABRIC-CONNECTION LIFECYCLE TRACED (2026-07-08)
Architecture = **per-op fabric mux + session-persistent EDM**:
- The **EDM** (ethernet-router firmware) is set up ONCE at `FABRIC_1D_RING` init (`client_instance.cc`) and lives for
  the whole process — it holds fabric routing + buffer/credit state (session-persistent).
- Each collective op (a2a-dispatch, moe_compute combine) launches a **per-op `tt_fabric_mux` kernel** on its
  `mux_core_range_set`. The combine/dispatch **workers NoC-write payloads to the mux** (they don't use the fabric API
  directly); the mux forwards to the persistent EDM.
- Mux lifecycle (`tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp`): `fabric_connection.open()` (l220) → forward loop
  (l227) → on **graceful** termination waits until `all_channels_drained` (l230-241) [**immediate** termination skips
  the drain] → `fabric_connection.close()` → status TERMINATED.
- **Explains the barrier-test failure:** a CQ `Synchronize` waits for the mux kernel/program to *complete* but does NOT
  reset the **persistent EDM** — so leaked EDM credit/connection state from collective N survives into N+1 regardless of
  CQ syncs.
- **Candidate mechanism (NOT proven):** per-op mux teardown/drain vs the persistent EDM's credit state across the two
  collectives, plausibly aggravated by the combine's **posted-write tail** leaving channel credits unreturned (→ a
  graceful-drain stall) — and/or the a2a-dispatch mux cores (default `(1,0)-(1,7)`) overlapping moe_compute's `(1,1)-(3,3)`.
- **Honest boundary:** static analysis can't prove the exact deadlocking step; confirming it needs device instrumentation
  of EDM/mux state during the hang (ptrace blocked / no device debugger) or tt-metal fabric-team expertise.

### RECOMMENDATION
Escalate to the **tt-metal fabric team** with the **2-layer minimal repro** (`--num-layers 2`) and this hypothesis:
back-to-back in-graph fused-MoE collectives deadlock because per-op mux teardown does not fully reset the
session-persistent EDM connection/credit state (a command-queue sync does not fix it). This is the natural stopping point
for tt-xla-side static analysis.

### CROSS-CHECK vs tt-metal reference `tt_moe_decode.py` (2026-07-08) — supports the hypothesis
`models/common/modules/moe/tt_moe_decode.py` is a working 2D-mesh fused-MoE decode using the SAME `moe_compute` op.
- Its `forward()` is a **single decode step, num_layers=1** (docstring: *"layer_id assumed 0 since the rest of the
  test/module pipeline is num_layers=1"*; reference test `test_tt_moe_decode.py` is "num_layers=1 throughout"). It never
  chains two `moe_compute` collectives.
- Different combine pipeline: `all_to_all_dispatch_metadata → moe_compute → tilize → deepseek_moe_fast_reduce_nc_fused →
  reduce_scatter → output` — an extra CCL + local ops sit between the combine and the layer output.
- Multi-layer usage = the deepseek_v3 model, run **eagerly** (per-op mesh programs, host-driven per decoder layer), NOT a
  single compiled graph packing all layers' collectives.
- ⟹ The reference **never creates tt-xla's triggering condition** (N back-to-back fused-MoE combines in ONE program,
  separated only by local non-CCL ops, no fabric drain between). It works because it structurally avoids exactly what
  tt-xla does. The op + 2D mesh are fine; the **chaining** is the problem. tt-metal has only ever validated
  num_layers=1-per-forward → tt-xla's single-graph back-to-back collectives are untested on the tt-metal side.
- Caveat: the reference's `reduce_scatter` is on the *replicated* axis (not the combine's *cluster* axis), so don't
  claim it "drains" the combine ring; the robust point is the structural one (single-layer / eager / never-in-one-graph).

Evidence (host + device agree; ptrace blocked, so via /proc + IR mtimes + hwmon):
- Compile finished **04:23** (IR artifacts in `modules/irs/…g1…`, ~45s); MoE bfp4 const-eval finished **04:57**
  (48 `[MOE_PREP_TIMING]` lines). Then the main decode graph **spun ~7h with zero progress** before SIGTERM.
- **Host:** one thread pegged at 100% with `voluntary_ctxt_switches` **frozen at 1314 for 5+ h** (busy-poll on a
  device completion — a mmap'd read, no syscalls); other 421 threads in `futex_wait`; RSS dead-flat 45 G.
- **Device:** all 32 WH chips uniformly **~34 W, dead-steady** (`/sys/class/hwmon/hwmon*/power1_input`) =
  clocked-but-not-computing → parked on a fabric barrier. (Active decode would draw higher and fluctuate.)
- **SIGTERM at 11:48** → clean exit 143 (no SIGKILL). Post-kill device power sagged **34.5→23.2 W** avg, no mesh
  holder → mesh **appears released (likely not stranded)**; run `jax.devices('tt')` to confirm before the next job.
  ⚠️ **`tt-smi` is NOT installed on this machine** — a reset would need install or a power-cycle.

**Next:** depth-bisect (2/4/8/16 layers) to pin the pass→hang threshold and the mechanism (see "### NEXT" above);
then inspect inter-collective synchronization. The mesh/fabric config (#1 torus descriptor), the fused-vs-unfused op
(#2), `layer_id`, and semaphore-count were all investigated and are NOT the cause (ruled out by depth-invariance /
the L1-buffer correction above). See auto-memory `moe-fused-wh-galaxy-progress`.

---

## 0. Bottom line / status

**The GPT-OSS fused MoE decode WORKS correctly on the WH galaxy — proven.** What remains is purely a
**host-side weight-prep performance** problem (see `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md`).

- ✅ Fabric + ops proven natively (tt-metal's own tests, WH 4x8 FABRIC_1D_RING).
- ✅ Full tt-xla path proven: **20B, 1 layer, device-vs-host PCC — prefill 0.999905, decode 0.999237, PASSED.**
- ⏳ **Full 24-layer 20B** run was in progress when this was written (see §6 — check if it finished/PASSed).
- ⛔ **120B end-to-end** and **full-model** runs are gated by ~1–2 hr of uncached host weight-quantize
  (NOT fabric, NOT compile). Fixes are prototyped/proposed in `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md`.

The BH-galaxy hang that motivated the whole port is **Blackhole-specific**; WH's 4-channel fabric
delivers the a2a payloads fine.

---

## 1. Key discovery vs the original handoff

This WH machine was a **clean checkout** of the pinned submodules — NONE of the Blackhole bring-up
edits were present (they lived only in the BH machine's working tree). So step 1 was to **reconstruct
the entire [KEEP] delta from the companion docs** and re-apply it. That reconstruction is done, built,
IR-verified, and now proven on device.

---

## 2. What's applied (and how to re-apply after a fresh clone / submodule reset)

### 2a. tt-mlir submodule edits — `/home/hshah/moe-fused-decode-tt-mlir.patch`
Apply into `third_party/tt-mlir/src/tt-mlir`:
```bash
cd /home/hshah/tt-xla/third_party/tt-mlir/src/tt-mlir
git apply /home/hshah/moe-fused-decode-tt-mlir.patch   # or --3way if the pin drifted
```
Full explanation + embedded diff: `MOE_FUSED_DECODE_THIRD_PARTY_CHANGES.md`. The 8 original files:
- `StableHLOToTTIRPatterns.cpp` — #9 expert-mapping global mesh coord.
- `TTNNOps.cpp` / `TTNNWorkaroundsPass.cpp` / `TTNNAllocateDistributedOpBuffers.cpp` /
  `TTNNDeduceMoEComputeLayouts.cpp` — #8 drain-core in-place (stash `ttnn.moe_metadata_drain_core`).
- `TTNNDecomposeLayouts.cpp` — #4 ui16 sharded-typecast gate.
- `TTNNResolveComposites.cpp` — #7 moe_compute `topology=Ring`.
- `runtime/.../ccl/all_to_all_dispatch_metadata.cpp` — #10 `SPARSE_UNICAST`.
- **NEWER (not in the patch file — regenerate it):** the two
  `runtime/.../ccl/prepare_moe_compute_w{0_w1,2}_weights.cpp` now have `[MOE_PREP_TIMING]`
  instrumentation + a `TTXLA_MOE_FAST_QUANTIZE`-gated on-device `ttnn::typecast` fast path. **Re-run
  `git -C <tt-mlir> diff > /home/hshah/moe-fused-decode-tt-mlir.patch` to capture these** (it will then
  be 10 files).
- **tt-metal submodule: NO edits.** (num_links is already 4 upstream; #6 is a no-op on FABRIC_1D_RING.)

### 2b. tt-xla-side WH config — already COMMITTED on the branch (a fresh clone has it)
`client_instance.cc`: 32-device galaxy → parent mesh `(4,8)` + `FABRIC_1D_RING`.
`tests/benchmark/test_llms.py`: `_galaxy_mesh_config_fn` → `(4,8)`, `cluster_axis=0` (EP-4/TP-8),
experts sharded on `"batch"`. This is tt-metal's validated orientation (matches `test_moe_gpt_e2e.py`).

### 2c. New PCC test (device-vs-host correctness, NOT a perf test) — committed
- Driver: `tests/benchmark/benchmarks/llm_pcc.py::run_llm_pcc_e2e` (reuses the benchmark's setup/shard/
  decode helpers; NO warmup/timing/metrics; prints host/device text + PCC via `print()`).
- Entries in `tests/benchmark/test_llms.py`: `test_gpt_oss_120b_moe_fused_galaxy_pcc`,
  `test_gpt_oss_20b_moe_fused_galaxy_pcc`, helper `run_llm_pcc`.
- Note: loguru is at WARNING in these runs, so the driver uses `print()` (not `logger.info`) for outputs.

---

## 3. What's proven (evidence)

- **Native tt-metal tests on WH** (via a built `python_env`, see §5): `test_moe_gpt_e2e.py::test_dispatch`
  PASS (a2a, all 32 devices, 7.8s); `::test_dispatch_compute_combine` PASS (a2a→compute→combine, PCC 0.987);
  `test_moe_compute_6U.py::test_moe_compute -k 1x8-torus` PASS incl. the `gpt_oss-bias-4experts-swiglu` config.
- **tt-xla 20B 1-layer PCC**: prefill 0.999905, decode 0.999237, device text == host text. PASSED (8:15).

---

## 4. The remaining problem (perf) — see `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md`
Runtime is dominated by **host-side const-eval weight quantization** (`quantize_weights_via_host`,
bf16→bfp4 on host, single-threaded, per-layer, uncached): ~34 min for 24-layer 20B MoE weights + ~50 min
(inferred) for embed/lm_head. tt-mlir compile is fast (~45s). Fixes (prototyped/proposed there):
on-device `ttnn::typecast` (env `TTXLA_MOE_FAST_QUANTIZE=1`, built/untested), packed-weight disk cache,
persistent compile cache, parallel host quantizer.

---

## 5. Environment & how to run

```bash
cd /home/hshah/tt-xla
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
source venv/activate            # recreates the venv if missing (python3.12 -m venv + pip)
cmake -G Ninja -B build && cmake --build build   # incremental after edits (OPMODEL is ON)
# device-vs-host PCC (1 layer is fast ~8min; omit --num-layers for full 24-layer ~1-2hr):
pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc
# fast-quantize A/B (untested): prefix the pytest with  TTXLA_MOE_FAST_QUANTIZE=1
```
- Native tt-metal `python_env` (for the §3 native tests) was built at
  `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/python_env` via `create_venv.sh`.
  Run: `env -u PYTHONPATH TT_METAL_HOME=$TTM ARCH_NAME=wormhole_b0 PYTHONPATH=$TTM python_env/bin/python -m pytest ...`
- `~/.cache` and `~/.ccache` are now **local** (were NFS symlinks; the user relocated them). GPT-OSS
  weights are cached in `~/.cache/huggingface` (20b ~13G, 120b ~61G).

### Gotchas (learned the hard way)
- **ptrace is blocked at the container level** here — gdb/py-spy/strace can't attach (even as root). No
  live profiling; use `fprintf`/log instrumentation + IR-dump mtimes instead.
- **Board reset:** killing a mesh-holding process **mid-collective** strands the board (needs `tt-smi -r`
  / power-cycle, usually the user does it). Killing during an **idle host-compute phase** (compile /
  const-eval — device idle) is SAFE (verified: no strand). Always confirm with `jax.devices('tt')` count
  before asking for a reset. See memory `dont-hardkill-galaxy-mesh-jobs`.
- **A frozen log ≠ a hang:** const-eval weight-prep and (rarely) compile go silent for many minutes.
  Confirm liveness via the busy thread's CPU advancing (`/proc/<pid>/stat` ticks), not the log.
- Rebuilding relinks `libTTMLIRRuntime.so`/`pjrt_plugin_tt.so`; don't rebuild while a run has them mmap'd
  (SIGBUS risk) — kill the run first.

---

## 6. Where we left off / next actions
1. **Check the full 24-layer 20B run** (`test_gpt_oss_20b_moe_fused_galaxy_pcc`, no `--num-layers`) —
   it was executing when this was written (log `.../scratchpad/pcc20b_full.log`, tail for
   `Decode PCC` / `PCC20B_FULL_EXIT` / the host/device text). Expected to pass (correctness proven); the
   only question was runtime (~1–2 hr, weight-prep bound).
2. **A/B the fast path**: `TTXLA_MOE_FAST_QUANTIZE=1` on 20B 1-layer — measure speedup AND PCC delta to
   decide if on-device `ttnn::typecast` quantization is safe to adopt. (Fast path is built, untested.)
3. **Implement a weight-prep perf fix** (packed-weight cache or compile cache) to make full/120B runs
   practical — see `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md`.
4. **120B**: only attempt end-to-end after a perf fix (else ~2+ hr/run). Ops already proven.
5. Before merging: regenerate `/home/hshah/moe-fused-decode-tt-mlir.patch` to include the instrumentation
   + fast-path; decide whether the instrumentation `fprintf`s and the `TTXLA_MOE_FAST_QUANTIZE` gate stay
   or get cleaned up; update `.github/workflows/perf-bench-matrix.json` (`runs-on` WH, unskip) once green.

## UPDATE 2026-07-08 (late): the hang op is NAILED — it is NOT the fused-MoE collectives

**Definitive finding (trace-proven).** Instrumented the tt-mlir runtime op-dispatch loop
(`runtime/lib/ttnn/program_executor.cpp`, env `TTXLA_OP_TRACE=1`) with numbered per-op `ENTER`/`EXIT`, plus dedicated
`MOE_TRACE` `ENTER`/`EXIT` in the `all_to_all_dispatch_metadata` and `moe_compute` handlers. Ran the 2-layer decode:

- **All four MoE collectives complete** — `all_to_all_dispatch #1/#2` and `moe_compute #1/#2` each show a matching
  `ENTER`→`EXIT`. The back-to-back-fused-collective-deadlock hypothesis is **REFUTED**.
- The process stalls at the **very next fabric op**: `[OP_TRACE] #1431 ENTER AllGatherOp` with **no `EXIT`**, right
  after `RMSNorm(#1424) → Matmul(#1427, LM head) → Reshape(#1430)`.
- That maps 1:1 onto the decode IR tail: **`ttnn.all_gather` at line 961 — `all_gather_dim=0, cluster_axis=0`, the
  LM-head logits gather `8×1×201088 → 32×1×201088`**. This is the *first `cluster_axis=0` collective after the second
  `moe_compute`*.

**Why 1 layer passes, 2 hang.** The 1-layer decode graph runs the *identical* LM-head `all_gather` and it **completes**
(PCC 0.999). The only difference at 2 layers is one extra `{dispatch, combine}` block on `cluster_axis=0` beforehand.
⟹ Root cause is **`cluster_axis=0` fabric/EDM state left behind by the fused-MoE combines** poisoning the next
`cluster_axis=0` collective — NOT the all_gather, the config, or the fused ops (which complete). The two per-layer
attention `all_gather`s use `cluster_axis=1` and are unaffected — only the MoE axis (`cluster_axis=0`) is poisoned.

**Corroborating:** a per-op mesh `Synchronize` (CQ drain) did not help (bad state is in the persistent EDM, not the CQ);
and `all_to_all_dispatch`'s non-persistent `init_semaphore` ring-rendezvous is *already* known to deadlock on
`cluster_axis=0` — likely the same fabric fragility.

**Escalation** (`MOE_FUSED_DECODE_WH_HANG_TTMETAL_ESCALATION.md`) has been rewritten around this: title, §1, §2, §3
(with the trace), §4 (revised hypothesis + concrete workarounds to try), §5 (fused ops ruled out).

**Workarounds to try next:** (a) host round-trip (`from_device`/`to_device`) or true fabric teardown+re-init between the
last `moe_compute` and the LM-head `all_gather`; (b) route the LM-head `all_gather` on `cluster_axis=1`; (c) give
`moe_compute`'s combine a destination-ACK barrier instead of the posted-write flush.

**Instrumentation left in the tree (dormant / env-gated, revert before merge):**
- `program_executor.cpp`: `<cstdio>` + RAII `OpTrace` (gated on env `TTXLA_OP_TRACE`).
- `moe_compute.cpp` / `all_to_all_dispatch_metadata.cpp`: `MOE_TRACE` fprintf (always-on) + `moe_compute` tolerates an
  unbound `optional_output_tensor`.
- `TTNNOps.cpp`: `MoeComputeOp::allocateBuffers` env-gated on `TTXLA_NO_MOE_OUTPUT_BUFFER`.

## UPDATE 2026-07-08 (later) — the mux/credit root cause above is REFUTED; root cause NOT yet found

The "cluster_axis=0 EDM state left by the fused combines poisons the next collective" conclusion (and the mux
departure-gated-teardown mechanism) is **refuted**. Evidence and current state:

- **Pure-ttnn negative control** `tmp/moe_credit_leak_repro.py` (standalone; drives the REAL `moe_compute`+`dispatch`+
  `all_gather` via `TTMoEDecode`): does **NOT hang** at 1/2/16 blocks, as a single trace program, with fresh per-layer
  semaphores, at the exact `(4,8)` dispatch-4 topology, or with a 201088-wide victim all_gather — nor any combination.
  So the isolated fused collectives are **not** the cause.
- **Cross-axis refuted:** `TTXLA_NO_ATTN_TP=1` (in `test_llms.py`) replicates attention (weights + KV-cache heads),
  dry-run-verified to remove the two `cluster_axis=1` attention all-gathers; the 2-layer device run **still hangs** at
  the same `cluster_axis=0` all_gather.
- The hang reproduces **only** in the full compiled decode graph → likely systemic (L1/DRAM/core/alloc pressure, the
  data-dependent lm-head-matmul→gather chain, or tt-mlir buffer/core allocation). Needs device visibility we don't have
  (Watcher broken).
- **New cheap tool:** `TTXLA_DRY_RUN=1` compiles + exports the graph to `modules/irs/` with no device run (~1 min, no
  hang risk) — use it to verify graph ablations before device runs.
- **The escalation (`MOE_FUSED_DECODE_WH_HANG_TTMETAL_ESCALATION.md`) was fully rewritten** around this honest picture;
  hand it + the negative-control harness to the tt-metal fabric team (Watcher the real stall).
- The earlier "workarounds to try" (route all_gather on axis-1 / ACK barrier / host round-trip) are **moot** — they
  targeted the refuted mux hypothesis.

## 7. Companion docs
- `MOE_FUSED_DECODE_WORMHOLE_HANDOFF.md` — the original brief (BH→WH port plan).
- `MOE_FUSED_DECODE_THIRD_PARTY_CHANGES.md` — the tt-mlir patch, per-file, with embedded diff.
- `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md` — the weight-prep bottleneck analysis + fixes.
- `MOE_FUSED_DECODE_GALAXY_BRINGUP.md`, `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`,
  `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md`, `MOE_FUSED_DECODE_BH_GALAXY_TTMETAL_ESCALATION.md` — deep dives.
- Auto-memory: `moe-fused-wh-galaxy-progress`, `dont-hardkill-galaxy-mesh-jobs`.
