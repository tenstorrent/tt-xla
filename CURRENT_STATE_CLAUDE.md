# Fused-MoE Decode on Wormhole Galaxy — Detailed State & Handoff (for Claude)

*This file is a single-file replacement for the older handoff/report docs (listed in §13). Read this first.*
*Branch: tt-xla `hshah/moe-compute-path`; tt-mlir submodule `hshah/dmilinkovic/moe-decode-composite`. Updated 2026-07-08.*
*Human summary companion: `CURRENT_STATE_HUMAN_READABLE.md`.*

---

## 0. TL;DR

- We lower GPT-OSS **decode** MoE to a fused TTNN pipeline (`all_to_all_dispatch_metadata` → `moe_compute`) on the
  32-chip Wormhole galaxy (`(4,8)` mesh, EP-4/TP-8, `FABRIC_1D_RING`).
- **1-layer gpt-oss-20b decode runs end-to-end and is bit-correct** (prefill PCC 0.999905, decode 0.999237). **2+
  layers HANG** on device, ~10 min in.
- The hang is at the **LM-head logit `ttnn.all_gather` on `cluster_axis=0`** (`8×1×201088 → 32×1×201088`), which is the
  first fabric op after both layers' MoE collectives complete. Depth-dependent (1 layer runs the same op fine).
- **Ruled out** (with a pure-`ttnn` negative control + galaxy ablations): the fused-MoE collectives themselves, a
  cross-axis (attn-axis1 + MoE-axis0) interaction, a per-op fabric-mux/EDM credit leak, and the test's lm-head hook.
- **Remaining hypothesis:** full-compiled-graph systemic state (L1/DRAM/core/alloc pressure, the data-dependent lm-head
  matmul→gather chain, or tt-mlir buffer/core allocation). **Blocked on device visibility — Watcher is broken here.**
- **Deliverables:** rewritten fabric-team escalation (`MOE_FUSED_DECODE_WH_HANG_TTMETAL_ESCALATION.md`), a runnable
  negative-control harness (`tmp/moe_credit_leak_repro.py`), and this doc. The fused-decode port changes are on the
  tt-mlir branch `hshah/dmilinkovic/moe-decode-composite` (§11).

---

## 1. Repo / environment layout

- **tt-xla working tree:** `/home/hshah/tt-xla` (branch `hshah/moe-compute-path`). Local NVMe (moved off NFS earlier for
  I/O perf). `~/.cache` + `~/.ccache` are local.
- **tt-mlir submodule (source of truth for the build):**
  `/home/hshah/tt-xla/third_party/tt-mlir/src/tt-mlir` (branch `hshah/dmilinkovic/moe-decode-composite`). The
  fused-decode port is on this branch — 10 files, §10.
- **tt-mlir build dir:** `<tt-mlir>/build`. **Install prefix (what the plugin loads):**
  `/home/hshah/tt-xla/third_party/tt-mlir/install/lib` (`libTTMLIRRuntime.so`, `libTTMLIRCompiler.so`, `libtt_metal.so`).
- **tt-metal (nested submodule):** `<tt-mlir>/third_party/tt-metal/src/tt-metal` (pinned commit `3a5f8033`, **unmodified**
  — we only read it). `TT_METAL_HOME` points here; device kernels JIT-compile from it (cache `~/.cache/tt-metal-cache`).
- **Env:** `ARCH_NAME=wormhole_b0`, `TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain`, venv `/home/hshah/tt-xla/venv`
  (activated; `import ttnn` works — it's a *complete* redirector shim at `python_package/ttnn` → the real ttnn +
  `_ttnn.so`). `TTMLIR_ENV_ACTIVATED=1`.
- **Hardware:** Wormhole TG/6U galaxy, 32 chips. `tt-smi` is NOT installed; container `CapEff=0` (no driver reset,
  no ptrace). Physical topology best viewed as `(4,8)` (the size-4 axis is a 1D ring — the EP dispatch axis).

---

## 2. What the fused decode is + the lowering pipeline

**Model side (tt-xla).** The gpt-oss model runs via PyTorch/XLA with the **`tt_moe_fused`** experts backend
(`register_tt_moe_backend(cluster_axis=0, use_interleaved=True, moe_decode_activation="swiglu", ...)` in
`tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc`). Decode (seqlen=1) emits a StableHLO composite
**`tt.moe_decode`** per layer; **prefill** routes to a dense bmm (`tt_dense_experts_forward`), *not* the fused op.

**Composite operands/result** (`tt.moe_decode`, name const `kTTMoeDecodeCompositeName` in
`include/ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h:62`):
```
operands (6 no-bias / 9 with-bias): tokens[1,1,M,H], expert_indices[1,1,M,K], expert_scores[1,1,M,K],
                                    w0[L,E,H,N], w1[L,E,H,N], w2[L,E,N,H], (bias0[L,E,N], bias1[L,E,N], bias2[L,E,H])
result: combine_output[K, M, H]
expert_mapping is NOT an operand — it's synthesized from topology inside tt-mlir.
```

**tt-mlir lowering (the ~10 changed files, §10):**
1. **StableHLO→TTIR** (`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`): converts the `tt.moe_decode`
   composite; `synthesizeMoeDecodeExpertMapping(meshShape, clusterAxis, expertsPerDevice)` builds the 32-row
   `[num_devices, num_experts]` global-coord expert mapping (value `a*numCols+col(d)`).
2. **Custom sharding rule** (`lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp`,
   `getMoeDecodeShardingRule`): the composite is opaque to Shardy; the rule shards **token (M)** and **expert (E)** dims
   along the cluster axis (`kPassThrough`) and **replicates** H/N/K/L (`kNeedReplication`). ⇒ result `[K,M,H]` has M
   sharded on cluster_axis=0, K & H replicated.
3. **TTIR→TTNN** (`lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`): emits `ttnn.all_to_all_dispatch_metadata` +
   `ttnn.moe_compute`.
4. **TTNN passes:**
   - `TTNNResolveComposites.cpp` — resolve the composite pieces.
   - `TTNNDeduceMoEComputeLayouts.cpp` — deduce moe_compute I/O layouts (drain-core L1 height-sharded metadata, etc.).
   - `TTNNAllocateDistributedOpBuffers.cpp` — allocate the **persistent** output buffers (dispatch out triple +
     combine output) and stash the drain core (`ttnn.moe_metadata_drain_core` = core (7,8)).
   - `TTNNAllocateDistributedOpSemaphores.cpp` — mint a **fresh `GlobalSemaphore` per collective per layer**
     (prelude-hoisted; e.g. 2 layers → 4 distinct semaphores).
   - `TTNNWorkaroundsPass.cpp` / `TTNNDecomposeLayouts.cpp` — pin indices/scores L1 HeightSharded on the drain core;
     a ui16 sharded-typecast gate.
   - `TTNNConfigureCCLOps.cpp` — set `topology=Ring` + `cluster_axis` on AllGather/MoeCompute/etc. from the mesh.
5. **Runtime handlers** (`runtime/lib/ttnn/operations/ccl/`):
   - `all_to_all_dispatch_metadata.cpp` — calls `ttnn::experimental::all_to_all_dispatch_metadata(...)` in **persistent
     mode** (binds the 3 output buffers + cross-device semaphore → sets `SKIP_INIT_SEMAPHORE`); **hardcodes
     `DispatchAlgorithm::SPARSE_UNICAST`** (the default `SPARSE_MCAST_SHORTEST_PATH` has a documented cluster_axis=0
     ring-hop bug); `WorkerMode::DIRECT`.
   - `moe_compute.cpp` — calls `ttnn::experimental::moe_compute(..., compute_only=false)` (full path, internal combine);
     requires bound `optional_output_tensor` + `cross_device_semaphore` (else tt-metal's `use_init_semaphore` path
     deadlocks); returns 6 tensors, exposes `results[5]` (combine output), frees `results[0..4]`.
   - `prepare_moe_compute_w0_w1_weights.cpp` / `prepare_moe_compute_w2_weights.cpp` — host weight prep +
     `quantize_weights_via_host` to `bfloat4_b`. **Carry `[MOE_PREP_TIMING]` timing prints and an opt-in
     `TTXLA_MOE_FAST_QUANTIZE` on-device-typecast fast-path** (default off; see §12).

**Config (the `(4,8)` galaxy test):** mesh axes `("batch", "model")` → **batch = axis0 (size 4, EP / cluster_axis=0
dispatch ring)**, **model = axis1 (size 8, attention TP)**. `input_output_sharding_spec=("batch", None)`,
`kv_cache_sharding_spec=("batch","model",None,None)`, weights `bfp_bf8` (attention) / `bfloat4_b` (experts).
gpt-oss-20b: hidden=intermediate=2880, select_k=4, has_bias=true, swiglu. (The gpt_oss.yaml in tt-metal is the 120B:
128 experts, `[8,4]` mesh.)

**Decode graph structure (per layer):** `attention` (TP-8 on axis1 → one `cluster_axis=1` all_gather per layer,
`dot.N_allGather`, plus sdpa_decode/rope/paged_update_cache) → `MoE` (`all_to_all_dispatch_metadata` +
`moe_compute` on cluster_axis=0). **Tail (once, after last layer):** `rms_norm → matmul (2880→201088, lm_head) →
reshape → all_gather(cluster_axis=0, gathers batch 8→32) → argmax → all_gather(cluster_axis=0, token)`. See the saved
IR `moe_2layer_decode_ttnn_graph.mlir` (hang op = `%159`, the `ttnn.all_gather` at line 961).

---

## 3. Reproducers & expected results

**Run from `/home/hshah/tt-xla` with the venv active.** Each run holds all 32 chips ~7–10 min.

- **1-layer (PASSES):**
  ```
  pytest -svv --num-layers 1 tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc
  ```
  Expect: `Prefill PCC = 0.999905`, `Decode PCC = 0.999237`, next-token HOST==DEVICE, `1 passed in ~7 min`.
- **2-layer (HANGS):**
  ```
  pytest -svv --num-layers 2 tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc
  ```
  Expect: hangs ~10 min in, after the model prints weight-prep + FABRIC_1D_RING messages. SIGTERM to release.
- **Compile-only DRY RUN (no device, ~1 min, no hang/strand risk — use for graph inspection):**
  ```
  TTXLA_DRY_RUN=1 pytest -svv --num-layers 2 tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc
  ```
  Fails only at the PCC assert (expected — no real device output). Exports IR to `modules/irs/`:
  `ttnn_runtime_..._g1_*.mlir` = the **decode** graph (g0 = prefill). Note: the pytest process lingers several minutes
  in nanobind atexit "leaked instance" cleanup after finishing — the IR is written mid-compile, so poll for a new
  `*g1*.mlir` file rather than waiting for process exit.

**Fast pure-ttnn negative-control harness** `tmp/moe_credit_leak_repro.py` (standalone, ~2 min, does NOT reproduce the
hang — see §7):
```
NUM_BLOCKS=2 MESH=4,8 TRACE=1 FRESH_SEMS=1 WIDE=201088 python tmp/moe_credit_leak_repro.py
```
Env knobs: `NUM_BLOCKS` (fused blocks), `MESH` (`4,8` or `8,4`), `TRACE=1` (capture all ops into one trace program),
`FRESH_SEMS=1` (fresh per-block semaphores), `WIDE=N` (gather an N-wide victim tensor instead of the 2880-wide combine
out). It drives the real `moe_compute`+`dispatch`+`all_gather` via `TTMoEDecode` (tt-metal's `models.common` module).

---

## 4. The hang — proven facts

- **Depth-dependent:** 1 layer PASSES; 2 (and 24) HANG. Everything else identical.
- **Localized (trace-proven):** per-op ENTER/EXIT instrumentation showed **all four MoE collectives complete** (both
  layers' `all_to_all_dispatch` + `moe_compute` each ENTER→EXIT), then the process hangs at the **next fabric op**:
  `[OP_TRACE] #1431 ENTER AllGatherOp` with **no EXIT**, right after `RMSNorm → Matmul(lm_head, →201088) → Reshape`.
  Maps 1:1 to `moe_2layer_decode_ttnn_graph.mlir:961` — `ttnn.all_gather`, `all_gather_dim=0, cluster_axis=0`,
  `8×1×201088 → 32×1×201088` (gathering the data-parallel batch to full).
- **Physical signature at hang** (how to detect it without ptrace/Watcher):
  - Host: one thread pegged `R`, `voluntary_ctxt_switches` **frozen** (busy-poll on a device flag; check via
    `grep voluntary_ctxt /proc/<pid>/status` sampled over time). `wchan=0` (userspace spin).
  - Device: all 32 chips uniformly **~34–37 W, dead-steady** (`/sys/class/hwmon/hwmon*/power1_input`, `/1e6` for W).
    Idle/released ≈ 22–24 W; deep idle ≈ 13 W. A hang holds ~34 W flat; active compute would fluctuate higher.
  - **Clean `SIGTERM`** reliably unwinds it and releases the mesh (power → ~23 W, `jax.devices('tt')` → 32). Do this,
    not SIGKILL.
- **1 layer runs the identical `all_gather` and it completes** (PCC 0.999) — so the all_gather op is fine; something
  about the 2-layer context breaks it.

---

## 5. What's RULED OUT (with evidence) — the valuable part

1. **The fused-MoE collectives are NOT the cause.** The pure-ttnn negative control (`tmp/moe_credit_leak_repro.py`) runs
   the *same real* `moe_compute`+`dispatch`+`cluster_axis=0 all_gather` and **never hangs** across: eager 1/2/16 blocks;
   single **trace** program (all ops back-to-back, no op boundary — mimics the compiled graph); **fresh per-layer
   semaphores**; exact **`(4,8)` dispatch-4** topology; **201088-wide** victim all_gather; and all of these combined.
2. **NOT a cross-axis interaction.** Ablation `TTXLA_NO_ATTN_TP=1` replicated attention (weights **and** KV-cache
   heads), dry-run-verified to remove both `cluster_axis=1` attention all-gathers; the 2-layer device run **still
   hangs** at the same `cluster_axis=0` all_gather. (This env gate is now reverted; re-add per §12 if needed.)
3. **NOT the per-op fabric-mux / EDM credit leak.** Earlier hypothesis: `tt_fabric_mux.cpp` graceful termination is
   gated on *departure* (`all_channels_drained` at `:230-242`; `forward_data` frees slots on
   `send_payload_flush_non_blocking` at `:117-124`, comment "not handling acks"), closing the EDM connection before
   forwarded packets are completion-credited into the never-reset single-digit sender-channel pool
   (`fabric_erisc_router.cpp:3037-38`). **Refuted** by #1 (same mux + collectives don't hang in isolation).
4. **NOT the test's lm-head sharding hook.** Gating `sharding_constraint_hook(model.lm_head, ..., (None,None,None))`
   (`llm_pcc.py`) was a **graph no-op** — the huge all_gather comes from output-sharding propagation, not that hook.
5. (Earlier, weaker) ruled out: mesh/fabric config (identical at 1 vs 2 layers), `layer_id` (compile-time weight
   offset), per-op semaphore reuse, the shared combine-output buffer `%31` (unbinding it via
   `TTXLA_NO_MOE_OUTPUT_BUFFER` still hung), op-usage (verified vs nanobind docstrings + 1-layer PCC).

**Key mental model:** eager ttnn runs each op as a *separate program* (recovery at boundaries); a single trace/compiled
graph runs them back-to-back. Neither the trace harness nor the compiled tt-xla graph of *just the fused ops* hangs —
so the trigger needs the *rest* of the full decode graph too.

---

## 6. What REMAINS — hypotheses & concrete next experiments

The hang reproduces only in the full compiled decode graph. Candidates (rough priority):

1. **Systemic on-device resource pressure.** The full program has 2× attention (sdpa, rope, `paged_update_cache`,
   axis-1 all_gathers) + 2× MoE (persistent dispatch/combine buffers, per-op mux cores (1,1)-(3,3)) + const-eval'd bf4
   weights + paged KV cache — all competing for L1 / worker cores / DRAM / fabric connections. The big `cluster_axis=0`
   all_gather may fail to acquire a core/L1/connection and block. The fast harness has abundant free resources; the full
   graph does not.
2. **Data-dependent LM-head chain.** The all_gather's input is the `2880→201088` lm-head matmul output; investigate the
   matmul's allocation/completion interacting with the following gather.
3. **tt-mlir buffer/core allocation** in the full program vs `TTMoEDecode` defaults (mux core ranges, num_links, memory
   configs, const-eval buffer placement).

**Concrete next experiments (in order of value):**
- **[needs Watcher-working env] Instrument the real hang.** Run the 2-layer tt-xla test with Watcher/tt-lens in an
  environment where Watcher works (ours is broken, §9) and see **which core/waypoint/semaphore the `all_gather`
  workers/EDM connection are stuck on**. This is the single datum that would resolve it. (This is the escalation ask.)
- **[fast harness] Add the missing full-graph pieces to `tmp/moe_credit_leak_repro.py`** incrementally until it
  reproduces: (a) a real lm-head matmul `2880→201088` feeding the gather (data-dependent chain); (b) interleaved
  attention-style `cluster_axis=1` all_gathers + `paged_update_cache`; (c) crank up L1/DRAM pressure. If any makes it
  hang, you have a fast repro + the missing ingredient. (Cross-axis alone was already ruled out on the real graph, but
  the *combination* with memory pressure hasn't been isolated in the harness.)
- **[tt-xla, ~10 min/run] Ablate the huge all_gather properly.** It's forced by output sharding, not the (void) lm-head
  hook. Change `input_output_sharding_spec` / the output handling so the logits stay batch-sharded (host gathers on
  readback) → does the device decode then complete? If yes, the huge in-graph gather is the terminal victim (fix →
  reconfigure/avoid it). Verify the graph change with a DRY RUN first.
- Reduce weight-prep pressure with `TTXLA_MOE_FAST_QUANTIZE=1` (on-device typecast) to change the allocation timeline and
  see if the hang moves — a cheap perturbation.

---

## 7. Build & iterate

**Build tt-xla from scratch:**
```
source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=DEBUG   # the build-type ensures tt-mlir gets the right flags
cmake --build build
```
(Needs `TTMLIR_TOOLCHAIN_DIR`; the venv sets `TTXLA_ENV_ACTIVATED`. See CLAUDE.md.) For a **clean rebuild**, `rm -rf
build third_party/tt-mlir third_party/loguru` first — the configure re-fetches tt-mlir **by branch name**
(`TT_MLIR_VERSION` in `third_party/CMakeLists.txt`), so it picks up the branch tip. (Verified: clean DEBUG build +
1-layer PCC pass. The tt-metal device kernels build Release regardless, so runtime is not much slower than a Release
plugin build.)

**Iterate a tt-mlir runtime/compiler change (fast loop):**
```
cd /home/hshah/tt-xla/third_party/tt-mlir/src/tt-mlir && source env/activate
cmake --build build --target TTMLIRRuntime TTMLIRCompiler        # only changed TUs recompile
cp build/runtime/lib/libTTMLIRRuntime.so   ../../install/lib/     # runtime handlers, program_executor
cp build/lib/libTTMLIRCompiler.so          ../../install/lib/     # dialect/passes (TTNNOps.cpp etc.)
```
(Install prefix = `/home/hshah/tt-xla/third_party/tt-mlir/install/lib`.) **Device kernels** (e.g. `tt_fabric_mux.cpp`)
are JIT-compiled from `TT_METAL_HOME` at runtime — edit + clear `~/.cache/tt-metal-cache`, **no rebuild needed** for
device-kernel changes.

## 8. Run & monitor the galaxy (no ptrace/Watcher)

**Monitor a galaxy run:**
- Power: `for f in /sys/class/hwmon/hwmon*/power1_input; do awk '{printf "%.0f\n",$1/1e6}' "$f"; done | sort -n | uniq -c`
  (~34 W flat across 32 = held/parked; ~23 = released; ~13 = deep idle).
- Hang fingerprint: `grep voluntary_ctxt /proc/<pid>/status` **frozen** across samples while a thread is `R`. Gate any
  hang-detector on "decode has started" (else the slow host weight-quant, >150 s for the big lm-head, false-positives).
- **Release a hung run with `SIGTERM`** (clean, releases mesh). **Never SIGKILL / never let it SIGABRT during fabric
  bring-up** — that strands an ETH core and the board needs an operator reset (`tt-smi -r` / power cycle; `tt-smi` not
  installed here, so a human must do it). See memory `dont-hardkill-galaxy-mesh-jobs`.

**Board health check:** `python -c "import jax; print(len(jax.devices('tt')))"` should print 32.

---

## 9. Watcher is broken here (the core blocker)

`TT_METAL_WATCHER` cannot be used on this build:
1. Hard `TT_THROW` on **fast-dispatch core (0,0)** reading `watcher.enable == 0`
   (`tt_metal/impl/debug/watcher_device_reader.cpp:588`) — unconditional, not guarded by any
   `TT_METAL_WATCHER_DISABLE_*` flag. (`init_device` writes WatcherEnabled to all Tensix cores, but (0,0) is clobbered
   by the fast-dispatch firmware's different L1 layout.)
2. Patching that throw to warn-skip lets it continue, then it **segfaults in the ETH-link retraining dump**
   (`DumpEthLinkStatus`) during setup, before decode.
3. Each crash is an unclean abort during fabric bring-up → **strands an ETH core** → board reset required.

So we have **no device-level visibility**. ptrace is also blocked (`CapEff=0`, Seccomp mode 2, yama=1). This is why the
investigation is black-box and why the fix now needs the tt-metal fabric team (working Watcher). Watcher itself needs
fixing (fast-dispatch (0,0) handling + the ETH-dump segfault).

---

## 10. Key files

**tt-mlir changed files (the fused-decode port — the 10 files added on the branch for the fused decode, §11):**
| File | Role |
|---|---|
| `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` | `tt.moe_decode` composite → TTIR; `synthesizeMoeDecodeExpertMapping` |
| `lib/Dialect/TTNN/IR/TTNNOps.cpp` | `MoeComputeOp` / `AllToAllDispatchMetadataOp` `allocateBuffers`/`allocateSemaphores`/`hasUnboundBuffers` |
| `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp` | pin moe_compute indices/scores L1 HeightSharded on drain core |
| `lib/Dialect/TTNN/Transforms/OptimizerPasses/TTNNDeduceMoEComputeLayouts.cpp` | moe_compute layout deduction |
| `lib/Dialect/TTNN/Transforms/TTNNAllocateDistributedOpBuffers.cpp` | persistent buffers + `ttnn.moe_metadata_drain_core` (7,8) |
| `lib/Dialect/TTNN/Transforms/TTNNDecomposeLayouts.cpp` | ui16 sharded-typecast gate |
| `lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp` | resolve composite |
| `runtime/lib/ttnn/operations/ccl/all_to_all_dispatch_metadata.cpp` | dispatch handler; **SPARSE_UNICAST** |
| `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w0_w1_weights.cpp` | w0/w1 prep + quantize; `[MOE_PREP_TIMING]`, `TTXLA_MOE_FAST_QUANTIZE` |
| `runtime/lib/ttnn/operations/ccl/prepare_moe_compute_w2_weights.cpp` | w2 prep + quantize; same instrumentation |

Also relevant (part of the branch, outside the 10-file set above): `RegisterCustomShardingRule.cpp`
(`getMoeDecodeShardingRule`), `TTNNConfigureCCLOps.cpp`, `TTNNAllocateDistributedOpSemaphores.cpp`,
`runtime/.../moe_compute.cpp`.

**tt-xla files (show as `M` in `git status` — this is PRE-EXISTING branch work, the PCC test scaffolding, NOT
diagnostics):** `tests/benchmark/benchmarks/llm_pcc.py` (the `run_llm_pcc_e2e` device-vs-host PCC driver),
`tests/benchmark/test_llms.py` (`test_gpt_oss_20b_moe_fused_galaxy_pcc` + `_galaxy_mesh_config_fn` +
`_gpt_oss_120b_moe_fused_galaxy_shard_spec_fn`).

**Artifacts:** `tmp/moe_credit_leak_repro.py` (negative control), `moe_2layer_decode_ttnn_graph.mlir` (decode IR, hang op
at line 961), `modules/irs/` (auto-exported IRs, `*g0*`=prefill / `*g1*`=decode).

---

## 11. tt-mlir branch notes

The fused-decode port (10 files, §10) is on the tt-mlir branch `hshah/dmilinkovic/moe-decode-composite`. A fresh
`tt-xla` build fetches tt-mlir **by branch name** (`TT_MLIR_VERSION` in `third_party/CMakeLists.txt`,
`GIT_TAG ${TT_MLIR_VERSION}`), so building tt-xla off this branch reproduces the working fused-decode path — verified
by a from-scratch clean build + a passing 1-layer PCC run (prefill 0.999905 / decode 0.999237). **The tt-metal submodule
is unmodified.** Note: the 2 `prepare_*.cpp` files carry the weight-prep-perf work (`[MOE_PREP_TIMING]` prints + dormant
`TTXLA_MOE_FAST_QUANTIZE`); kept intentionally (default-off). Backup patches of the port (outside the repo):
`/home/hshah/moe_decode_ttmlir_changes.patch` (git-diff form) and `/home/hshah/moe_decode_ttmlir_commit.patch`
(commit form).

---

## 12. Debug instrumentation — currently REMOVED; how to re-add

All this-session hang-investigation instrumentation was **reverted** (tree + installed `.so` are clean). To re-enable
for further debugging, re-add:
- **Per-op ENTER/EXIT trace:** in `runtime/lib/ttnn/program_executor.cpp::runOperation`, an RAII guard printing
  `[OP_TRACE] #N ENTER/EXIT <OpType>` gated on env `TTXLA_OP_TRACE` (destructor covers the `switch`'s per-case
  `return`s). Add `#include <cstdio>`.
- **Collective trace:** `[MOE_TRACE] <op> #N ENTER/EXIT` `fprintf(stderr,...)` in the `moe_compute.cpp` /
  `all_to_all_dispatch_metadata.cpp` handlers.
- **Attention-TP ablation (reverted):** `TTXLA_NO_ATTN_TP` gate in `test_llms.py` — replicate attention q/k/v/o weights
  **and** set `kv_cache_sharding_spec` heads `"model"→None` (weights alone are a graph no-op).
- **lm-head-gather / buffer ablations (reverted, mostly no-ops):** `TTXLA_NO_LMHEAD_GATHER` (llm_pcc.py),
  `TTXLA_NO_MOE_OUTPUT_BUFFER` (TTNNOps.cpp `MoeComputeOp::allocateBuffers` early-return).
Rebuild + reinstall per §7. **Do not commit these** to the branch.

---

## 13. Superseded docs (this file replaces them)

Older docs in the repo root, kept for deep-dive reference but **superseded by this file** — do not treat their
conclusions as current (esp. the "mux/EDM credit-leak root cause", which is REFUTED):
`MOE_FUSED_DECODE_WH_PROGRESS_HANDOFF.md`, `MOE_FUSED_DECODE_HANDOFF.md`, `MOE_FUSED_DECODE_WORMHOLE_HANDOFF.md`,
`MOE_FUSED_DECODE_GALAXY_BRINGUP.md`, `MOE_FUSED_DECODE_CONFIG_COMPARISON.md`, `MOE_ALL_TO_ALL_DISPATCH_EXPLAINER.md`,
`MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`, `MARK_ARGUMENT_UINT32_TYPECAST_BUGREPORT.md`,
`MOE_FUSED_DECODE_BH_GALAXY_TTMETAL_ESCALATION.md` (Blackhole-specific), `MOE_FUSED_DECODE_WEIGHT_PREP_PERF.md`
(weight-prep perf analysis — still useful for the `TTXLA_MOE_FAST_QUANTIZE` path).
**Current/authoritative:** this file + `CURRENT_STATE_HUMAN_READABLE.md` + `MOE_FUSED_DECODE_WH_HANG_TTMETAL_ESCALATION.md`
(the fabric-team escalation).

---

## 14. Gotchas / lessons

- **1 galaxy run ≈ 7–10 min** (model load + ~1–3 min bf4 weight quant + prefill + decode). Weight quant is host-side and
  slow; a naive hang-detector will false-positive during it — gate on "decode started".
- **Nanobind atexit** prints a long "leaked instance" dump and the pytest process can linger many minutes after the test
  finishes; don't treat that as "still running". Exported IRs are written mid-compile.
- **SIGTERM = clean mesh release; SIGABRT/SIGKILL during fabric bring-up = stranded board.** Reset needs a human.
- The importable `ttnn` in the tt-xla venv is a **complete** redirector (has `ttnn.device`, `ttnn.experimental.*`,
  `MeshDevice`, fabric config). Standalone ttnn scripts work if you `import ttnn` first, then `sys.path.insert(0,
  <tt-metal-src>)` for `models`/`tests` (see the harness). Running `python -m pytest` from the tt-metal src dir shadows
  ttnn with the bare package — avoid.
- **Mesh orientation:** WH galaxy uses `(4,8)` cluster_axis=0 (size-4 dispatch ring). The earlier Blackhole galaxy used
  `(8,4)` — opposite physical orientation. `with_mesh_shape` refuses to change the cluster-axis dim size (can't
  transpose `(8,4)↔(4,8)`); substitute `mesh_shape` in the YAML text to get a native config.
