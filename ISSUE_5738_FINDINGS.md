# Issue #5738 — local repro & root-cause localization

Restore 2D `(2, N//2)` weight-sharded mesh for `test_llama_3_1_70b_tp_qb2` (PR #5717 worked around it with a TP-only `(1,N)` mesh; decode PCC ~0 on the 2D mesh).

## Setup
- Fresh build of tt-xla @ main `2ea8f4d4f` (tt-mlir `9f06802f`) in the qb2 docker image (Blackhole, 4 devices → default mesh `(2,2)`).
- **Minimal repro**: 1 layer, `(2,2)` mesh, `--pcc-decode` — no need for 80 layers / large mesh.
- Test made env-driven (`QB2_MESH`/`QB2_OPT`/`QB2_KV`/`QB2_NORM`) — see `tests/benchmark/test_llms.py::test_llama_3_1_70b_tp_qb2`.

## Results (1 layer, `--pcc-decode`)
| Config | Prefill PCC | First-decode PCC |
|---|---|---|
| TP-only `(1,4)`, opt1 (baseline) | 0.9997 | **0.9998** PASS |
| 2D `(2,2)`, opt1 | 0.9993 | **0.000000** (rel_l2 4.1e19) FAIL |
| 2D `(2,2)`, opt0 | 0.9989 | **0.0** (device output all-constant) FAIL |
| 2D + KV cache batch-sharded | 0.9995 | **0.0** FAIL |
| 2D + RMS-norm weights replicated | 0.9993 *(identical)* | **0.0** FAIL |

## Conclusions
1. **Reproduced** the first-decode PCC≈0 (prefill fine, decode inf/nan) on a 1-layer `(2,2)` mesh.
2. **Not opt-level** — opt0 and opt1 both fail (opt2 is the separate known hang).
3. **Not KV-cache sharding** — batch-sharding the cache does not help (confirms #5487).
4. **Not `distributed_rms_norm`** — replicating the norm weights (which removes the distributed-rms-norm path) leaves prefill *byte-identical* and decode still 0.0. At these opt levels the op decomposes to `all_gather` + plain `ttir.rms_norm` on full hidden anyway.

## Where the corruption is (from exported IR — `modules/irs`, g0=prefill, g1=decode)
Differential of the **working TP-only** vs **broken 2D** decode graph:

| | TP-only `(1,4)` (works) | 2D `(2,2)` (broken) |
|---|---|---|
| batch-axis (`cluster_axis=0`) collectives in decode | **0** | **12** (`all_to_all`, `reduce_scatter scatter_dim=0`) |
| batch dim | 32 (unsharded) | **16 (sharded on batch axis)** |
| KV cache | `[32,2,128,128]` | `[16,4,128,128]` |
| `update_cache` value→cache layout | `[1,kv,batch,hd]`→`[batch,kv,seq,hd]` | **same** |

The `update_cache` permute (`[2,1,0,3]`) is identical in both paths → it is the intended TTNN convention, **not** the bug.

## Isolation of the individual metal ops (do NOT trust the issue — it is AI-generated)
The issue asserts "SDPA-decode is correct, KV-cache read is corrupted." Tested directly instead of trusting it:

- **`scaled_dot_product_attention_decode` is CORRECT** at the exact 2D per-shard shapes (single device, `Q[1,16,32,128]`, `K/V[16,4,128,128]`, GQA 32/4, mask, cur_pos=100): **PCC 0.9997**, no nan/inf (`sdpa_decode_repro.py`). It is a *local* op (no cross-device comm), so the per-shard result equals the mesh result. **SDPA-decode is NOT the bug** — the issue's framing of it is misleading.
- **`reduce_scatter` (cluster_axis=0, batch axis) on a real `(2,2)` mesh is CORRECT** — gathered sums match exactly (`reduce_scatter_repro.py`).
- These ops (and `all_to_all`) are also used by the **working prefill** (0.999), further arguing against a raw kernel bug.
- The 2D decode attention subgraph is **structurally sound and analogous to the working TP graph** (rotary, mask `ge→where→repeat`, `paged_update_cache`, per-device GQA head/batch mapping all consistent).

## THE decisive test: eager vs compiled (same sharding)
Ran the **exact same** 1-layer 2D-`(2,2)` sharded decode two ways:

| Execution | First-decode logits | PCC vs CPU golden |
|---|---|---|
| **eager** (`logits_wrapper(**input_args)`, no `torch.compile`) | normal (lm_head max **9.6**, std 1.7) | **0.9999 (all 32 users)** |
| **compiled** (`torch.compile(backend="tt")`) | exploded (~**1e19**) / zeros | **0.0** |

Same model, weights, sharding annotations, KV cache, inputs. Eager runs the identical ttnn ops + batch-axis collectives + mesh sharding **and is correct**. Only the tt-mlir **compilation pipeline** corrupts it.

Per-op eager dump (2D mesh) — every stage normal & uniform across users:
`embed 0.045 → input_ln 1.42 → self_attn 0.10 → post_ln 0.27 → mlp 0.11 → final_norm 18.0 → lm_head 9.6`.

## CONCLUSION (proven)
The bug is a **tt-mlir compiler bug in the whole-graph compilation of the 2D-mesh (batch-axis-sharded) decode** — **not** a metal kernel, **not** the sharding semantics, **not** the model. Everything the AI-generated issue blamed is disproven:
- ✗ `distributed_rms_norm` (replicating norm weights changes nothing; norm runs on full hidden after `all_to_all`)
- ✗ `scaled_dot_product_attention_decode` kernel (PCC 0.9997 at 2D shapes)
- ✗ KV-cache read / prefill cache-write (isolated decode with CPU-golden cache still fails; eager with same cache is 0.9999)
- ✗ opt-level (opt0 and opt1 both fail), ✗ `reduce_scatter` kernel (correct on `(2,2)`)

**Trigger:** batch-axis sharding of the decode graph (`(2,2)` and `(4,1)` fail; `(1,4)` TP works). **Symptom:** magnitude explosion to ~1e19 (+ zeroed sub-blocks), per-device-block, NOT a permutation (no user matches). `capture_or_execute_trace` is partially involved (trace-off recovers a subset of users but not all), so the corrupting pass runs in the compiled path even at opt0.

## IMPORTANT CORRECTIONS
- The default-config decode (the actual CI failure) does **NOT** use the `point_to_point`/`all_to_all`/`reduce_scatter` reshard analyzed below — that was the `QB2_KV=batch` *experiment* variant. The default config's decode uses **`ttnn.distributed_rms_norm` (cluster_axis=0)** (×3) + **8× `ttnn.all_reduce` (cluster_axis=0/1)** and only a final vocab `all_gather`. So the p2p-dealloc fix below does **not** fix the default config (verified: still PCC 0).
- The earlier "`distributed_rms_norm` ruled out" claim is **WITHDRAWN**: the `QB2_NORM=replicate` experiment did not actually remove the op (compiled decode still emits `distributed_rms_norm` count=3 — it is driven by the batch-axis-sharded *activation*, not the norm weight). `distributed_rms_norm` (fused multichip `rms_allgather`) is a **prime live suspect** for the default config, along with `all_reduce(cluster_axis=0)`.

Still proven & unchanged: it is a **compiler bug** (eager 2×2-sharded decode = 0.9999 vs compiled = 0.0), triggered by batch-axis sharding, with uninitialized-memory-scale explosion; not opt-level, not trace, not const-eval, not KV-spec; sdpa_decode & reduce_scatter kernels are individually correct.

---

## (Separate latent bug, kv_batch variant only) — premature free of the reshard buffer in TTNNDeallocate

The 2D-mesh decode reshards batch↔hidden via an `all_to_all` lowered to a **multi-hop `ttnn.point_to_point` chain** (TTIRToTTNN.cpp:~2952). `ttnn.point_to_point` takes an `optional_output_tensor` (a `[MemWrite]` buffer) and **its result aliases that buffer** (it writes the received data into the caller buffer and returns it). But `TTNN_PointToPointOp` is `TTNN_Op<[OpModelExempt]>` — **not** a `DestinationStyleOpInterface` op — so the deallocation pass (`lib/Dialect/TTNN/Transforms/Passes.cpp`, `TTNNDeallocate`) does not model the operand↔result aliasing.

Result (from the exported ttnn IR): the pass frees the buffer immediately after the p2p, while the aliasing result is still live:
```
%12 = ttnn.point_to_point(%9, %10)   // %12 aliases buffer %10
ttnn.deallocate(%10)                 // frees %12's buffer  <-- PREMATURE
%14 = ttnn.point_to_point(%9, %12)   // reads %12 (freed)   <-- use-after-free
ttnn.deallocate(%12)                 // same buffer          <-- double free
%16 = ttnn.concat(%14, %15)
```
→ the reshard output is reused/uninitialized → `rms_norm` over garbage → logits ~1e19 / zeroed device blocks → decode PCC 0. Only triggers when the batch axis > 1 (the reshard only exists on a 2D mesh); eager runs each op standalone so the buffer is never reused → 0.9999.

The identical aliasing case for `ttnn.moe_compute` was **already** handled in the same function (skip deallocating a value that feeds a moe optional_output_tensor); `point_to_point` was simply missed.

## FIX (applied, in tt-mlir)
`lib/Dialect/TTNN/Transforms/Passes.cpp`, in `TTNNDeallocate::runOnOperation`'s result walk — mirror the existing `moe_compute` skip for `PointToPointOp`:
```cpp
bool feedsPointToPointOptionalOutput =
    llvm::any_of(result.getUsers(), [&](Operation *user) {
      auto p2pOp = dyn_cast<ttnn::PointToPointOp>(user);
      return p2pOp && p2pOp.getOptionalOutputTensor() == result;
    });
if (feedsPointToPointOptionalOutput) { continue; }
```
This lets the aliasing p2p result carry the (single, correct, later) deallocation instead of freeing the buffer under the still-live result. Validation on the `kv_batch` variant (which uses `point_to_point`) is pending; this does NOT fix the default config (which has no p2p).

## Repro / instrumentation left in tree
`QB2_*` env knobs + `QB2_EAGER_DUMP` / `QB2_PERUSER` / `QB2_COMPILED_DUMP` in `tests/benchmark/`; `sdpa_decode_repro.py`, `reduce_scatter_repro.py`; preserved IR `~/tp_decode_ttir.mlir`, `~/2d_decode_ttir.mlir`, `~/2d_decode_ttnn.mlir`, `~/tp_decode_ttnn.mlir`.

## ★ EXACT CAUSE (default config) — confirmed by per-op runtime dump + TP control
Env-gated per-op max|abs| dump added to `runtime/lib/ttnn/program_executor.cpp` (`TTXLA_OP_DUMP=1`, syncs before read). Default-config decode, op-by-op:
```
gather.44 (embedding) output                    maxabs = 0.045    ← normal
custom-call.78 = distributed_rms_norm  input    maxabs = 0.045    ← normal input
custom-call.78 = distributed_rms_norm  output   maxabs = 2.9e36   ← EXPLODES
```
TP-only (correct) control: identical op normal. ⇒ **the fused `ttnn.distributed_rms_norm` (cluster_axis=0) in the 2D-mesh DECODE turns a normal 0.045 input into ~1e36 garbage** — the op the issue names. My earlier "rms_norm ruled out" was invalid (the QB2_NORM=replicate experiment never removed the op).

**Mechanism:** runtime calls `ttnn::fused_rms_minimal(input, programConfig, clusterAxis, …, statsTensor, …)`. `statsTensor`=`op->stats()`=`%arg22`, a `1x1x32x32xf32` scratch buffer **shared across all 3 decode norms**. It computes local E[x²], all-gathers stats over cluster_axis=0 into `statsTensor`, then `rsqrt(stats+eps)·x·weight`. `rsqrt(~1e-73)≈3e36` ⇒ the stats buffer holds garbage/uninitialized values for the decode shape (batch=32, seq=1, sharded layernorm program_config block_h=1). Matches the memory-reuse heisenbug (keep tensors alive → different alloc → recovers), is decode-specific (prefill seq=128 norm normal) and 2D-specific (cluster_axis>1).

**Fix direction:** eager & prefill use a *decomposed* norm (`rms_norm_pre_all_gather`+`all_gather`+`rms_norm_post_all_gather`) which is correct. Fix = (a) tt-mlir: emit the decomposed sequence instead of the fused `ttnn.distributed_rms_norm` for the decode (small-seq/sharded) case [tractable, tt-xla-side], or (b) tt-metal: fix `fused_rms_minimal` stats/`statsTensor` handling for the decode shape+config.

## ★★ ROOT CAUSE LOCALIZED TO LAYER: the metal op on Blackhole (not tt-xla lowering, not tt-mlir config)
Device arch confirmed **BLACKHOLE** (qb2). The op is `ttnn.distributed_rms_norm` → `fused_rms_minimal` → experimental `rms_allgather`.
- **Metal op is the culprit, on Blackhole:** per-op runtime dump on real BH silicon shows the op turns input 0.045 → output 2.9e36. tt-metal's decode unit test `tests/nightly/tg/ccl/test_distributed_rms_norm_decode_configs.py` is `@skip_for_blackhole("This is a wormhole test")` (wormhole 8×4 only) — the decode path is **never validated on Blackhole**. Kernel `rms_allgather_device_operation.cpp:34` has an explicit BH limitation ("does not support blackhole dram … does not use an accessor to get the noc address as needed by the fabric api"). Model input is L1-width-sharded so it skirts the DRAM guard but runs the unvalidated BH L1 path → garbage.
- **NOT tt-xla lowering:** rms_norm → fused ttnn.distributed_rms_norm is the intended single-op lowering, not a mis-decomposed composite.
- **NOT a bad tt-mlir config:** the emitted config (L1-sharded, block_w=2, cluster_axis=0) is what the op expects and what wormhole runs correctly.
- **Reconciles with eager (works):** eager uses decomposed primitives (all_gather + plain rms_norm / pre+post), supported on BH; compiled path uses the fused rms_allgather, broken on BH.

**Fix options:** (a) tt-metal: fix `rms_allgather` Blackhole path + add BH test coverage [root fix, keeps fused op]; (b) tt-xla/tt-mlir: on Blackhole, emit the DECOMPOSED distributed rms norm (rms_norm_pre_all_gather + all_gather + rms_norm_post_all_gather) instead of the fused op — matches eager, unblocks #5738 immediately.

## ★ Standalone metal repro with the EXACT IR config (op-vs-config localization)
Extracted the exact `distributed_rms_norm` config from the ttnn IR: program_config `compute_grid=(11,6), block_h=1, block_w=2, subblock_w=1`; input L1 width-sharded, per-device `[1,1,32,4096]`, shard `(32,64)`=1x2 tiles, irregular 64-core grid `[(0,0)-(10,4),(0,5)-(8,5)]`; cluster_axis=0, num_devices=2, eps=1e-5. (`~/tt-xla/dist_rmsnorm_exact.py`)
- **EXACT config on Blackhole (2,2) mesh → `fused_rms_minimal` HANGS** (printed setup, then stuck in the op 200s, killed; no result).
- **Smaller config (hidden 7168, 28-core grid, block_w=4, batch 8) → completes, PCC 0.999937** on Blackhole (`dist_rmsnorm_repro.py`).
⇒ the op is NOT universally broken on BH; **the model's exact decode config breaks it** (hang standalone / garbage in-model per-op dump 0.045→2.9e36). Matches: tt-metal's decode test is `@skip_for_blackhole` (never validated on BH) and the kernel has explicit BH limitations. So: op/kernel problem for this config class on Blackhole — not a tt-xla mis-lowering, and the config is what the op is *supposed* to handle (it does on wormhole).

## ★★★ CORRECTION (after tt-smi device reset) — the op is NOT the bug
Installed tt-smi on host; `tt-smi -r` resets all 4 BH devices. Root methodological issue: every prior standalone hang + kill -9 left the fabric dirty, so post-first-run standalone results were artifacts.
On a FRESHLY-RESET device, `ttnn.fused_rms_minimal` with the model's EXACT IR config (irregular 64-core grid, block_w=2, hidden 8192, batch 32, cluster_axis=0, eps 1e-5) → **PCC 0.9999** (confirmed with zero-init AND garbage-init stats buffer; also rectangular 8x8 grid → 0.9999). ⇒ **the fused distributed_rms_norm op works correctly on Blackhole for the model's exact config in isolation.**
Therefore the earlier "metal op / distributed_rms_norm on Blackhole is broken" localization is WITHDRAWN — it was confounded by (1) device-dirt hangs and (2) an unvalidated per-op-dump read of the norm's L1-width-sharded output (TP control has no distributed norm, so that read was never validated).
**Reconciled root cause = the ORIGINAL heisenbug finding: a tt-mlir COMPILED-CONTEXT bug (memory-planning / buffer-aliasing / trace) in the 2D-mesh decode** (eager 0.9999 vs compiled garbage; keeping intermediates alive collapses the explosion). The norm is where it *manifests*, not the cause. Changing the op config won't fix it (config is correct, op works). Repro of op-correctness: ~/tt-xla/dist_rmsnorm_exact.py + `tt-smi -r` first. Device reset: /home/mvasiljevic/.ttsmi-venv/bin/tt-smi -r (host).

## ★★★★ RELIABLE FOUNDATION (all on tt-smi-reset clean devices) — supersedes earlier dirty-device runs
Earlier in-model runs (trace-off "partial recovery", per-user pattern, etc.) were CONFOUNDED by device-dirt (no reset between runs). Re-run each on a freshly `tt-smi -r` device:
| Config (clean device) | First-decode PCC |
|---|---|
| eager (whole 1-layer 2D decode) | **0.9999** (all ops normal) |
| compiled 2D, opt1 | **0.0** (rel_l2 5.9e19) |
| compiled 2D, opt0 | **0.0** |
| compiled 2D, trace OFF | **0.0** (identical) |
| compiled 2D + keep intermediates alive (output_hidden_states) | **recovers (logits 9.56)** |
| all individual ops standalone (distributed_rms_norm exact cfg, sdpa_decode, reduce_scatter) | **correct** |

**Reliable conclusion:** a **buffer-reuse / allocation-collision bug in the base compiled 2D-decode memory management** — independent of opt-level, trace, and const-eval; fixed by keeping intermediates alive; no individual op is at fault. NOT the greedy L1-spill (opt0 has it off and still fails). Most consistent with **async CCL ops (distributed_rms_norm's rms_allgather + cluster_axis=0 all_reduce) whose operand/result buffers get deallocated and their memory reused before the async op completes** — same CLASS as the point_to_point `optional_output_tensor` premature-dealloc I fixed for the kv_batch variant (TTNNDeallocate not modeling an op's true buffer lifetime). Standalone ops pass because single-op + explicit `synchronize` leaves no subsequent reuse. Device reset REQUIRED between hardware runs: /home/mvasiljevic/.ttsmi-venv/bin/tt-smi -r (host).

## ★★★★★ DECISIVE: the compiled decode READS UNINITIALIZED MEMORY (nondeterministic)
Determinism test (reset once, then 3 runs, same 2D-opt1 config):
- RUN 1 (first after `tt-smi -r`): decode output **constant** → `PCC computation failed: denominator is zero` (reads freshly-**zeroed** memory).
- RUN 2: decode rel_l2 = **4.15e19**.  RUN 3: rel_l2 = **5.6e19** (garbage, **different each run**).
Prefill is deterministic (0.999320 every run); only decode varies.
⇒ **The compiled decode reads a buffer that was never written** — content = whatever's in device memory (zeroed right after reset → constant; stale from prior runs → varying garbage). This is a use-of-uninitialized/unbound-buffer bug, NOT a numeric op error and NOT a deterministic dealloc-timing bug. Fits: eager writes+reads the right buffer (0.9999); keep-alive keeps the intended buffer bound (recovers); trace/opt/const-eval independent; all ops correct standalone (given correctly-written inputs). KV-cache read (SDPA-decode over `%arg3/%arg4 [32,4,128,128]`) is the prime candidate: a zeroed cache → attention over zeros → constant logits = RUN 1 exactly. Localizing the exact uninitialized buffer via per-op dump on a FRESH device (uninit=0, unambiguous). Determinism repro: `~/determinism.sh`.

## ★★★★★★ KEY: model runs DECOMPOSED norm, not fused — I tested the wrong path
Codegen (graph_0/main.py) + runtime per-op dump show the distributed rms norm is executed as the DECOMPOSED sequence:
  ttnn.rms_norm_pre_all_gather -> ttnn.all_gather(stats, dim=3, cluster_axis=0, Ring) -> reshape/slice -> ttnn.rms_norm_post_all_gather
NOT the fused rms_allgather I tested standalone (which passed 0.9999). This resolves the contradiction: the fused op is fine, but the model doesn't use it.
- PREFILL norm: input [1,1,576,4096], INTERLEAVED DRAM -> works (0.999).
- DECODE norm: input [1,1,32,4096], WIDTH-SHARDED L1 (ttnn_layout59, irregular 64-core), stats all_gather cluster_axis=0 -> FAILS.
Fresh-device decode per-op dump: gather.44 (embed)=0.045 normal; custom-call.78 (decode norm) early sub-stages 0.045, later sub-stages -> 2.9e36 (dump-perturbed; clean run -> constant/uninitialized). So corruption enters INSIDE the decode's decomposed norm (pre_all_gather / all_gather-of-stats / post), on the sharded-L1 small-shape config.
NEXT: standalone repro of the DECOMPOSED decode norm (pre+all_gather+post) with decode config (sharded-L1 [1,1,32,4096], cluster_axis=0) on fresh device; bisect pre vs all_gather vs post; test interleaved-DRAM vs sharded-L1 input (prefill uses interleaved and works).

## ★★★★★★★ LOCALIZED: decode's WIDTH-SHARDED-L1 decomposed-norm path is the bug
Standalone decomposed norm (rms_norm_pre_all_gather + all_gather(cluster_axis=0,Ring) + rms_norm_post_all_gather), hidden 8192, 2 devs, 32 rows, eps 1e-5, on clean (2,2) device:
- MEM=interleaved DRAM (prefill's layout): **PCC 0.99998** ✓
- MEM=width-sharded L1 (decode's layout): tt-metal FATAL "Sharded inputs require sharded outputs" (path is finicky/unvalidated on BH).
Combined with: prefill (decomposed, interleaved-DRAM, 576 rows) works; decode (decomposed, width-sharded-L1, 32 rows) reads uninitialized memory (fresh->constant, reused->nondeterministic garbage). The decode norm test in tt-metal is @skip_for_blackhole.
⇒ ROOT CAUSE: the DECODE distributed rms norm runs DECOMPOSED on a WIDTH-SHARDED L1 input+stats path that is not correct/validated on Blackhole — it reads uninitialized buffer(s) (likely the sharded stats all_gather / sharded layernorm scratch). Prefill's interleaved-DRAM decomposition is fine.
TRUE FIX candidates: (a) tt-metal: fix the sharded-L1 distributed-rmsnorm (pre/all_gather/post) path on Blackhole (proper buffer init) + add BH test coverage; (b) tt-mlir: route the decode norm through the interleaved-DRAM decomposition (proven correct here) instead of width-sharded L1 for this shape on Blackhole — a layout fix, not a disable. Repro: ~/tt-xla/dist_rmsnorm_decomposed.py (RMS_MEM=dram|l1); tt-smi -r + ~20s before each run.

================================================================================
## ★★★★★★★★★★ DEFINITIVE ROOT CAUSE (issue #5738) ★★★★★★★★★★
================================================================================

### Which op reads the uninitialized buffer
Per-op magnitude dump of the DECODE program (`trace_1_main`) on a fresh device
(TTXLA_OP_DUMP=1 TTXLA_OP_DUMP_PROG=trace_1_main). Execution order:
  1  add.9          maxabs 0
  ...
  5  gather.44      maxabs 0.04492   (token embedding lookup — GOOD)
  6  custom-call.78 maxabs 0.04492   (norm, first partial reads — GOOD)
  8  custom-call.78 maxabs 2.908e+36 (norm, later reads — GARBAGE)
  11 dot.80         maxabs 7.3e+36   (everything downstream is now garbage)
==> **The FIRST op to emit garbage is `custom-call.78` — the first RMSNorm**,
    executed immediately after the token-embedding gather. Its output is
    PARTIALLY correct (0.04492, matching the input magnitude) and PARTIALLY
    garbage (2.9e36) — the exact signature of a partially-uninitialized read.
    All subsequent explosions (dot/add/select NaN) are downstream fallout.

### Correction to earlier notes
The DECODE norm is **NOT decomposed**. The decode ttnn IR
(codegen_2d/graph_1/ttnn.mlir:325) is the **FUSED** op:
  `ttnn.distributed_rms_norm(... cluster_axis=0,
     program_config = layernorm_sharded_multicore<grid=<11,6>, block_w=2, ...>)`
  input/output = #ttnn_layout59 (width-sharded L1).
(Only the PREFILL norm — graph_0 — is decomposed on interleaved DRAM; that path
is correct. Earlier "model runs decomposed" applied to prefill.)

### The exact defect — phantom cores in a non-rectangular shard grid
`#ttnn_layout59` (decode norm in/out):
  width_sharded, grid <1x64>, shard <1x2 tiles> in L1,
  core_ranges = [ (0,0)-(10,4) , (0,5)-(8,5) ]  = 55 + 9 = **64 cores**.
But the fused kernel's compute grid + all-gather semaphore =
  core_range (0,0)-(10,5) = 11x6 = **66 cores** (the BOUNDING BOX of the shards).

The 64 shard cores form a NON-RECTANGULAR region (5 full rows of 11 + a partial
6th row of 9). Its bounding box is the 66-core rectangle. The 2 leftover cells
**(9,5) and (10,5)** are "phantom cores": inside the compute/mcast rectangle but
holding NO shard data -> their L1 is uninitialized.

tt-metal confirms it multicasts/reduces over the whole rectangle
(`rms_allgather_program_factory.cpp:288-296`):
  "num_mcast_dests = num_cores_x * num_cores_y ... the full rectangle, which may
   be larger than num_blocks (the shard worker count) when the shard grid is
   non-rectangular." (They credited the NoC ack counter for the rectangle to
   avoid a hang, but the phantom cores still participate in the STATS reduction
   with uninitialized L1.)

RMS-norm reduces sum(x^2) across ALL cores in the grid; the 2 phantom cores
inject uninitialized partial-stats into that reduction -> the 1/rms denominator
is corrupted -> the whole normalized output is garbage. Fresh device -> phantom
L1 is a constant (=> constant/denominator-zero output); reused device -> varying
prior contents (=> nondeterministic ~1e19..1e36). Matches every symptom
(nondeterminism, keep-alive "fix", fresh-vs-reused difference).

### Why Blackhole-only (and why the model "worked before")
The shard count is chosen in tt-mlir
(DistributedRMSNormWidthShardInputRewritePattern.cpp) as the largest divisor of
numWidthTiles (=4096/32 = 128) that is <= the physical core count, then placed
canonically (row-major, wrapping at the grid WIDTH):
  * Wormhole grid 8x8=64 -> numCores=64 -> fills an 8x8 rectangle EXACTLY.
    Bounding box == shard set. No phantom cores. CORRECT.
  * Blackhole grid 11x6=66 -> largest divisor of 128 that is <=66 is 64 ->
    64 cores wrap to 55+9 -> non-rectangular -> 66-core bbox -> 2 phantom cores.
    BROKEN.
64 cannot form ANY rectangle inside an 11x6 grid (64's divisors <=11 are
1,2,4,8, whose partners 64,32,16,8 all exceed 6 rows). So the geometry itself is
the trap. This is also why the tt-metal decode distributed-rms-norm test is
`@skip_for_blackhole` — the fused sharded path was never validated on BH.

### THE FIX (tt-mlir, true fix — keeps the fused op, no disabling)
File: lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/
      DistributedRMSNormWidthShardInputRewritePattern.cpp
Instead of "largest divisor, canonical wrap", choose a **rectangular** core grid
(gridW x gridH) that (a) fits the physical worker grid, (b) divides numWidthTiles
evenly, (c) maximizes core count; and set that rectangle as an EXPLICIT
CoreRangeSet (not canonical placement). The shard cores then fill their bounding
box exactly on every arch, so num_mcast_dests == worker count -> no phantom
cores -> no uninitialized read.
  * Blackhole 128 tiles, 11x6 grid -> 8x4 = 32 cores (block_w=4). Rectangular.
  * Wormhole  128 tiles, 8x8  grid -> 8x8 = 64 cores (unchanged). No regression.
The fix only changes cases where the old canonical placement was already
non-rectangular (i.e. exactly the buggy configs).

### Alternative / deeper fix (tt-metal)
The most fundamental fix belongs in `rms_allgather_program_factory.cpp`: when the
shard grid is non-rectangular, the phantom cores in the bbox must be excluded
from (or zero-initialized before) the stats reduction, not just credited in the
ack counter. That would let tt-mlir keep 64 cores on BH. Filed as follow-up; the
tt-mlir rectangular-grid fix above is the correct, self-contained compiler-side
fix and is what is validated here.

================================================================================
## ★★★★★★★★★★ FIX VALIDATED ★★★★★★★★★★
================================================================================
Rebuilt tt-mlir + tt-xla with the rectangular-grid fix. Ran the 2D-mesh decode
(QB2_MESH=2d QB2_OPT=1, --num-layers 1 --pcc-decode) 3x: run 1 on a freshly
reset device, runs 2-3 on the reused device (the bug was nondeterministic and
fresh-vs-reused dependent). All three PASS:
  run 1 (fresh):  Prefill PCC 0.999320 | First decode PCC 0.999835  PASSED
  run 2 (reused): Prefill PCC 0.999320 | First decode PCC 0.999810  PASSED
  run 3 (reused): Prefill PCC 0.999320 | First decode PCC 0.999829  PASSED
Before the fix: decode PCC ~0 / NaN, nondeterministic across runs. After: stable
~0.9998 on both fresh and reused devices. The uninitialized read is gone.
On Blackhole the decode norm now uses an 8x4 = 32-core rectangular width-shard
grid (was 64-core non-rectangular). Wormhole is unchanged (8x8 = 64).
Log: debug_logs/validation_after_fix.log

================================================================================
## CI PERF + CLEAN BRANCHES (restore 2D-mesh qb2 coverage)
================================================================================
Clean, mergeable branches (fix isolated from debug/instrumentation commits):
- tt-mlir: mvasiljevic/5738-fix-distributed-rmsnorm-blackhole  (off 9f06802f = SHA
  pinned by latest tt-xla main; ONLY the rectangular-grid fix). commit 77cd34576bd3.
- tt-xla:  mvasiljevic/5738-restore-2d-mesh-tests  (off origin/main). Keeps
  test_llama_3_1_70b_tp_qb2 (tp1x4, (1,N) mesh, opt2) UNCHANGED and ADDS
  test_llama_3_1_70b_tp_qb2_2d (tp2x2, loader-default (2,N//2) mesh, opt1) to restore
  2D weight-sharded + distributed_rms_norm coverage. Adds 4 perf-bench-matrix.json
  entries (perf+accuracy for the 2d variant); no shared-runners field => dedicated
  qb2-blackhole device ("shared device = false").

How to run in CI: Performance Benchmark workflow (manual-benchmark.yml, dispatch) with
mlir_override=77cd34576bd3e183c182a2367148caf36ce722e5, test_filter=llama_3_1_70b_tp_qb2,
runs-on-filter=qb2-blackhole, sh-runner=false, skip-device-perf=false. Expands the
matrix into `pytest <entry.pytest> --output-file ...` (no --pcc/--num-layers => full
model perf). The dispatch auto-sets accuracy-testing:false, so it runs exactly the two
perf entries (tp1x4 + tp2x2).

CI run 29944004903 (branch mvasiljevic/5738-restore-2d-mesh-tests, built at the fix SHA):
- build (tt-mlir @ fix + tt-xla): success.
- tp1x4 (baseline, FULL 80-layer): PASS in 57m. Prefill PCC 0.996180, decode PCC 0.994723.
- tp2x2 (2D, FULL 80-layer): first attempt FAILED at 2x2 mesh-open with a FABRIC/TOPOLOGY
  abort (NOT the norm bug):
    TT_FATAL topology_mapper.cpp:546 mapping_result.success — "Graph specified in MGD could
    not fit in the discovered physical topology. Inter-mesh mapping failed ... STRICT."
  Same transient bad-fabric/device-contention issue seen locally (cured by tt-smi -r).
  Both perf jobs had launched concurrently on the qb2 pool -> contention. Reran the failed
  job alone (gh run rerun --failed) after tp1x4 released the machine.
- Local control: 2D perf-mode (warmup+timed decode loop, 1-layer, opt1) on a freshly-reset
  device PASSES with the fix -> confirms the 2D perf path/code is correct; the CI 2d failure
  was infra. (debug_logs/perf2d_local_1layer_PASS.log)
- 2d rerun result: <pending — see CI run 29944004903>.
Note: benchmark harness emitted "found 4 perf metrics files, expected 2 -> Skipping perf
metrics" for tp1x4 (pre-existing harness quirk, unrelated to this fix); PCC still asserted.
