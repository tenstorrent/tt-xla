# Devstral-2-123B + Qwen3-32B chunked prefill on BH galaxy — engineering handoff

> Reconstructed from the recovered timeline of Claude Code session `40409e16`
> (Jul 13–14 2026) plus the surviving working-tree artifacts. Where the timeline
> and the repo disagree, the repo wins; discrepancies are flagged in §9.
>
> Branch: `ssalice/devstral-qwen-wip-07-13-2026`
> tt-mlir pin: `ssalice/devstral-wip-06252026-mlir`
> tt-metal pin: `ssalice/bh_galaxy` (frozen SHA `3113e9138`)
> Companion durable notes from the session: `devstral_batch128_notes/report.md`
> and `devstral_batch128_notes/decisions.md` (decisions D45–D59).

---

## 1. Summary

**Goal:** get **chunked prefill** working for the production DP+TP config on the
Blackhole (BH) galaxy — `test_dptp_devstral[mesh_shape0-True-bfp_bf8]`
(Devstral-2-123B, mesh `[4, 8]` = DP=4 × TP=8, opt level 1, trace on, bfp8
KV+weights, `prefill_chunk_size=128`, batch 128, `num_hidden_layers=2` for fast
compile). Sibling target: Qwen3-32B at mesh `[8, 4]`.

**What happened:** the run climbed a ladder of four distinct blockers, each
root-caused and fixed: (1) an fp8 load crash from the torch/vLLM uplift, (2) a
chunked-SDPA page-table layout `TT_FATAL`, (3) a fused `ttnn.all_reduce` that
hung `end_trace_capture` on the galaxy, and (4) — the deepest — a stale-semaphore
program-cache collision between the chunked path's two byte-identical CCL traces.
Four fixes are in place (2 Python in tt-xla, 1 in tt-mlir, 1 in tt-metal) and the
two native fixes are compiled into the installed `.so`s.

**Current state:** the **central blocker — the `end_trace_capture` hang — is
resolved and observed to succeed** on the last trace-on run
(`devstral_test_ttmetalfix.log`: `end_trace_capture` succeeded twice, the exact
op that hung on every prior trace-on run). **End-to-end has NOT been validated.**
That run then hung at `TIMEOUT: device timeout in fetch queue wait` on the first
runtime const-eval weight load (`main_const_eval_0` → `ttnn.to_device` of the
embedding) — the accumulated-**device-wedge** signature, not a new op bug.

**What is blocking:** a clean device. The galaxy is wedged (repeated hangs +
tests killed mid-teardown), and the **in-container `tt-smi -glx_reset` cannot
fully clean the 6U trays** (`POST_RESET failed` on all 32 chips — needs
host/BMC access). The next step needs a **host-side `tt-smi -glx_reset`** followed
by a rerun (command in §6). Also unverified: the `max_model_len` 1024→4096→8192
sweep never completed on a clean device.

---

## 2. Fixes landed (all UNCOMMITTED — see §7 for the persistence risk)

Four code changes plus test/config edits. None are committed. Two live in
vendored submodule source and exist as compiled artifacts only in the rebuilt
`.so`s under `third_party/tt-mlir/install/lib/`.

| # | Layer | File | Status | Rebuild? |
|---|-------|------|--------|----------|
| 1 | tt-xla (Python) | `integrations/vllm_plugin/vllm_tt/fp8_dequant.py` | uncommitted (working tree) | none |
| 2 | tt-xla (Python) | `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` | uncommitted (working tree) | none |
| 3 | tt-mlir (C++) | `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` | uncommitted on `ssalice/devstral-wip-06252026-mlir` | `libTTMLIRCompiler.so` (Jul 13 22:21) |
| 4 | tt-metal (C++) | `ttnn/.../ccl/reduce_scatter/device/reduce_scatter_device_operation.cpp` and `.../ccl/all_gather/device/all_gather_device_operation.cpp` | uncommitted on `ssalice/bh_galaxy` | `_ttnncpp.so` (Jul 14 05:02) |

Plus non-code edits (tt-xla working tree, uncommitted):
- `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py`
  — `test_dptp_devstral` promoted to the full production knob set + env-overridable
  knobs (`TT_DEVSTRAL_MAX_MODEL_LEN`, `TT_DEVSTRAL_TRACE`).
- `tests/integrations/vllm_plugin/generative/test_prefill.py` — **new, untracked**
  file (prefill sanity + DP+TP chunked-hang bisection).

### Fix 1 — fp8 dequant version-skew (tt-xla, Python) — blocker #1

`TTFp8DequantLinearMethod` subclasses vLLM's `Fp8LinearMethod` but skips
`super().__init__()`. The torch 2.11 / vLLM uplift changed the base class:
`__init__` now sets `activation_quant_key` / `weight_quant_key` / `input_dtype`,
and it **moved `init_fp8_linear_kernel()` out of `__init__` into
`create_weights()`**. On the Tenstorrent OOT platform that kernel-init call
raises `KeyError('OOT')`, and the base `create_weights` (reused verbatim) now
reads the three unset attributes → `AttributeError` at model load.

Fix (in `fp8_dequant.py`): set the three placeholder attrs in `__init__`, and
override `create_weights` to monkeypatch `init_fp8_linear_kernel` to a no-op
around the base call (the dequant path never uses `self.fp8_linear`). Pure
Python, no rebuild.

### Fix 2 — embedding DP round-trip (tt-xla, Python) — sharding cleanup

In `partition_vocab_parallel_embedding` (`vllm_distributed_utils.py`, ~line 347→354)
the embedding-output forward hook constrained the output to `(None, None, None)`
(fully replicated). Because model inputs are pinned batch-sharded on the DP axis,
that forced a per-forward DP-axis round-trip: `all_gather` (batch 32→128) then
`mesh_partition` back (128→32). Changed the hook to `("batch", None, None)`,
keeping only the legitimate TP hidden-dim gather (`cluster_axis=1`, 1536→12288).
Validated: post-fix runs show **0 `cluster_axis=0` (DP) all_gathers** (was 2).
The embedding weight itself stays `(None, "model")` (vocab deliberately not
sharded — a vocab shard needs a `CollectivePermute` tt-mlir can't lower,
tt-mlir #3370). Pure Python, no rebuild.

The commented-out lines added to `partition_parallel_lm_head` are dead
scratch (an earlier lm_head shard-spec experiment); the live lm_head spec remains
`("model", None)`.

### Fix 3 — two tt-mlir workarounds in one file — blockers #2 and #3

`third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`
(+213 lines, uncommitted). **Both** fixes confirmed present in the working-tree
diff:

- **(3a) SDPA row-major op-enablement — blocker #2.** Added three op names to
  `enabledOpsForWorkaroundWithOptimizer` (the restricted set used at
  `optimization_level >= 1`):
  `ttnn::ChunkedScaledDotProductAttentionOp`,
  `ttnn::PagedScaledDotProductAttentionDecodeOp`,
  `ttnn::PagedFlashMultiLatentAttentionDecodeOp`.
  This lets the existing row-major operand workarounds
  (`createChunked.../createPaged...` in `TTNNWorkaroundsPass.cpp`) run at opt≥1.
  Full root cause in §3.

- **(3b) `TTNNAllReduceWorkarounds` re-added — blocker #3.** The full decomposition
  pattern class (`ttnn.all_reduce` → `reduce_scatter` + `all_gather`, with a
  `rewriteAsAllGatherLocalReduce` fallback for non-divisible axes) was re-added
  verbatim from the commit before it was deleted, and registered
  **unconditionally** (`patterns.add<TTNNAllReduceWorkarounds>(&getContext())` —
  not opt-gated). Full root cause in §4.

Built via the incremental loop (§6) into `libTTMLIRCompiler.so`, atomically copied
to `third_party/tt-mlir/install/lib/` (the copy the plugin `dlopen`s). No tt-xla
rebuild needed — the changes are function-body / pattern-registration, ABI-stable.

### Fix 4 — tt-metal #45332 port (tt-metal, C++) — blocker #4

`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal` on branch
`ssalice/bh_galaxy`, two files modified (uncommitted):
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp`

Both add `tensor_args.input_tensor.buffer()->address()` to the op's
`compute_program_hash(...)`, mirroring the existing `all_to_all_combine` fix from
tt-metal PR #45332. This forces a program-cache **miss** (hence fresh
`GlobalSemaphore` addresses) whenever the input buffer address changes, while
staying stable across genuine replays that reuse the same buffer. Full root cause
in §4. Built into `_ttnncpp.so` and installed to `third_party/tt-mlir/install/lib/`
(Jul 14 05:02).

---

## 3. The chunked-SDPA page-table layout bug (blocker #2)

**Symptom:** `chunked_scaled_dot_product_attention` `TT_FATAL`ed — the page_table
arrived TILE-layout (`memref<...!ttcore.tile<32x32, si32>>`) but the tt-metal
kernel strictly requires ROW_MAJOR + INT32 and does **not** auto-convert
(`sdpa_device_operation.cpp:212`, "Page table must be row major"). Only appeared
on DP+TP; single-device chunked prefill (`test_chunked_prefill.py`) worked.

**Root cause (final, after two discarded interim hypotheses):** tt-mlir
unconditionally tilizes every op operand during layout assignment
(`TTNNLayout.cpp:243`), so the page_table becomes TILE regardless of the plugin or
the CCL. A compiler workaround already exists —
`createChunkedScaledDotProductAttentionOpOperandsWorkarounds`
(`TTNNWorkaroundsPass.cpp:1202`) forces ROW_MAJOR on both `page_table` and
`chunk_start_idx` — and it is wired to the op. But it only fires when the op is in
`enabledOpsForWorkaroundWithOptimizer`, and at `optimization_level >= 1` that
restricted set is the one used (`TTNNWorkaroundsPatterns.cpp:508`). The set
contained Conv3d / Sampling / ArgMax but **none of the SDPA index ops** — so at
opt≥1 the workaround was silently skipped, the page_table stayed TILE, and the
kernel fatal'd.

**The discriminator is `optimization_level`, not DP+TP.** single-device works
because it ran at opt0 (all ops workaround'd). Two earlier interim guesses were
explicitly **wrong** and are recorded so nobody re-chases them: (a) the
`cluster_axis=0` `mesh_partition` does *not* tilize the page_table (it forces its
own I/O row-major), and (b) a runtime `to_layout` is *not* the fix (the decode
runtime doesn't do that either) — the compiler owns it.

**Fix:** §2 fix 3a. Added all three SDPA index ops (chunked = the prefill blocker;
the two paged-decode ops would fatal identically once decode is reached) in one
rebuild. Verified: chunked SDPA op executes past the fatal at opt1.

---

## 4. The CCL / trace-capture hang (blocker #3, then #4)

This is two nested regressions. Fix 3b cleared the first; it exposed the second,
which fix 4 cleared.

### 4a. Fused `all_reduce` hangs `end_trace_capture` (blocker #3)

**Symptom (`devstral_test_trace_on_rerun_v2.log`):** chunked SDPA executed, then
`ttnn.end_trace_capture` hung → tt-metal 60s device TIMEOUT "waiting for physical
cores 15-3, 15-2". Reproduced twice with DEBUG logs.

**Decisive comparison vs the pre-uplift PASS run** (`devstral_1024_bench_PASS.log`):

| | TP all-reduce lowering | `end_trace_capture` | Result |
|---|---|---|---|
| PASS (pre-uplift) | decomposed: ~120 `reduce_scatter` (+all_gather), 0 `all_reduce` | 48, all fine | PASS |
| rerun_v2 (post-uplift) | fused: 32 `ttnn.all_reduce`, 0 `reduce_scatter` | hangs | HANG |

**Root cause pinned to a single tt-mlir commit:** `1d91fcf556` (#8961, 2026-07-06,
"[TTNN] Reshape sub-4D all_reduce to 4D; drop reduce_scatter decomposition") — an
ancestor of the pinned tt-mlir HEAD — **deleted the `TTNNAllReduceWorkarounds`
pattern** (and its `enable-all-reduce-workaround` option). Since that commit,
`all_reduce` lowers to the fused form, which hangs `end_trace_capture` on the BH
galaxy; the decomposed form traced fine on the exact same hardware.

**Fix:** §2 fix 3b — re-add the decomposition pattern (from `1d91fcf556^`),
registered unconditionally. A prior-session caveat (`ca019aa82a`: `reduce_scatter`
deadlocks on cluster axes wider than 2 chips) was checked and does **not** bite
here — it was on a 1×4 *linear* mesh; on this galaxy 2D mesh `reduce_scatter` on
the 8-wide TP axis works (matches the PASS run).

**Validated (`devstral_test_allreduce_fix.log`):** `all_reduce=0`,
`reduce_scatter=8`, `all_gather=12`, and `end_trace_capture` **succeeds**.

> **Honest correction (D52 → D55):** the "fix works" call was first made off
> `devstral_test_allreduce_fix.log`, but that run was on a wedged device. The
> clean-device rerun (`devstral_test_bothfixes.log`) showed trace-on **still hung
> at `end_trace_capture`** — the decomposition alone was necessary but **not
> sufficient**. That reopened the investigation and led to blocker #4.

### 4b. Stale-semaphore program-cache collision (blocker #4) — the deepest one

**Root cause (the precise, actionable mechanism):** it is *not* "CCL-in-trace is
broken." The chunked path compiles both the standard-prefill and cached-prefix
graphs of the same bucket (`prefix_chunk_options=[False, True]`):
- `trace_0` (standard prefill) contains `reduce_scatter`+`all_gather`; its
  `end_trace_capture` **succeeds**.
- `trace_1` (chunked cached-prefix) has **byte-identical** CCLs → they
  program-cache-**HIT** `trace_0`'s CCL programs and reuse `trace_0`'s **stale
  baked `GlobalSemaphore` addresses** → hang.

This is exactly tt-metal's #44408/#45332 stale-RTA bug. #45332 fixed it for
`all_to_all_combine` (by hashing the buffer address to force a cache-miss) but was
**never ported to `reduce_scatter` / `all_gather`**. The pre-uplift PASS run never
hit it: no chunked path, and its buckets had distinct shapes → distinct hashes →
fresh semaphores. So it is the vLLM chunked config newly *exercising* a latent
tt-metal bug — **not** a revertable commit, **not** sharding, **not** the
all_reduce form.

**Fix:** §2 fix 4 — port #45332's input-buffer-address hashing to the
`compute_program_hash` of both `reduce_scatter` and `all_gather` device ops.

**Validated (`devstral_test_ttmetalfix.log`):** `end_trace_capture` **succeeded
twice** on the trace-on chunked run — the exact op that hung on every prior
trace-on run. This is the resolution of the central blocker. (The run then hit
the device-wedge fetch-queue stall; see §1 and §7.)

---

## 5. Sharding analysis

Full writeup: `sharding_analysis.md`. It answers one question — can the TP-axis
RowParallel all-reduce (`o_proj`, `down_proj`, `cluster_axis=1`) be *avoided* by
better sharding rather than merely decomposed?

- **The cross-TP reduction is fundamental.** A Megatron column→row matmul pair
  leaves each TP device with a partial sum; one reduction per pair is the
  theoretical minimum for dense TP. No sharding deletes it. Confirmed the reduction
  lives only at the legitimate locations (`o_proj` / `down_proj`) — nothing at the
  embedding or lm_head.
- **Only the fused `ttnn.all_reduce` *op* is avoidable**, via the
  `reduce_scatter`+`all_gather` decomposition (already shipped, §4a). Measured
  profile matches: `all_reduce=0, reduce_scatter=8, all_gather=12`.
- **Sequence parallelism rejected** as the fix: it emits the *same two collective
  types* as the shipped decomposition (so no better for the trace hang), carries a
  known lowering risk here (a `reduce_scatter → ttnn.rms_norm` graph hits a
  `TTNNDecomposeLayouts` bug, `model_runner.py:1775-1777`), and its only benefit
  (activation memory) is marginal in decode.
- **lm_head:** column-parallel → final gather only, no reduction. Confirmed.

**Embedding DP round-trip fix (applied, §2 fix 2):** the genuine sharding win. The
`(None,None,None)` forced-replication caused a spurious DP-axis `all_gather`(32→128)
+ `mesh_partition`(128→32) every forward. Changed to `("batch", None, None)`.
Validated: 0 DP all_gathers post-fix.

**KV-cache DP-sharding idea (investigated, resolved as already-implemented):** the
user asked whether the replicated KV cache could be *effectively* DP-sharded by
providing correct per-replica batch ids (recalling an `arange` that "got hoisted
out"). Finding: **this is already the implemented design.** The cache is
physically replicated but correctness-DP-sharded — each replica writes/reads only
its own users via a global `batch_idx` arange (`model_runner.py:742-757`) plus an
in-graph `% local_batch` (`attention.py:492-511`); the tt-mlir rule keeps the cache
dim-0 `kNullDim` (replicated) by design. The "arange got hoisted out" memory was an
old tilize crash (since neutralized by tt-metal #8867), **not** a loss of
per-replica semantics. So correctness DP-sharding is **done**. The only open KV
item is **physical DRAM de-replication** (each device holds the global block pool
×4) — a vLLM KV-manager block-pool sizing change, **not** a sharding or tt-mlir
change. A true batch-axis shard of the paged cache remains blocked on
`ttir.paged_update_cache` (the paged layout has no batch/sequence axis;
`model_runner.py:3444-3449`).

---

## 6. Test infrastructure & how to run

### `test_prefill.py` (new, untracked)
`tests/integrations/vllm_plugin/generative/test_prefill.py`. Prefill-graph tests +
the DP+TP chunked-prefill hang reproducer. Constructing `vllm.LLM` triggers warmup
(`capture_model`), which compiles + traces every prefill bucket, so a bad prefill
graph fails at construction before any `generate`. Tests:
- `test_prefill_single_device` / `_chunked` — single-chip sanity/controls (no CCLs;
  isolates a plain/chunked prefill regression from the DP+TP CCL path).
- `test_prefill_dptp_chunked_repro` — the galaxy bisection: 4-cell matrix
  `{chunked-off, chunked-on} × {trace-off, trace-on}` over Devstral `[4,8]` and
  Qwen3-32B `[8,4]`, `num_hidden_layers=2`, opt1, bfp8, `cpu_sampling=True`.
  Expectation: `chunked-off` passes, `chunked-on` hangs in a collective.
- `test_prefill_dptp_chunked_smallmesh` — an 8-chip `[2,4]` carve-out for scale
  isolation. **Deliberately NOT `nightly`** (the chunked-on/trace-on cell is
  designed to hang; only converts to a fast failure when
  `TT_METAL_OPERATION_TIMEOUT_SECONDS` is set). Note: the carve-out
  `TT_VISIBLE_DEVICES=0,4,8,12,16,20,24,28` proved a **bad non-contiguous
  topology** (the excluded chips' eth links are genuinely "remote") — it fails at
  cluster init before reaching compile, so it was **inconclusive** for the fix.
  Needs a valid connected submesh descriptor to be useful.

### The target test — how to run (after a HOST device reset)
```
cd /home/ssalice/temp/tt-xla   # == /data/ssalice/temp/tt-xla on the host
TT_METAL_OPERATION_TIMEOUT_SECONDS=60 pytest -svv \
  tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral'[mesh_shape0-True-bfp_bf8]' \
  2>&1 | tee devstral_test_clean.log
```
Add `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TTXLA_LOGGER_LEVEL=DEBUG` for the verbose
per-op execution trace used throughout the debugging.

Env knobs added to `test_dptp_devstral`:
- `TT_DEVSTRAL_MAX_MODEL_LEN` (default `1024`) — for the length sweep.
- `TT_DEVSTRAL_TRACE` (default `1`) — set `0` to disable trace (bypass the
  `end_trace_capture` path). Note trace-off has its own problems (§7).

### Device reset — **must run on the HOST, not the container**
```
cd /data/ssalice/tt-smi && uv run tt-smi -glx_reset
```
The container's home `/home/ssalice` is bind-mounted to `/data/ssalice` on the
host. `exit && ...` does **not** work from the Bash tool: `exit` short-circuits
the `&&` chain and each tool call respawns inside the container (proven in the
session). The in-container `tt-smi -glx_reset` (after rebuilding tt-smi's venv for
Python 3.12 with tt-umd 0.9.5) *triggers* a reset but `POST_RESET` **fails on all
32 chips** — the 6U tray reset needs host/BMC access the container lacks. So a
truly clean device requires the host reset.

### Incremental native rebuild loops (validated this session)
tt-mlir (function-body / pattern changes, no tt-xla rebuild needed — plugin
`dlopen`s the install copy):
```
ninja -C third_party/tt-mlir/src/tt-mlir/build TTMLIRCompiler \
  && cp third_party/tt-mlir/src/tt-mlir/build/lib/libTTMLIRCompiler.so \
        third_party/tt-mlir/install/lib/libTTMLIRCompiler.so
```
tt-metal (`_ttnncpp.so`) is rebuilt in its own `build_Release` tree and copied to
`third_party/tt-mlir/install/lib/_ttnncpp.so` (the loaded copy). Use an **atomic
rename** when copying while a hung test process may still have the old lib mmap'd.

---

## 7. Open problems / next steps

1. **Device wedge is the immediate blocker.** The last trace-on run got past
   `end_trace_capture` (the win) but hung at `TIMEOUT: device timeout in fetch
   queue wait` on `main_const_eval_0` → `to_device(embedding weight)` — a
   dispatch-fetch-queue stall on a trivial weight load, 0 CCLs executed yet, on a
   device with 35 `remote mmio` FATALs and 28 POST_RESET/hang lines. This is the
   accumulated-wedge signature, almost certainly not a new op bug. **Next:** host
   `tt-smi -glx_reset`, then the §6 rerun.
   - If it runs end-to-end → chunked prefill + trace-on works on the galaxy (goal).
   - If the fetch-queue stall **recurs on a clean device** → it is a genuine
     const-eval weight-load issue and needs its own trace.

2. **Uncommitted / at-risk state (operational risk #1 for the next session).**
   All four fixes are uncommitted; the two native ones live in vendored submodule
   source and exist compiled only in `third_party/tt-mlir/install/lib/{libTTMLIRCompiler.so,_ttnncpp.so}`.
   **Any build that re-fetches the pinned submodule refs will silently drop the
   tt-mlir + tt-metal edits** (the plugin keeps loading the old rebuilt `.so`s
   until then, which masks the loss). To persist: commit fix 3 to the tt-mlir
   branch `ssalice/devstral-wip-06252026-mlir`, and fix 4 to the tt-metal branch
   `ssalice/bh_galaxy` (note tt-metal is itself pinned to a frozen SHA
   `3113e9138`, so plan the re-pin). Also worth an upstream note that
   `enabledOpsForWorkaroundWithOptimizer` is missing the SDPA index ops, and that
   #45332 was never ported to `reduce_scatter`/`all_gather`.

3. **trace-off is not a clean fallback.** `TT_DEVSTRAL_TRACE=0`
   (`devstral_test_traceoff.log`) correctly skips `end_trace_capture` but hung ~36
   min into an eager warmup at a *legitimate* TP `all_gather` (`cluster_axis=1`,
   the 64-token bucket) that had run fine moments earlier — reads as device/fabric
   instability over a long eager run, not a clean op bug. So the durable answer is
   trace-**on** with fix 4, not trace-off.

4. **`max_model_len` sweep — unverified.** Plan: on a clean device, after 1024
   passes, sweep `TT_DEVSTRAL_MAX_MODEL_LEN=4096` then `8192` (both at
   `num_hidden_layers=2`). Both satisfy the chunked-SDPA page-table alignment
   (4096/32=128, 8192/32=256, both `% 8 == 0`). Caveat: the test prompts are short
   (~30 tokens), so these validate KV-cache allocation + page-table sizing + compile
   with no DRAM-OOM at those context sizes — **not** genuine multi-chunk
   long-context streaming. For a real long-context test, lengthen the prompts.

5. **Full-depth serving unvalidated.** `num_hidden_layers=2` is a bring-up device;
   the production model is ~88 layers. `gpu_memory_utilization=0.3` and
   `max_num_batched_tokens` must be re-tuned at full depth.

6. **On-device sampling still blocked** on this 2D-mesh DP+TP path (issues #4387
   trace-insertion crash at opt≥1, #4440 2D-mesh sampler token-soup) — hence
   `cpu_sampling=True` is required in the test.

---

## 8. Key file & log index

### Code / analysis artifacts
| Path | What it is |
|------|------------|
| `sharding_analysis.md` | Full sharding writeup (§5): all-reduce fundamentality, SP rejection, embedding + KV-cache analysis, CCL accounting. |
| `tests/integrations/vllm_plugin/generative/test_prefill.py` | New (untracked) prefill sanity + DP+TP chunked-hang bisection tests. |
| `tests/.../test_data_tensor_parallel_generation.py` (diff) | `test_dptp_devstral` promoted to production config + `TT_DEVSTRAL_*` env knobs. |
| `integrations/vllm_plugin/vllm_tt/fp8_dequant.py` (diff) | Fix 1 (fp8 version-skew). |
| `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` (diff) | Fix 2 (embedding `("batch",None,None)`). |
| `third_party/tt-mlir/src/tt-mlir/lib/.../Workarounds/TTNNWorkaroundsPatterns.cpp` (diff) | Fix 3 (SDPA row-major op-enable + re-added `TTNNAllReduceWorkarounds`). |
| `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/.../ccl/{reduce_scatter,all_gather}/device/*_device_operation.cpp` (diff) | Fix 4 (#45332 buffer-addr hash port). |
| `third_party/tt-mlir/install/lib/libTTMLIRCompiler.so` | Rebuilt with fix 3 (Jul 13 22:21). Plugin-loaded copy. |
| `third_party/tt-mlir/install/lib/_ttnncpp.so` | Rebuilt with fix 4 (Jul 14 05:02). Plugin-loaded copy. |
| `devstral_batch128_notes/report.md` | Session live-state report (blocker ladder, rebuild loops). |
| `devstral_batch128_notes/decisions.md` | Session decision log D45–D59 (each with revert steps). |

`devstral_batch128_notes/{task.md, instructions.md, gemma_8x4_notes.md, NEXT_SESSION_PROMPT.md}` predate this session (Jul 8). The untracked
`TT_INFERENCE_SERVER_INTEGRATION.md`, `data/`, `logs/`, and `~/` are incidental and
not part of this session's technical thread.

### Run logs (chronological; each shows how far the ladder got)
| Log | Lines | What it shows |
|-----|------:|---------------|
| `devstral_1024_bench_PASS.log` | 83,813 | **Pre-uplift known-good reference** (Jul 9). Decomposed CCLs (~120 `reduce_scatter`, 0 `all_reduce`), 48 `end_trace_capture` all fine. The baseline the regressions are measured against. |
| `devstral_test_128x128_FAIL.log` | 15,223 | Early failing 128×128 run (Jul 13, pre-fix). |
| `devstral_test_trace_on.log` | 392 | First run — died at model **load** (fp8 crash, blocker #1). |
| `devstral_test_trace_on_rerun.log` | 15,351 | After fp8 fix — hit the chunked-SDPA page-table row-major `TT_FATAL` (blocker #2). |
| `devstral_test_trace_on_rerun_v2.log` | 15,885 | After row-major fix — the fused `all_reduce` `end_trace_capture` hang, confirmed with DEBUG (blocker #3). The log the user pointed at. |
| `smallmesh_chunkedon_traceon.log` | 215 | 8-chip `[2,4]` carve-out — failed at cluster init (bad non-contiguous topology). Inconclusive. |
| `devstral_test_allreduce_fix.log` | 8,440 | all_reduce decomposition — `all_reduce=0, reduce_scatter=8, all_gather=12`, `end_trace_capture` succeeds. **But on a wedged device** (D52 later corrected). |
| `devstral_test_bothfixes.log` | 15,761 | Clean device, all_reduce + embedding fixes — 0 DP all_gathers (embedding fix good) but trace-on **still hangs** at `end_trace_capture` → exposed blocker #4 (the D52→D55 correction). |
| `devstral_test_traceoff.log` | 20,976 | `TT_DEVSTRAL_TRACE=0` — no trace-capture, but hung ~36 min at a legit TP `all_gather` (eager-run instability). |
| `devstral_test_ttmetalfix.log` | 8,290 | **Decisive** trace-ON run with all 3 native fixes — `end_trace_capture` **succeeded twice** (blocker #4 resolved), then hung at `fetch queue wait` (device wedge). |
| `devstral_test_trace_off.log` | 19,925 | Timestamped Jul 14 **19:14**, after the session ended (~05:14) — a **post-session** rerun, out of this session's scope (see §9). |

---

## 9. Discrepancies (timeline vs repo — repo trusted)

1. **Both tt-mlir fixes live in ONE file.** The timeline narrated the SDPA
   row-major functions at `TTNNWorkaroundsPass.cpp:1202` and the enable-set at
   `TTNNWorkaroundsPatterns.cpp:508`, and the all_reduce class as re-added
   separately. The actual uncommitted edit is entirely within
   **`TTNNWorkaroundsPatterns.cpp`** (+213 lines): both the 3 SDPA op names added
   to `enabledOpsForWorkaroundWithOptimizer` **and** the re-added
   `TTNNAllReduceWorkarounds` class + its unconditional registration. The referenced
   `...Pass.cpp` workaround *functions* are pre-existing and unmodified; only the
   enable-set (in Patterns.cpp) was touched. Both fixes are confirmed present in the
   working tree, so a rebuild reproduces both.
2. **vLLM version.** Session docs (report.md, timeline) say the uplift was **torch
   2.11.0 / vLLM 0.20.2**. The repo's `main` now carries `[vLLM][uplift] v0.22.1
   (#5634)`; this branch was cut before that. Use the session's 0.20.2 as the
   context in which the regressions arose.
3. **Commit hashes rebased.** The git-status snapshot in the task showed hashes
   (`139fb5536`, `10cc84c75`, ...) that differ from the current log
   (`90f09e7fe`, `1db97886d`, ...). The **commit messages match** — the branch was
   rebased/amended between snapshot and now. No substantive change.
4. **Two trace-off logs.** `devstral_test_traceoff.log` (Jul 14 04:51, in-session)
   vs `devstral_test_trace_off.log` (Jul 14 19:14, **after** the ~05:14 session
   end). The 19:14 file is a post-session rerun and is outside this reconstruction's
   scope; treat only the 04:51 one as session evidence.
5. **`fp8_dequant.py` has both a committed base and the uncommitted fix.** There is
   a landed commit "[vllm] Add fp8->bf16 dequant hook for OOT platform"; the §2
   fix 1 version-skew change sits **uncommitted on top of it** in the working tree.
   Likewise `test_dptp_devstral` exists in a landed test commit and has uncommitted
   production-config edits layered on top.
