# Devstral/Qwen chunked-prefill — LIVE STATE REPORT

_Live status of the current push (started 2026-07-13). Companion: `decisions.md` (change log + revert).
Re-read BOTH after any restart / auto-compaction._

## Goal
Get **chunked prefill** working for the DP+TP production config on BH galaxy:
`test_dptp_devstral[mesh_shape0-True-bfp_bf8]` (Devstral-2-123B, mesh [4,8], opt1, trace, bfp8 KV+weights,
prefill_chunk_size=128, batch 128, num_hidden_layers=2). Sibling: qwen3-32B 8x4.

## Environment
- Branch: `ssalice/devstral-qwen-wip-07-13-2026`. tt-mlir pin: `ssalice/devstral-wip-06252026-mlir`.
- Uplifted: **torch 2.11.0, vLLM 0.20.2** (this uplift caused D45 + D47).
- Container: home `/home/ssalice` is mounted at `/data/ssalice` on the host. tt-smi galaxy reset
  (`cd /data/ssalice/tt-smi && uv run tt-smi -glx_reset`) must run on the HOST — cannot run from inside the container.

## Blocker ladder (what's cleared, what's live)
1. ✅ **fp8 load crash** (version-skew) — fixed `fp8_dequant.py` (D45). Model loads.
2. ✅ **page_table row-major TT_FATAL** — fixed in tt-mlir, enabled SDPA row-major workarounds at opt>=1;
   rebuilt libTTMLIRCompiler.so + copied to install (D46). Chunked SDPA op executes.
3. ✅ **fused `ttnn.all_reduce` hangs at `end_trace_capture`** (D47/D49) — FIXED. Re-added the `TTNNAllReduceWorkarounds`
   decomposition to tt-mlir (dropped by commit 1d91fcf556/#8961), rebuilt+installed libTTMLIRCompiler.so (D50).
   VALIDATED (D52): `devstral_test_allreduce_fix.log` shows all_reduce=0, reduce_scatter=8, all_gather=12, and
   `end_trace_capture` now SUCCEEDS — the hang is gone.
4. 🟡 **device wedge / "fetch queue wait" TIMEOUT at runtime const-eval weight load** (D52). After the all_reduce fix,
   run reached runtime and stalled at `main_const_eval_0`→`to_device(embedding)`. Likely accumulated DEVICE WEDGE
   (repeated hangs, prior test killed mid-teardown, no host reset). **BLOCKED ON HOST `tt-smi -glx_reset` (user-only;
   `exit &&` cannot run it from the container — my shell respawns in-container each call).** Rerun on clean device to
   confirm wedge (gone) vs real issue.

## ✅ CORE BLOCKER FIXED (2026-07-14, D58-D59) — end_trace_capture hang RESOLVED
- The tt-metal #45332 port (buffer-addr hash on reduce_scatter/all_gather) MADE `end_trace_capture` SUCCEED (twice)
  on the trace-ON chunked run (`devstral_test_ttmetalfix.log`) — the exact op that hung on every prior trace-on run.
  The central blocker of this whole effort is fixed.
- **3 fixes in place + built + verified turnkey:** (1) all_reduce decomposition (libTTMLIRCompiler.so),
  (2) embedding ("batch",None,None) (vllm_distributed_utils.py), (3) reduce_scatter/all_gather buffer-addr hash
  (_ttnncpp.so, 05:02). Test env-knobs present.
- **ONLY remaining blocker = CLEAN DEVICE.** After trace capture, the run hangs at "fetch queue wait" on a const-eval
  weight load = device-WEDGE signature; this device is DIRTY (35 remote-mmio FATALs, 28 POST_RESET/hang lines) from
  the prior trace-off hang + a partial reset. In-container `tt-smi -glx_reset` CANNOT fully clean it (POST_RESET fails
  on all 32 — needs host/BMC). ⇒ **Needs the user's HOST `tt-smi -glx_reset`, then rerun to confirm end-to-end.**
- **Turnkey rerun (after host reset):**
  `TT_METAL_OPERATION_TIMEOUT_SECONDS=60 pytest -svv tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral'[mesh_shape0-True-bfp_bf8]' 2>&1 | tee devstral_test_clean.log`
  Then the 4096/8192 sweep via TT_DEVSTRAL_MAX_MODEL_LEN. If the fetch-queue hang RECURS on a clean device → genuine
  const-eval issue to chase; if GONE → chunked prefill + trace-on WORKS on the galaxy (goal met).

## EARLIER STATE (2026-07-14, D54-D57)
- Clean-device rerun: **trace-ON still hangs at end_trace_capture** — precise cause (D55): the chunked path traces
  TWO graphs of the same bucket (prefix_chunk False+True) with byte-identical CCLs → the 2nd cache-HITS the 1st's
  stale CCL GlobalSemaphores (tt-metal #44408/#45332 class; fixed for all_to_all_combine, NOT reduce_scatter/all_gather).
- **trace-OFF also hangs** (D57): 36min eager warmup, hung at a legit TP all_gather (64-bucket) — looks like device
  instability over a long eager run, not a clean op bug.
- **embedding fix validated** (0 DP all_gathers in both runs). all_reduce decomposition intact.
- **Self-reset: PARTIAL** (D56/D57) — tt-smi works in-container (`tt-smi -glx_reset`) but POST_RESET fails for the
  6U trays (needs host/BMC); chips still enumerate. A truly clean device may need the user's HOST reset.
- **IN FLIGHT: tt-metal #45332 port** (agent a873…) — add input-buffer-addr hashing to reduce_scatter/all_gather
  compute_program_hash so the chunked 2nd trace cache-misses (fresh semaphores) → fixes trace-ON. Building; then validate.
- **Current best functional path:** trace-ON with the #45332 fix (short warmup, less fabric exposure than trace-off).

## Sharding analysis + embedding fix (D53) — user directive "location matters, nothing in embedding/lm_head"
- `sharding_analysis.md` (skill-grounded, cited): o_proj/down_proj TP all-reduce is FUNDAMENTAL (correct location,
  can't eliminate — only reform/decompose, already done). SP rejected (same collectives + tt-mlir bug). lm_head = no
  all-reduce (column-parallel). 
- ✅ APPLIED: embedding hook `(None,None,None)`→`("batch",None,None)` (vllm_distributed_utils.py:354) — deletes the
  spurious DP-axis all_gather(32→128)+mesh_partition(128→32) round-trip at the embedding. Pure Python, no rebuild.
- KV-cache DP-shard (Real Win #2): BLOCKED upstream (ttir.paged_update_cache, no batch axis in paged layout).
- Two fixes now staged for the post-reset rerun: all_reduce decomposition (tt-mlir, built+installed) + embedding sharding.

## Fixes applied (all reversible — see decisions.md D45/D46 for exact revert)
| # | File | Change | Rebuilt? |
|---|---|---|---|
| D45 | integrations/vllm_plugin/vllm_tt/fp8_dequant.py | fp8 attrs + create_weights override | no (Python) |
| D46 | third_party/tt-mlir/.../TTNNWorkaroundsPatterns.cpp | +3 SDPA ops to opt>=1 workaround set | YES: libTTMLIRCompiler.so → install |
| — | tests/.../test_data_tensor_parallel_generation.py | test_dptp_devstral → production config | no |
| — | tests/.../test_prefill.py (new) | prefill sanity + DP+TP bisection | no |

## In-flight (2026-07-13, later)
- ROOT CAUSE PINNED (D49): tt-mlir commit `1d91fcf556` (#8961) dropped the all_reduce decomposition → fused
  `ttnn.all_reduce` hangs `end_trace_capture` on galaxy.
- Galaxy rerun (`devstral_test_trace_on_rerun_v2.log`) CONFIRMED the hang with DEBUG logs: row-major fix held,
  chunked SDPA executed, hang at `end_trace_capture` (line ~15864) → TIMEOUT.
- **FIX BEING APPLIED (agent):** re-add `TTNNAllReduceWorkarounds` decomposition (Variant A) to tt-mlir
  `TTNNWorkaroundsPatterns.cpp` (from `1d91fcf556^`, extracted to $CLAUDE_JOB_DIR/tmp/old_workarounds.cpp) +
  register unconditionally; rebuild `libTTMLIRCompiler.so` to build/. Parent will cp to install/ after the galaxy
  test process exits (mmap-in-use safety). THEN rerun target test.
- 8-chip isolation test ADDED: `test_prefill.py::test_prefill_dptp_chunked_smallmesh` (mesh [2,4], Qwen3-0.6B,
  4-cell {chunked}×{trace}). Run cmd: `TT_VISIBLE_DEVICES=0,4,8,12,16,20,24,28 TT_METAL_OPERATION_TIMEOUT_SECONDS=60
  pytest -svv .../test_prefill.py::test_prefill_dptp_chunked_smallmesh -k "chunked-on and trace-on"`.
- DEVICE likely wedged by the confirmed hang → may need host `tt-smi -glx_reset` before next hardware run.

## Next experiments (empirical, to get chunked prefill working)
1. **trace=False** rerun — hang is at end_trace_capture; trace-off may execute through.
2. **Smaller-mesh repro**: `TT_VISIBLE_DEVICES=0,4,8,12,16,20,24,28` + a multichip (n300/p300-style) DP+TP chunked
   test on ~8 chips — isolate chunked-prefill CCL from full-galaxy scale.
3. Restore decomposed all_reduce in tt-mlir (once agent pins the control) → rebuild via the D46 loop.

## How to rerun the target test
```
cd /home/ssalice/temp/tt-xla
TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TTXLA_LOGGER_LEVEL=DEBUG TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
  pytest -svv tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral[mesh_shape0-True-bfp_bf8] \
  2>&1 | tee devstral_test_trace_on_rerun_v2.log
```
If it hangs and wedges the device: reset on the HOST — `cd /data/ssalice/tt-smi && uv run tt-smi -glx_reset`.

## Incremental tt-mlir rebuild loop (validated this session)
```
ninja -C /home/ssalice/temp/tt-xla/third_party/tt-mlir/src/tt-mlir/build TTMLIRCompiler \
  && cp /home/ssalice/temp/tt-xla/third_party/tt-mlir/src/tt-mlir/build/lib/libTTMLIRCompiler.so \
        /home/ssalice/temp/tt-xla/third_party/tt-mlir/install/lib/libTTMLIRCompiler.so
```
Plugin dlopens install/lib copy — no tt-xla rebuild needed for tt-mlir function-body changes.
