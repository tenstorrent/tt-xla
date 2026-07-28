# Devstral-123B high-seq-length / chunked-prefill — TASK TRACKER

> **Re-read this file + `decisions.md` (same dir) on every restart and after every auto-compaction.**
> This is the CONTINUATION of `devstral_batch128_notes/chunked_prefill_issue/` (that dir = prior session, D45–D59).

## Goal
Enable **chunked prefill** so we can unlock **higher context lengths** for Devstral-2-123B
on the BH galaxy, and progress tt-inference-server integration. Concrete target test:

```
TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TTXLA_LOGGER_LEVEL=DEBUG TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \
  pytest -svv tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral'[mesh_shape0-True-bfp_bf8]' \
  |& tee <log>
```
Config: Devstral-2-123B, mesh **[4,8]** (DP=4×TP=8), opt1, trace ON, bfp8 KV+weights,
`prefill_chunk_size=128`, batch 128, **`num_hidden_layers=2`** (bring-up). Sibling: Qwen3-32B [8,4].

## Environment (verified 2026-07-16)
- **I run on the HOST `bh-glx-b06u08`** (NOT inside the container — confirmed no /.dockerenv).
  - Prior session's shell respawned *inside* the container; THIS session's Bash is on the host.
  - Tests must run **inside** the container: `docker exec -it --user 4076:4076 tt-xla-ird-ssalice /bin/bash`
  - Container `tt-xla-ird-ssalice` is Up.
- Working tree: `/data/ssalice/temp/tt-xla` (host) == `/home/ssalice/temp/tt-xla` (container). NOT `~/ssalice/tt-xla`.
- Every shell that builds/tests: `export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/ && source venv/activate`.
- Branch tt-xla: `ssalice/devstral-qwen-wip-07-13-2026`. tt-mlir pin: `ssalice/devstral-wip-06252026-mlir`.
  tt-metal pin: `ssalice/bh_galaxy` (frozen SHA 3113e9138aa).
- Galaxy reset (host): `uvx tt-smi@latest -glx_reset` → expect `Re-initialized 32 boards after reset. Exiting...`.
  - **NOTE: the auto-mode classifier BLOCKS this reset without explicit live user approval** (shared HW).
    Needs the user to OK it or add a Bash permission rule. In-container reset is PARTIAL (POST_RESET fails on 6U trays).
- Common env vars: `TTXLA_LOGGER_LEVEL=DEBUG|VERBOSE`, `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG`,
  `TT_METAL_OPERATION_TIMEOUT_SECONDS=60|120`, `TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/`.
- Test env knobs: `TT_DEVSTRAL_MAX_MODEL_LEN` (default 1024; sweep 4096/8192), `TT_DEVSTRAL_TRACE` (default 1).
- **Notion**: hub "Tenstorrent" (id 30232b7acc118054a574dac29c100bed) + subpages "Get bh galaxy working",
  "Devstral 123b BH galaxy bringup", "Devstral 123b tt-inference server integration". (Notion page has live
  secrets — never echo tokens into logs/code.)

## Current fixes in place (all UNCOMMITTED — verified present 2026-07-16)
1. **fp8 dequant** (D45) — `integrations/vllm_plugin/vllm_tt/fp8_dequant.py` (Python, no rebuild).
2. **embedding DP round-trip** (D53) — `vllm_distributed_utils.py:354` `(None,None,None)→("batch",None,None)` (Python).
3. **tt-mlir SDPA row-major (D46) + all_reduce decomposition (D50)** — `TTNNWorkaroundsPatterns.cpp`
   (3 SDPA ops in enabled set ✓, `TTNNAllReduceWorkarounds` present ✓). Built → `libTTMLIRCompiler.so` (15:18 Jul16).
4. **tt-metal buffer-addr hash** (D58) — reduce_scatter + all_gather `compute_program_hash` (`buffer()->address()` ✓).
   Built → `_ttnncpp.so` (14:51 Jul16).
   > **Persistence risk:** fixes 3+4 live only in uncommitted submodule source + rebuilt .so. `git submodule
   > update --init --recursive` or a full rebuild will DROP them. Commit to the tt-mlir/tt-metal branches to persist.

## Blocker ladder status (from prior session, re-verified against today's log)
1. ✅ fp8 load crash — fixed.
2. ✅ chunked-SDPA page_table row-major TT_FATAL — fixed (opt≥1 workaround enabled).
3. ✅ fused all_reduce hangs end_trace_capture — fixed (decomposition; all_reduce=0 in today's log).
4. ✅ stale-semaphore trace-capture collision — fixed (buffer-addr hash; **trace capture SUCCEEDED in today's log, line 9064**).
5. 🔴 **DIRTY/WEDGED DEVICE = the live blocker.** We have NEVER had a clean-device run. Needs full galaxy reset.

## Today's log (`devstral_test.log`, Jul 16 16:16–16:26) — new datapoint
- ~30 `TT_FATAL: ... connects to a remote mmio device` at cluster init. **NOTE (corrected): these are BENIGN
  topology-enumeration messages — the PASS run has 24 of them too. NOT a dirtiness signal. See decisions.md H1-CORRECTION.**
- Got further than any prior run: trace capture SUCCEEDED (line 9064), const-eval weights loaded, executing forward ops.
- Hung at a **reduce_scatter** (`dot.269_all_reduce_4d_reduce_scatter`, cluster_axis=1) ~line 15200,
  "device timeout... waiting for physical cores 15-3, 15-2" → same wedge signature. all_reduce=0, RS=25, AG=31.
- Interpretation: consistent with device wedge (started dirty, limped further than before). NOT proven to be a
  new op bug until reproduced on a CLEAN device.
- **SHARPENED (see decisions.md H1-followup):** the hang is at physical cores **15-3, 15-2** — the SAME cores in
  EVERY failing run (trace-on & trace-off, different ops). Today: devices 0 & 2. Leading hypothesis is now a
  specific bad/wedged fabric location on this galaxy, not a code bug. `tt-smi -ls` shows all 32 boards enumerate
  (alive, resettable). The bucket compiles 2 graphs (prefix_chunk F/T); phase 1 (F) fully succeeded incl. trace
  capture; phase 2 (T) hung at its first reduce_scatter during EAGER exec.

## NEXT STEPS (in order)
1. **[BLOCKED on user OK] Full galaxy reset** `uvx tt-smi@latest -glx_reset` (host) → verify literal
   `Re-initialized 32 boards after reset. Exiting...` (not just exit 0).
2. Rerun target test on clean device (2-layer, inside container). → `devstral_test_clean.log`:
   ```
   docker exec --user 4076:4076 tt-xla-ird-ssalice bash -lc '
     cd /home/ssalice/temp/tt-xla && export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/ && source venv/activate &&
     TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TTXLA_LOGGER_LEVEL=DEBUG TT_METAL_OPERATION_TIMEOUT_SECONDS=120 \
       pytest -svv "tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral[mesh_shape0-True-bfp_bf8]"
   ' 2>&1 | tee devstral_test_clean.log
   ```
   **Decisive read (see decisions.md H1-followup):**
   - FIRST verify cluster init has NO `remote mmio` TT_FATALs. If present → reset didn't take; re-reset before trusting anything.
   - Passes / hangs at a DIFFERENT location → 15-3/15-2 was transient wedge → GOAL likely met for 2-layer.
   - Hangs AGAIN at cores 15-3/15-2 → likely a bad physical link on this node (NOT a code bug). Then check eth-link
     health for the device owning 15-3/15-2, try a different slurm BH node, and only then treat as a CCL/op bug.
3. Sweep `TT_DEVSTRAL_MAX_MODEL_LEN=4096` then `8192` (2-layer) for high-context validation (both satisfy %256).
4. Only after 2-layer + sweep pass: **ASK USER** before running full ~88-layer.
5. Persist fixes (commit tt-mlir `ssalice/devstral-wip-06252026-mlir` + tt-metal `ssalice/bh_galaxy` branches),
   run pre-commit, clean up branches/logs.

## Do-NOT-re-chase (from decisions.md)
- 8-chip smallmesh carve-out (D51): bad non-contiguous topology, inconclusive.
- "1D-mesh fabric hang" theory (D47): CCLs work in other galaxy tests.
- config-revert of chunked prefill (raise min_context_len / drop chunk): defeats the goal.
- Do NOT revert the 3 validated fixes (D50, D53, D58).
- D52's "end_trace_capture succeeds" was over-claimed on a wedged device (corrected by D54); D59 is the real validation.

## Agent-gathered reference (see also decisions.md and prior-session notes)
- Skills most relevant: `sharding-model-analysis`, `graph-break-analysis`, `finding-missed-fusions`,
  `ci-benchmark-analyzer`, `code-reviewer`, `superpowers:systematic-debugging`, `verify`.
- vllm_tt file map + chunked-prefill flow + test-config knobs: see `vllm_tt_reference.md` (same dir).
- Prior-session synthesis (D45–D59 with revert map, sharding analysis): `../chunked_prefill_issue/` and this session's `decisions.md`.
