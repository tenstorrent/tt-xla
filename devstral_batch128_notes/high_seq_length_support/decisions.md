# Decisions log — high_seq_length_support (continuation of chunked_prefill_issue D45–D59)

> Fork-in-the-road decisions for THIS session. New IDs continue as **H1, H2, …**.
> Prior session's D45–D59 live in `../chunked_prefill_issue/decisions.md` — consult it to check whether a
> past decision could be causing a new error. **Re-read this file on every restart / auto-compaction.**

## Inherited state (do NOT re-litigate — from D45–D59)
- KEEP fixes: D45 (fp8), D46 (SDPA row-major opt≥1), D50 (all_reduce decomposition), D53 (embedding batch-shard),
  D58 (tt-metal reduce_scatter/all_gather buffer-addr hash). All verified present 2026-07-16.
- Do-not-re-chase: 8-chip smallmesh (D51), 1D-mesh-fabric theory (D47), chunked-prefill config-revert.

---

## H1 — Reset the galaxy before any diagnosis (2026-07-16)
- **Fork:** today's `devstral_test.log` hung at a `reduce_scatter` during execution. Is it (a) the accumulated
  device wedge, or (b) a genuine CCL/op bug?
- **Evidence:** the run STARTED dirty — ~30 `remote mmio` TT_FATALs at cluster init (lines 56–114). It still got
  further than any prior run (trace capture SUCCEEDED at line 9064; all_reduce=0, RS=25, AG=31). No clean-device
  run has EVER happened across the whole effort.
- **Decision:** the only way to disambiguate (a) vs (b) is a full galaxy reset + rerun. That is the top action.
- **Status:** BLOCKED — auto-mode classifier denied the autonomous `uvx tt-smi@latest -glx_reset` (shared 32-chip
  HW needs explicit live user approval). Task instructions pre-authorized resets, but the harness gate requires the
  user to OK it or add a Bash permission rule. Surfaced to user; doing all non-HW prep meanwhile.
- **Revert:** n/a (a reset is non-destructive to code/data).

## H1-followup — SHARPENED hypothesis: persistent hang localizes to physical cores 15-3/15-2 (2026-07-16)
Deep-mined today's `devstral_test.log` + all prior logs before the (blocked) reset:

- **Same-run structure:** the `num_tokens=128` bucket compiles TWO graphs (prefix_chunk False+True, per D55).
  - Phase 1 (prefix_chunk=False, lines ~8144–9064): const-eval loaded, 8 reduce_scatters (cluster_axis=1) ran
    (4 eager + 4 in-trace), **end_trace_capture SUCCEEDED (9064)**. So trace capture + CCLs work here.
  - Phase 2 (prefix_chunk=True, lines 15036–15252): const-eval reloaded, transformer layer executed EAGERLY
    (no begin_trace_capture before the hang), hung at the **first reduce_scatter (%61, cluster_axis=1, 15252)**.
  - ⇒ 8 identical-shape reduce_scatters succeeded earlier in the SAME run; the hang is the first CCL of the
    second (chunked) graph, during eager execution — a point PAST where any prior run reached.
- **Cross-run signature (the sharp signal):** EVERY failing run hangs at the *exact same physical cores* **15-3, 15-2**:
  `128x128_FAIL`, `bothfixes`, `trace_off`, `trace_on_rerun`, `trace_on_rerun_v2`, `traceoff`, and today. It spans
  trace-ON and trace-OFF and different ops (end_trace_capture / all_gather / reduce_scatter). The PASS run has none;
  the two runs that got furthest (`allreduce_fix`, `ttmetalfix`) hung elsewhere (const-eval to_device), not here.
  Today it's **Device 0 and Device 2** whose cores 15-3/15-2 never finish.
- **`tt-smi -ls` (read-only):** all 32 BH galaxy boards enumerate and are listed resettable → chips alive, not bricked.
  Notion flagged 28/32 devices PCIe-downtrained to x1 historically.

**Leading hypothesis (revised):** a **specific bad/wedged fabric location (cores 15-3/15-2, devices 0/2 in the [4,8]
mesh)** on THIS galaxy, not a code bug. Every run has inherited it (never a clean baseline). A soft `-glx_reset` may
or may not clear a genuinely bad link.

**Decisive test after the (user-approved) reset:**
1. Confirm cluster init has NO `remote mmio` TT_FATALs (else reset didn't take → rerun confounded, re-reset).
2. If run passes or hangs at a DIFFERENT location → 15-3/15-2 was transient wedge (GOAL likely met for 2-layer).
3. If it hangs AGAIN at 15-3/15-2 → likely a bad physical link on this node. Then: check eth-link health for the
   device owning 15-3/15-2, consider a different slurm node (Notion has salloc for named BH nodes), and only THEN
   treat it as a code/CCL bug. Do NOT assume a code fix until a clean device reproduces it.

## H1-CRITICAL UPDATE (user, 2026-07-16): the prior hung run was ALREADY post-reset
User confirmed today's earlier `devstral_test.log` (hung at 15-3/15-2, ~30 remote-mmio FATALs at init) was run
**after a reset**. So a `-glx_reset` did NOT clear the remote-mmio FATALs nor the 15-3/15-2 hang last time. This
strengthens the "bad/persistent fabric location on this node" reading — a soft reset may be insufficient.

Now running a FRESH-reset clean rerun (`devstral_test_clean.log`, bg bzyj5zod5, started ~18:45). Decisive:
- If init STILL shows remote-mmio FATALs after this fresh reset → reset genuinely doesn't clean this node's fabric;
  escalate (different BH slurm node, or deeper/host-BMC reset, or hardware check of the device owning 15-3/15-2).
- If init is clean this time → last reset didn't take; proceed and watch whether 15-3/15-2 recurs.

## H1-CORRECTION (2026-07-16): `remote mmio device` FATALs are BENIGN — not a dirtiness signal
Cross-log check: the **PASSING** run `devstral_1024_bench_PASS.log` also has **24** `remote mmio device` TT_FATALs
and passed. Counts: PASS=24, ttmetalfix=35, allreduce_fix=35, bothfixes=35, rerun_v2=35, today's hung=40, clean=40.
These are galaxy-topology enumeration messages (logged "critical" but caught; runs continue past them). The earlier
"started dirty / reset didn't take (because of remote-mmio FATALs)" reading (H1, H1-followup, H1-CRITICAL) was WRONG
on that point. **The ONLY real discriminator is the 15-3/15-2 execution hang** (PASS=0; hung runs=6/11/2; the two
const-eval failures had 0). So init-time remote-mmio count tells us nothing about whether the reset worked — judge
the reset purely by whether the run progresses through execution without the 15-3/15-2 hang.

## H2 — first fresh-reset rerun KILLED early (inconclusive); user trying a HARDER dev-level reset (2026-07-16)
- `devstral_test_clean.log` (bg bzyj5zod5) was killed by user request during compilation, BEFORE execution — it
  never reached the 15-3/15-2 point. **Inconclusive; do not treat as a result.** (It did confirm remote-mmio FATALs
  reappear post-`glx_reset`, but those are benign per H1-CORRECTION.)
- Rationale for the harder reset: across the effort a soft `uvx tt-smi@latest -glx_reset` has NOT cleared the
  15-3/15-2 hang. A deeper/host-BMC-level reset directly tests whether the bad fabric location clears. Good instinct.
- Device confirmed FREE after kill (no /dev/tenstorrent holders, no leftover EngineCore/pytest).
- **Next after the harder reset:** rerun the turnkey 2-layer test (task.md step 2), judge purely by whether it
  progresses through execution without the 15-3/15-2 hang (ignore remote-mmio count).

## H3 — MAJOR REFRAME: the all_reduce decomposition (D50) was a REMOVED upstream workaround (2026-07-16)
User surfaced tt-mlir commit a4dc183d: the `all_reduce → reduce_scatter + all_gather` decomposition was a
**workaround**, removed upstream once the fused `ttnn.all_reduce` op became stable. So **fused all_reduce is the
INTENDED path**, and the prior session's D50 (re-add decomposition) was fighting an upstream fix.
- Evidence: `devstral_dptp_test.log` (19:46) correctly shows **fused all_reduce=25, reduce_scatter=0** — the current
  committed tt-mlir emits fused all_reduce. (The earlier `devstral_test.log` with reduce_scatter=25 was the OLD
  re-added-decomposition state.)
- **Strategy shift:** do NOT re-decompose. Keep fused all_reduce and ROOT-CAUSE its hang directly.
- **Why fused all_reduce hangs (leading hypothesis):** same byte-identical two-graph program-cache/stale-semaphore
  collision (D55) — phase-1 all_reduce works, phase-2 (structurally identical chunked graph) hangs. And fused
  all_reduce has NO D58-style buffer-addr-hash protection (D58 was only added to reduce_scatter/all_gather).
  Agent `allreduce_collision_analysis` is verifying + whether all_reduce bakes GlobalSemaphores on cache-miss.
- **Caveat (don't tunnel):** breadth of hangs — `devstral_dptp_test_trace_off.log` hangs on a DP-axis `all_gather`
  (`convert.58_all_gather_2d`, cluster_axis=0, ui32 1x4096→1x16384) — so more than one CCL type/axis hangs; could be
  a more systemic 2D-mesh CCL/fabric issue, not only the semaphore collision.

## H4 — embedding weight `(None,"batch")` change BACKFIRED → point_to_point storm (2026-07-16)
User set embedding weight to `(None,"batch")` (hidden on DP) hoping to steer the hidden gather off the TP axis.
Result (`devstral_dptp_test_2.log`): **768 `ttnn.point_to_point` ops, ALL on `loc("gather.59")`** (384/graph ×2).
- Mechanism: replicating a DP-sharded hidden dim on the 2D mesh can't lower as a native all_gather → tt-mlir emits
  a CollectivePermute → O(mesh²) point_to_point sends (the #3370 limitation the embedding comment itself warns about).
  It also reintroduces a DP-axis all_gather (convert.58) — the OPPOSITE of steering CCLs off. And it hangs.
- **Recommendation: REVERT embedding weight to `(None,"model")`** (hidden on TP → one clean all_gather) to get a
  clean baseline and isolate the fused-all_reduce hang. The `(None,"batch")` idea is counterproductive here.

## H5 — all_reduce collision agent verdict: stale-semaphore REFUTED; asymmetric intermediate-buffer PLAUSIBLE
Agent (`allreduce_collision_analysis.md`) findings:
- MLIR "fused" `ttnn.all_reduce` **decomposes to reduce_scatter+all_gather at tt-metal RUNTIME** for this shape
  (1x1x4096x12288, cluster_axis=1, 8-way) → fused-vs-decomposed is a red herring at the metal level.
- **Stale-GlobalSemaphore theory REFUTED**: the reduce_scatter override rewrites ALL buffer+semaphore addresses into
  runtime args on every cache hit; semaphores kept alive in shared_variables. And a byte-identical `all_gather`
  cache-hits post-trace and SUCCEEDS immediately before the all_reduce hangs → not "any byte-identical CCL hangs."
- Program hash IS spec-only (empty attribute_values) → same-spec/diff-buffer DO cache-collide (that part confirmed),
  but the collision alone isn't fatal (the all_gather proves it).
- **SURVIVING hypothesis (PLAUSIBLE, unproven): asymmetric (re)allocation of the reduce_scatter INTERMEDIATE
  (scratch) buffer across the 8 TP devices after graph-1 trace capture.** The ring CCL assumes the intermediate is
  at the same address on every peer; if post-trace allocator state diverges per-device, the intermediate lands at
  different addresses → ring writes to wrong peer address → hang. Confirm via per-device intermediate-address dump
  or a program-cache-disabled A/B run.
- Q2 CONFIRMED: the two runs differ because libTTMLIRCompiler was rebuilt across #8961 (`1d91fcf556`) which DELETED
  the decomposition. opt_level gating was a red herring (it was `allReduceWorkaroundEnabled`, not opt-level).

## H6 — embedding: removing the forward HOOK (keep weight `(None,"batch")`) kills the p2p, gives clean DP all_gather
User removed the `sharding_constraint_hook` output constraint. Result (`devstral_dptp_test_trace_off_no_hook_forward_shard.log`):
- NO point_to_point. Embedding output gather becomes a clean `all_gather(cluster_axis=0, dim=2, 3072→12288)` (%23).
  So the HOOK (`("batch",None,None)`) was the p2p trigger (it forced a token/hidden reshard transpose = all-to-all);
  without it GSPMD gathers hidden in place. Weight `(None,"batch")` alone → clean DP all_gather.
- BUT still HANGS — on a DP-axis `all_gather` in the `num_tokens=64` bucket (got through the 128 chunked+nonchunked
  buckets first). So DP-axis CCLs hang too, and the hang is on a LATER bucket (not just "2nd identical graph").
- User reverted the embedding change for now; may revisit if `(None,"batch")` proves faster.
- **Cross-cutting takeaway:** hangs now seen on BOTH cluster_axis=1 (TP reduce_scatter/all_reduce) AND cluster_axis=0
  (DP all_gather), and on later buckets — consistent with the asymmetric-intermediate-alloc hypothesis (accumulating
  allocator divergence) and/or systemic 2D-mesh CCL fragility, more than a single-op cache collision.

## H6-CORRECTION — no-hook `(None,"batch")` embedding is MORE efficient (the p2p was a HOOK artifact)
Graph slice comparison (`embedding_variant_graph_compare.md`): OG `(None,"model")`+hook (slice A) vs no-hook
`(None,"batch")` (slice B), two decoder layers each. Matmuls IDENTICAL; only the embedding gather region differs.
- **B is the cheaper graph on comms:** drops A's ~50 MB bf16 DP token all_gather (replaced by a ~64 KB ui32 index
  gather + local lookup), and moves the hidden gather from TP(8-way) to DP(4-way) — OFF the TP axis that carries all
  4 all_reduces. No point_to_point in either. ~20–25% less data received/device in the embedding path.
- **A is cheaper only on memory:** embedding shard `131072x1536` vs B's `131072x3072` (~2× less embedding table/device).
- **Correction to H4:** the 768 point_to_points were the FORWARD HOOK forcing a token↔hidden reshard transpose, NOT
  the `(None,"batch")` weight itself. Hook-removed `(None,"batch")` is the genuine "steer CCLs off TP at memory cost"
  win the user intended. KEEP this variant (no hook) once the CCL hang is fixed, IF the 2× embedding memory fits at full depth.
- **Caveat:** efficiency only — BOTH variants still hang downstream on the CCLs. Orthogonal to this win.

## H7 — unit-test repro of the CCL hang: written, but likely NOT reproducible in pure isolation
`repro_allreduce_hang_test.py` + `repro_allreduce_hang_notes.md` written (4-variant matrix on a contiguous
create_submesh [1,8]/[2,8] TP line; runs the o_proj all_reduce; cache-on+trace vs cache-off control; dumps per-chip
output addresses). **Caveat:** a clean mesh is SPMD (host issues identical alloc to every chip), and the ttnn API
can't inject per-chip-divergent allocation — exactly the asymmetry the leading hypothesis needs. So if the isolated
test PASSES, that's a real finding (trigger is model/vLLM-alloc-driven, not op-intrinsic), not a broken test. Confirming
intermediate-buffer divergence definitively still needs a tt-metal-side per-device dump. UNRUN (needs clean device).

## H8 — BREAKTHROUGH: TT_RUNTIME_SYNC_AFTER_OP=1 pinpoints the hang at `paged_fill_cache`, NOT a CCL
Run `devstral_dptp_test_synced.log` (sync-after-op, PR #7649, trace OFF). Sync-after-op makes the reported op the
TRUE culprit (no async pipeline masking).
- **4 all_reduce + 4 all_gather COMPLETED successfully** before the hang → the CCLs are NOT the culprit.
- **5 paged_fill_cache ran; first 4 succeeded (graph 1), the 5th HANGS** (line 14747) — in the SECOND (chunked
  prefix_chunk=True) num_tokens=128 graph. Timeout at cores 15-3/15-2, devices 8/12/16/20.
- ⇒ **The earlier "all_reduce/reduce_scatter/all_gather hangs" (H5, devstral_dptp_test.log, trace_off all_gather)
  were almost certainly ASYNC-DISPATCH ARTIFACTS** — the pipeline stalled downstream of the real stall. The real
  first-to-hang op is the KV-cache WRITE in the chunked graph. This deprioritizes the CCL-collision + asymmetric-
  intermediate-buffer hypotheses (H5) and the CCL unit-test repro (H7).
- **Leading hypothesis:** the hanging `paged_fill_cache(cache, key, page_table, batch_idx)` receives a page_table
  that's been `to_layout`'d INTO TILE (`!tile<32x32,si32>`) — the row-major-vs-tile class that bit chunked SDPA
  (blocker #2). If tt-metal paged_fill_cache assumes ROW_MAJOR page_table, a tiled one → garbage block addresses →
  hang. The SDPA fix added row-major workarounds for the SDPA index ops but maybe NOT for paged_fill_cache/update_cache.
- Agent `paged_fill_cache_hang_analysis` investigating: succeeding-vs-hanging fill arg layouts, tt-metal
  paged_fill_cache layout requirements, whether a row-major workaround exists/enabled at opt1, and the tt-xla
  chunked fill_page_table/batch_idx path. Candidate fix mirrors the SDPA row-major op-enable.
- **Caveat:** sync-after-op serializes execution, so it could shift timing/allocation; treat "paged_fill_cache is the
  culprit" as strong-but-verify. But it's far more trustworthy than the async logs.

## H9 — ROOT CAUSE CONFIRMED: paged_fill_cache page_table is TILE at opt1 (workaround gated off). Same class as D46.
Verified in tt-mlir source (HEAD now **detached at `ad2012fdc6`** #9027 — MOVED from session-start `c69b6fa85d`;
libTTMLIRCompiler.so rebuilt 19:25 from it):
- `enabledOpsForWorkaroundWithOptimizer` (TTNNWorkaroundsPatterns.cpp:549-596) is the set that gets operand
  workarounds at optimization_level>=1. It contains Where/Full/Embedding/Scatter/TopK/FlashMlaPrefill/MoE.../
  Conv3d/Sampling/ArgMax — and **NONE of: `PagedFillCacheOp`, `ChunkedScaledDotProductAttentionOp`,
  `PagedScaledDotProductAttentionDecodeOp`, `PagedFlashMultiLatentAttentionDecodeOp`.**
- PagedFillCacheOp HAS a row-major/int32 page_table operand workaround (TTNNWorkaroundsPass.cpp under IR/), but it
  only fires if the op is in the opt>=1 enabled set. It isn't → page_table stays TILE → tt-metal paged_fill_cache
  reads garbage block addresses → hang (the synced-log culprit, H8).
- **The current tt-mlir checkout LOST the prior session's D46** (SDPA index ops in the enabled set) — so BOTH
  paged_fill_cache (prefill KV write) AND chunked SDPA (the read, next op) are unprotected at opt1. This is why the
  tt-mlir tree "shifted under us" (async CCL red herrings, decomposition gone, etc.): the user moved tt-mlir to a
  clean upstream commit that never had the workarounds.

**THE FIX (tt-mlir, same form as D46):** add to `enabledOpsForWorkaroundWithOptimizer`:
`ttnn::PagedFillCacheOp` (immediate culprit), `ttnn::ChunkedScaledDotProductAttentionOp` (chunked read, hangs next),
and for decode: `ttnn::PagedScaledDotProductAttentionDecodeOp`, `ttnn::PagedFlashMultiLatentAttentionDecodeOp`,
and (opt<2) `ttnn::PagedUpdateCacheOp`. Then rebuild libTTMLIRCompiler.so → install. No tt-xla rebuild needed.
- tt-xla-first check: only tt-xla lever is opt_level=0 (getAllTTNNDialectOps → everything workaround'd → row-major)
  — behavior/perf change, not a real fix, but a GREAT zero-rebuild CONFIRMATION: if opt0 does NOT hang at
  paged_fill_cache, the opt>=1 gating theory is confirmed.
- Upstream note: `enabledOpsForWorkaroundWithOptimizer` is missing the paged-cache + SDPA index ops that need
  row-major page_table/index at opt>=1 (worth an upstream PR, not just a local patch).

## H10 — FIX APPLIED: page_table row-major workaround enabled for paged-cache/SDPA ops at opt>=1
- New tt-mlir branch **`ssalice/devstral-pagedcache-sdpa-rowmajor-opt1`** off detached `ad2012fdc6`.
- Commit `6359bfb045`: added `PagedFillCacheOp`, `ChunkedScaledDotProductAttentionOp`,
  `PagedScaledDotProductAttentionDecodeOp`, `PagedFlashMultiLatentAttentionDecodeOp` to
  `enabledOpsForWorkaroundWithOptimizer` (TTNNWorkaroundsPatterns.cpp). (Did NOT add PagedUpdateCacheOp — the decode
  write already has its own `PagedUpdateCacheOpRewritePattern` at opt<2. Follow-up if decode write hangs.)
- Rebuilding `libTTMLIRCompiler.so` (bg bl3bahhu2) → atomic-copy to install/lib. No tt-xla rebuild needed.
- **Mesh-dependence note (answer to user Q):** the tile-page_table bug is mesh-INDEPENDENT (layout decision), but the
  HANG symptom is mesh/shape-dependent — page_table is [32,32]=1 tile on 4x8, [64,32] on 2x4, [128,32] on 1x8; plus
  batch_idx `% local_batch` (DP>1) and block-pool range differ. The mis-read index can land out-of-range (→hang) on
  4x8 but in-range-but-wrong (→silent corruption, no hang) on 1x8/2x4. So a small-mesh "no hang" ≠ bug cleared;
  must check OUTPUT correctness. (Reinforces that the CCL/asymmetric-intermediate hypotheses H5 were async red herrings.)
- **Validation plan:** user runs the target test at opt1 (unchanged config) on a clean device. Expect: gets PAST
  paged_fill_cache in the chunked graph (+ chunked SDPA) → ideally end-to-end. Cheap pre-check: opt0 should already
  not hang there. If it still hangs at paged_fill_cache → the operand workaround isn't forcing row-major (dig into
  the PagedFillCache operand-workaround def); if it hangs LATER (e.g. decode write) → add PagedUpdateCacheOp.

## H10-status — REBUILT + INSTALLED (2026-07-16 22:18)
libTTMLIRCompiler.so rebuilt with commit 6359bfb045 (MLIR_REBUILD_OK), install copy md5-matches build output.
Plugin will load the fixed compiler. Awaiting user's opt1 validation run on a clean device.
(Device was left wedged by the prior synced run — reset the galaxy before running.)

## H11 — ✅ FIX VALIDATED: the page_table row-major workaround RESOLVES the hang
User's opt1 run with the rebuilt compiler got **PAST** the chunked-graph paged_fill_cache + chunked SDPA — the hang
is GONE. It failed only on "stop word ratio too low" = incoherent output from num_hidden_layers=2 (the 2-layer stub),
NOT a hang/crash. ⇒ root cause (H8/H9) confirmed and fixed. The whole CCL-hang investigation (H5) was async red
herrings; the real bug was the tilized page_table into paged_fill_cache at opt1.

## H12 — Productionize for CI (in progress): context-length sweep + full layers + pin the fix
User directives: (1) parametrize test by max_model_len [1024, 4096, 8192, 16384, 32768] (all %256==0 ✓);
(2) num_hidden_layers = FULL (remove the 2-layer override); (3) push the tt-mlir fix branch + repin CMakeLists to it;
(4) keep gpu_memory_utilization constant across context lengths (prefill is chunked). Goal: runnable on BH galaxy CI
as a "run test". Caveat to watch: full 88 layers × constant gmu gives a much smaller KV pool (weights dominate) — may
need gmu tuning at large context; let CI reveal it.

## H13 — CI on BH galaxy: YES, job already exists on main; only blocker is un-merged/un-pushed test
(Full report: `ci_bh_galaxy_report.md`.)
- BH-galaxy runner = **`galaxy-bh`**. The job **`run_vllm_bh_galaxy`** already exists on origin/main
  (`.github/workflows/schedule-nightly-experimental.yml`), runs `./tests/integrations/vllm_plugin` on `galaxy-bh`
  with filter **`nightly and tensor_parallel and bh_galaxy`**. `test_dptp_devstral` ALREADY has those markers
  (bh_galaxy on the mesh_shape param) — no marker change needed. vLLM plugin CI is fully wired (installs vllm-tt
  wheel; HF_TOKEN + shared HF cache set).
- **Blocker:** the branch's changes are UNCOMMITTED (CI runs the committed branch HEAD). Needed on the branch for CI:
  the parametrized test (mine), `third_party/CMakeLists.txt` pin (mine), `fp8_dequant.py` (model load), and
  `vllm_distributed_utils.py` (sharding). Two run paths: (A) push the WIP branch + dispatch `manual-test-single.yml`
  (runs_on=galaxy-bh, dir=the test node id, install_vllm_wheel=true) — bring-up friendly, no main; (B) merge to main
  (the nightly picks it up). Use (A) — instructions say no main / no PRs.
- **Model gate:** Devstral-2-123B is gated (license:other, ~125B fp8). Dropping num_hidden_layers truncates only
  COMPILATION, not the HF DOWNLOAD — so CI's HF token must have license access and the full repo should be pre-cached
  (`HF_HOME=/mnt/dockercache/huggingface`) to avoid a huge cold download. `check_host_memory` is a no-op for Devstral.
- **OPEN — embedding state discrepancy:** working tree has embedding weight `(None,"batch")` (hook apparently removed)
  at vllm_distributed_utils.py:346, but user earlier said they "reverted it." Must confirm which config to commit for
  CI before pushing (affects behavior/CI meaning). Do NOT push until confirmed.

## H14 — CI-ready: branch pushed, pin fixed. Dispatch BLOCKED on gh auth (no token in env).
- tt-xla branch `ssalice/devstral-qwen-wip-07-13-2026` pushed at **b90c10039** (test sweep + CMake pin + fp8 +
  embedding `(None,"batch")` no-hook). tt-mlir fix branch pushed (6359bfb045).
- **CMake pin bug fixed:** the CI grep `grep -oP 'set\(TT_MLIR_VERSION "\K[^"]+'` matched my 2 commented set() lines
  too (3 matches). Removed the commented set() lines → grep now returns exactly `ssalice/devstral-pagedcache-sdpa-
  rowmajor-opt1`. CI (no mlir_override) resolves the fix branch from CMakeLists (call-build-docker.yml:56-60).
- **Dispatch blocked:** `gh` in container (v2.94.0) is NOT authed; no GH_TOKEN/GITHUB_TOKEN/gh-config anywhere.
  Git push works (SSH) but workflow_dispatch needs an API token. Did NOT pull the token from Notion (secret).
- **Turnkey dispatch (user runs once gh is authed, e.g. `gh auth login` or export GH_TOKEN):**
  `gh workflow run manual-test-single.yml --repo tenstorrent/tt-xla --ref ssalice/devstral-qwen-wip-07-13-2026
   -f dir="tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py"
   -f contains="test_dptp_devstral" -f runs_on=galaxy-bh -f install_vllm_wheel=true -f parallel_groups=1`
  (No mlir_override → exercises the branch pin, which is the intent.) Watch: `gh run list --workflow=manual-test-single.yml --branch ssalice/devstral-qwen-wip-07-13-2026`.
- Runs all 5 context lengths of test_dptp_devstral (full 123B). High-context cases may OOM on KV pool — expected signal.

## H15 — CI run 1: 1024 hit the 1h SIGALRM fallback → fixed the hang-guard, re-dispatched
- Run 29542566595 (1024, full model) FAILED: `TimeoutError ... exceeded 3600s` (conftest `_test_timeout` SIGALRM).
  Root: the test isn't in `.test_durations` → 3600s fallback. The JOB budget was fine (240m: any un-recorded test →
  240m via calculate_test_timeout.py), and the `notimeout` marker only affected the JOB budget, NOT the conftest SIGALRM.
- Fix (commit fe8d54f3d): `_test_timeout` now honors `notimeout` (opts out of the SIGALRM, relies on the 240m job
  budget); marked `test_dptp_devstral` `notimeout`. Both files parse.
- **Re-dispatched 1024-only:** run **29548544865** (#4040) on galaxy-bh, commit fe8d54f3d. Dispatched via REST API
  (`curl` POST workflow_dispatch, host GH_TOKEN — no gh, no container). Watching for completion.
- NOTE for later high-context runs: the JOB budget is a flat 240m (4h) for un-recorded tests. If 16k/32k full-depth
  compile exceeds 4h, bump calculate_test_timeout default or record durations. 1024 should be well within 4h.

## H16 — CI 1024 (full depth) result: everything worked EXCEPT device DRAM OOM at warmup
Run 29548544865 (#4040, commit fe8d54f3d): FAILED after 81 min of test (~234 min total incl. build).
- ✅ Build succeeded incl. **tt-xla built against the tt-mlir FIX BRANCH via the CMakeLists pin** ("Override tt-mlir
  SHA" step skipped → confirms the pin drives CI, no override). vLLM wheel built, model downloaded (119.44 GiB fp8),
  weights loaded, KV pool sized. ✅ Timeout fix worked (ran 81 min, no SIGALRM).
- ❌ **Device DRAM OOM** during `capture_model → _precompile_backbone → _run_backbone_dummies` (warmup compile of the
  num_tokens=128 backbone). `TT_FATAL: Out of Memory ... allocate 88 MB ... free: 3 MB`. Device DRAM = 31.88 GiB;
  KV pool = 9.56 GiB (gmu 0.3); full-depth weights ~15 GiB (119 GiB fp8 / TP 8); warmup activations filled the rest.
- **The gmu-constant hypothesis is REFUTED at full depth:** chunked prefill caps *prefill activation* to 128-token
  chunks, but full-depth warmup activations + const-eval + 15 GiB weights + 9.56 GiB KV pool exceed 31.88 GiB. gmu
  DOES matter at full depth.
- **Fix lever (recommended): lower gpu_memory_utilization** to ~0.15–0.2. KV pool at gmu=0.3 is 9.56 GiB / 56,960
  tokens / 55x concurrency @1024 — far more than the SHORT-prompt test needs (~46 tok/user × 128 ≈ 6k tokens). At
  gmu=0.2 → ~6.4 GiB pool (frees ~3.2 GiB); gmu=0.15 → ~4.8 GiB (frees ~4.8 GiB) — plenty of headroom for warmup and
  still ample KV for the workload. Alternatives: lower max_num_seqs (batch), or trim const-eval/dequant staging.
- Per user's conditional ("if 1024 passes I'll trigger higher contexts"): it did NOT pass → do NOT trigger 4k/8k/…
  Surface the OOM + gmu recommendation; the gmu value is the user's call (they'd assumed constant).

## H17 — page_table workaround mechanics + why Gemma [1,4] doesn't hang (agent: pagetable_workaround_and_mesh_analysis.md)
- **tt-mlir HEAD moved to `3abca42835`** (user rebasing; was ad2012fdc6). The 4 ops are ABSENT from the enabled set
  again (rebase dropped them); factories + .td wiring still present. Re-apply = purely additive (4 getOperationName()
  lines to `enabledOpsForWorkaroundWithOptimizer`). **NOTE: my fix branch/pin is off ad2012fdc6 → diverges from the
  new base; the fix must be re-cut on the rebased base and the CMakeLists pin updated.**
- **Q1 — what the workaround does (layout-only except MLA):** default layout tilizes ALL operands unconditionally
  (TTNNLayout.cpp:243-245). Rewriter inserts corrective `to_layout(page_table→RowMajor)` before an op ONLY if it's in
  the enabled set (gate 272-276; opt0=all ops, opt≥1=curated set). Per op:
  - PagedFillCacheOp (:500): page_table + batch_idx → RowMajor.
  - ChunkedScaledDotProductAttentionOp (:1208): page_table + chunk_start_idx → RowMajor.
  - PagedScaledDotProductAttentionDecodeOp (:1157): page_table + cur_pos → RowMajor.
  - PagedFlashMultiLatentAttentionDecodeOp (:1241): page_table → RowMajor **+Int32** (only this forces dtype); cur_pos → RowMajor.
- **Q2 — bug is MESH-INDEPENDENT** (gate keys on op name, not mesh). So Gemma [1,4] "working" is most likely because
  its compiler never had the bug active: **opt0** (getAllTTNNDialectOps → all ops workaround'd) OR a tt-mlir tree that
  still had the ops. CONFIRM coworker's opt_level + tt-mlir SHA — that breaks the tie.
  - If both ran the SAME buggy compiler: symptom is shape/value-driven. Tiled page_table mostly PERMUTES valid
    block-ids → wrong-but-in-range → **silent KV corruption, no fault**. A HANG needs the misread to hit uninitialized
    tile padding (out-of-range NoC) — Devstral's high-seq/many-blocks/large block-ids makes that likely; Gemma's tiny
    page_table (short ctx, max_num_seqs~8, mostly zero-pad → block 0, in-range) corrupts silently. **"No hang" ≠
    correct — Gemma [1,4] is likely producing silently-wrong KV; coworker should check PCC, not just liveness.**
  - Correction to my earlier take: the 1D-vs-2D mesh path (Gemma [1,4] page_table NOT batch-sharded, dp_size=1) is a
    real difference but is NOT what protects Gemma (tilization is unconditional); and `batch_idx % local_batch` is not the cause.

## H18 — paged_fill_cache deep-dive (agent: paged_fill_cache_deepdive.md)
Answers to user's structural questions + the [4,8]-hang vs [2,4]-works discriminator:
- **Q1 operand shards ([4,8], from real IR):** cache %arg8 global [78336,8,32,128] bf16, DP-replicated + TP-sliced
  8→1 → per-device [78336,1,32,128] (**num_blocks=78336**); key per-device [32,1,128,128] (DP batch + TP kv-head);
  page_table [128,32]→[32,32] `("batch",None)` TILE (the bug); batch_idx [128]→[32] `("batch",)` then %local_batch.
  Hanging bucket's per-device page_table is a **FULL [32,32] tile** (not padded [1,32]).
- **Q2 KV-cache-sharing hypothesis: REFUTED.** Single persistent buffer/layer, allocated once, bound before warmup;
  BOTH precompiled graphs (prefix_chunk F/T) share the same bound arg. No aliasing/double-alloc/per-step realloc.
- **Q3 inputs:** [4,8] and [2,4] take the IDENTICAL 2D DATA_TENSOR_PARALLEL path (differ only in dp_size). [1,4] is 1D
  (page_table replicated, no %local_batch).
- **Q4 shard_weights_on_batch_axis: ORTHOGONAL** to the fill path — only sets the DP axis on WEIGHT specs; never read
  by page_table/batch_idx/KV marking. All 3 configs use False.
- **Q5 discriminator:** user's Qwen [2,4] has max_num_seqs=64 → 32 users/dev → **also a full [32,32] tile** (same as
  Devstral) → tile SHAPE is NOT the discriminator. A full-tile swizzle permutes VALID block-ids → in-range → **silent
  corruption for BOTH**; a HANG needs an out-of-range NoC address, a tt-metal-kernel detail NOT provable from this repo.
  Co-leading, repo-unprovable: **(a) block-pool size** (num_blocks 78336 @ gmu0.3 vs tens–hundreds @ gmu0.01 → large
  ids/sentinels more likely to escalate out-of-range) and **(b) arch** (BH NoC faults/timeouts where WH masks).
  Secondary amplifier (d) 1 kv-head/dev @ TP8. **(c) mesh-size/fabric RULED OUT** — sync-after-op localized the hang to
  paged_fill_cache, a device-LOCAL write into the DP-replicated pool, not a collective.
- **Tie-breaker experiment:** Qwen [2,4] on the SAME n300, bump only gmu 0.01→0.3 (arch held constant). Hangs → (a)
  pool size; no hang → (b) arch. AND re-enable coherence/PCC on the Qwen [2,4] + Gemma [1,4] runs — mechanism predicts
  they are SILENTLY CORRUPT (tiled page_table), so their "pass" is unverified liveness, not correctness.
- The validated H11 RowMajor-page_table fix makes ALL configs correct regardless of which axis wins.

## H19 — gmu=0.1 on Wormhole llmbox: NO hang → discriminator leans ARCH, not block-pool
User: gmu=0.3 OOMs; gmu=0.1 on llmbox (Wormhole n300, [2,4]) runs fine (no hang). A 0.1 pool is large (closer to
Devstral's regime), yet Wormhole still doesn't hang → **block-pool-size hypothesis (H18 a) weakened; arch (H18 b)
strengthened**: Blackhole's paged_fill_cache/NoC faults on a tiled page_table where Wormhole tolerates/masks it. User
believes it's arch-specific and reproducible. (Correctness on the Wormhole "pass" still unverified — likely silently
corrupt per H17/H18.)
- User rebased tt-mlir + re-pinned CMakeLists to `ssalice/devstral-07-20-2026` (off 3abca428). The page_table fix must
  live on that branch (verify the 4 ops are in the enabled set there).

## H20 — INVESTIGATE: can we avoid the tt-mlir enabled-set edit with a tt-xla-side fix? (in progress)
User wants: what the paged_fill_cache workaround does, whether it's DP+TP-related, and whether tt-xla can force the
page_table RowMajor (or otherwise dodge the tile issue) WITHOUT adding paged_fill_cache to the tt-mlir workaround set.
Prior-session belief to verify/challenge: "compiler owns the layout (TTNNLayout.cpp:243 unconditionally tilizes all
operands); a runtime to_layout is NOT the fix." Agent investigating tt-xla op registration/lowering (custom_call
operand layouts), compiler-option knobs tt-xla passes, and restructuring the KV write.

## H21 — VERDICT: no tt-xla-only fix; durable path = upstream the enabled-set add to tt-mlir main
Agent (ttxla_side_fix_investigation.md) exhaustively confirmed a tt-xla-side fix is NOT FEASIBLE:
- **Load-bearing obstacle:** `TTNNLayout.cpp:220-254` unconditionally tilizes EVERY op operand (`tiled=true` hardcoded;
  only skips operands already from Broadcast/ToLayout — internal ops the frontend can't emit). So whatever layout the
  page_table has, it's re-tilized; ONLY the gated workaround pass un-tilizes it.
- 3a custom_call operand-layout hint → NOT FEASIBLE (paged_fill_cache emits custom_call with no attrs; conversions 1:1;
  even if set, re-tilized). 3b compiler-option knob → NOT FEASIBLE (ttnn-workaround pass has only 3 opts: enable-layout
  bool / enable-decomposition bool / opt-level int — none take an op list; only opt0 fires it, a global regression).
  3c annotate tensor row-major → NOT FEASIBLE (shouldForceInputRowMajor sets func-ARG layout only, doesn't reach the
  operand; mark_sharding is Shardy-only). 3d restructure write → NOT FEASIBLE (paged cache needs page_table indirection;
  non-paged ops can't express block-indexed write; host pre-convert meaningless).
- **NOT DP+TP-related:** layout/workaround code has ZERO mesh/shard conditionals; opt-level + per-op + Blackhole-arch.
  DP batch-sharding only changes per-device SHAPE; flows through the same tilize→(missing correction) path. Observed on
  DP+TP chunked because that path first exercised paged_fill_cache at opt1 on Blackhole. Locus ≠ cause. Confirms H17.
- Prior-session belief "compiler owns the layout, no frontend hook" = CONFIRMED CORRECT.
- **Per the tt-xla-first rule (try tt-xla, then tt-mlir after 3 fails): all 4 tt-xla avenues exhausted → tt-mlir fix
  justified.** The fix (add 4 ops to `enabledOpsForWorkaroundWithOptimizer`, mirrors existing SDPA/TopK entries) is
  commit **183b2b45d8** on `ssalice/devstral-07-20-2026` (user's rebased branch, which CMakeLists now pins). It is NOT
  an ancestor of origin/main → that's exactly why rebases keep dropping it.
- **DURABLE FIX = upstream it:** land the 4-op addition in tt-mlir `main` via PR (change already written), then bump the
  tt-xla submodule to a commit containing it — rebases carry it automatically. (I will NOT open the PR — user said no PRs;
  recommend the user open it.) Interim-only stopgap if ever needed: opt0 (global perf regression, not recommended).
- Open experiment (H19/H20): pure-TP 1D chunked on Blackhole at opt1 w/o workaround → hangs?=arch-only / runs?=DP-sharded
  page_table needed. And re-check correctness (PCC) on the Wormhole "passing" runs (likely silently corrupt).

## H22 — Full-model OOM (cpu_sampling=False) + OpModel verdict
**Full-depth OOM (`devstral_dptp_test_synced_cpu_false_FULL.log`, gmu=0.3):** same deterministic wall as CI #4040
(byte-identical 88 MB alloc / 3 MB free), during capture_model warmup. Per-device budget: 31.88 GiB − ~14.9 GiB
weights (119.44 GiB fp8 ÷ TP8, **DP-replicated** because shard_weights_on_batch_axis=False) − 9.56 GiB KV pool (gmu0.3)
= ~7.4 GiB for const-eval staging + activations + on-device sampler → OOM. cpu_sampling=False not the cause (same OOM).
- **Dominant consumer = DP-replicated weights.** Levers: (1) **shard_weights_on_batch_axis=True (FSDP)** → 119÷32 =
  ~3.7 GiB/device, frees ~11 GiB (cost: per-layer DP weight all_gathers) — the structural fix for full-depth on this
  mesh. (2) gmu 0.3→0.1 frees ~6.4 GiB (cheapest; try first; prompts short so pool barely used). (3) smaller batch / const_eval.

**OpModel investigation (opmodel_investigation.md) — VERDICT (b): OpModel COMPLEMENTS, does NOT fix the hang.**
- An OpModel is the optimizer's (opt≥1) cost/constraint model, backed by **PROBING tt-metal's `validate`** (reads
  success/error from a graph-capture query) — NOT a static per-operand layout declaration.
- **paged_fill_cache ALREADY HAS a full OpModel** (TTNNOpModelInterface.cpp:3345-3387). Premise moot. It didn't help
  because tt-metal `paged_fill_cache` validate checks page_table **dtype=INT32 + INTERLEAVED** but has **NO
  `page_table.layout()==ROW_MAJOR` check** (paged_fill_cache_device_operation.cpp:42-44) → a TILE page_table PASSES
  validate → optimizer sees Success → nothing forces row-major → tile survives → hang. Only the enabled-set workaround forces it.
- **Positive control:** chunked SDPA's metal validate DOES `TT_FATAL(page_table==ROW_MAJOR)` (sdpa_device_operation.cpp:216),
  so its OpModel (added by #9027) HAS teeth and enforces row-major on its own. **CORRECTION to my earlier take:** chunked
  SDPA re-entered the enabled set only via OUR commit 183b2b45d8 → its whitelist entry is plausibly **redundant** with its
  OpModel (I was wrong that "still whitelisted ⇒ orthogonal"). paged_fill_cache's is NOT redundant (its validate has no teeth).
- OpModel OpConstraints is **L1-only** — does NOT model DRAM → irrelevant to the full-depth OOM.
- **Principled way to retire the workaround:** add a `page_table.layout()==ROW_MAJOR` FATAL to **tt-metal** paged_fill_cache
  validate (mirror SDPA) → gives the existing OpModel teeth. Caveat (unverified): paged_fill_cache is a no-result in-place
  op → RM-propagation stops at it; only OperationValidationAndFallback::tryFallbacks could re-lay the operand — unconfirmed
  it does so for no-result in-place ops. So verify before relying on it.
- **Recommendation: keep the enabled-set entry for PagedFillCacheOp** — it is the actual fix. (Could likely trim the SDPA
  ops from our whitelist add since their OpModel has teeth — verify the two paged-decode ops' validates first.)

## H23 — Const-eval bottleneck REFRAMED: it's bf16→bfp8 conversion round-trips, not fp8 dequant (consteval_dequant_fix_methodologies.md)
- **fp8→bf16 dequant IS host-upfront + fast** (fp8_dequant.py:176 process_weights_after_loading; every const-eval weight input is bf16, zero fp8). NOT the bottleneck. (Corrects my earlier "79-min = fp8 dequant on CPU".)
- **The 79-min = 1062 const-eval `load_cached`, 880 of which convert bf16→bfp8 tiles**, each doing a HOST↔DEVICE round-trip
  (`to_device→to_layout→from_device→typecast-on-HOST→to_device`), ~4.4 s/call × ~1000. **Data MOVEMENT dominates** (user's hunch right).
- The 352 cpu_hoist calls are NOT 88×4 fp8 linears — they're 177 RoPE inv_freq reshapes + 177 fused-weight builds. (Corrects H22.)
- **Both user-proposed fixes are DEAD ENDS:** (a) device bf16→bfp8 typecast EXISTS (unary.cpp bfp8_pack_precise) but tt-mlir
  #8140 deliberately moved it to HOST because the device packer is numerically INACCURATE for weights (+48% TOP1 on gpt-oss-120B
  with host packer). Reusing it reintroduces the accuracy regression. (b) host-upfront bf16→bfp8 impossible — **bfp8 is not a host
  torch dtype**, so host can only reach bf16 (already done). No leverage.
- **TIME ≠ MEMORY.** OOM is TRANSIENT const-eval inflation (device bf16 staging + f32 intermediates on top of resident bfp8), NOT
  steady-state: bfp8 weights 15.2 + KV 9.56 = 24.8 < 31.8 GiB would fit, and KV wasn't even allocated (0 forward ops ran).
- **Testable config levers (no rebuild):** (A) experimental_weight_dtype=bf16 → removes the 880 round-trips (fixes TIME) but full
  depth 28.6+9.56=38.2 > 31.8 → OOM (needs weight sharding to fit); (B) lower gmu → frees staging room; (C) shard_weights_on_batch_axis
  =True (FSDP) → resident weights 15.2→3.7 GiB → room for staging (adds DP weight all_gathers).
- **Agent's recommended DURABLE fix:** keep bfp8; **bound const-eval residency / free transient staging buffers eagerly** during the
  bf16→bfp8 materialization — fixes the OOM AND cuts round-trip traffic. This is a **tt-mlir change** (TTNNWeightDtypeConversion.cpp /
  const-eval scheduling), not a Python edit. Full elimination of the conversion needs tt-metal device-bfp8-packer precision work.

## H24 — gmu=0.1 FIXES the full-model OOM; new blocker = on-device sampler shape bug (cpu_sampling=False)
`devstral_dptp_test_synced_cpu_false_FULL_0_1.log` (full 88 layers, **gmu=0.1**, cpu_sampling=False): **OOM GONE** —
model fit, materialized (~long const-eval), ran the FULL forward, reached sampling (2h41m total). So **lower gmu is a
working memory lever for full depth** (confirms H16/H22). New failure: `IndexError` at `model_runner.py:2079`
`token_id = valid_sampled_token_ids[i][0]` — the on-device DP-sampler (cpu_sampling=False) returns FEWER token rows
than num_reqs on the 2D mesh (combine loop slices `[:num_reqs]` @2010 + `torch.cat` @2018 vs `self.input_batch.num_reqs`
@2036). **Structural shape/gather bug (#4440-class), NOT the chunked-prefill accuracy concern** (bad accuracy → wrong
tokens, not a short list). ⇒ cpu_sampling=False is NOT fully fixed on the 2D DP+TP mesh.
- Quick unblock: `cpu_sampling=True` (CPU sampler path, no DP-combine bug). A robustness guard on [i] would MASK it
  (some reqs silently get no token → incomplete output) — not a real fix. Proper fix = the on-device DP-sampler
  combine/gather (deeper, not a one-liner).
- Chunked-prefill accuracy ↔ fill_cache: separate question, agent `chunked_prefill_accuracy_analysis` running.

## H25 — Chunked-prefill accuracy error is bfp8 KV quantization, NOT fill_cache/tile bug (chunked_prefill_accuracy_analysis.md)
- Expected chunked-prefill accuracy delta = **bfp8 KV-cache quantization**, via asymmetry: cached-prefix chunk path reads
  BOTH prefix + current chunk from the bfp8 cache (attention.py:551-558), vs full prefill attending fresh bf16 K/V
  (575-579). That extra bfp8 round-trip is the whole delta.
- **NOT related to paged_fill_cache logic or the tile bug:** with the row-major fix, fill is lossless-modulo-dtype (pure
  scatter, verified vs CPU ref). Tile bug was gross wrong-block-index corruption, not a bounded PCC delta. Chunking MATH
  is exact (one masked softmax over full range, not lossy online-softmax merge). User's fill_cache hypothesis: refuted.
- **Caveat:** KV bfp8 uses the WEAKER device packer (the one #8140 moved WEIGHTS off of for accuracy); KV forward path
  has no host-pack option → real PCC hit may exceed textbook bfp8. Lever if accuracy matters: experimental_kv_cache_dtype=bf16.
- Short prompts (<128 tok) → chunk_start_idx None → chunked op never fires → chunked-on/off bit-identical prefill; any
  short-prompt error is common-mode decode bfp8, not the multi-chunk path.
- Confirming experiment: long prompt (>128), toggle ONLY KV dtype bfp8→bf16, compare first-token PCC vs bf16 full-seq ref.

## H26 — IndexError root-caused + fixed: on-device sampler batch=32 truncation (index_error_sampler_analysis.md)
- **Root cause:** `sampler.py` pads the batch to 32 via `F.pad(..., 32 - batch)`; for batch>32 that's negative →
  F.pad TRUNCATES to 32. `tt::sampling` kernel requires exactly batch=32. Non-greedy + cpu_sampling=False + decode
  batch>32 → sampler returns 32 token rows → `valid_sampled_token_ids[i][0]` IndexError (model_runner.py:2079). NOT a
  DP-gather bug (logits are already replicated [128,vocab]); NOT the #4440 per-replica thing. Mesh-independent.
- **Reproduced SINGLE-CHIP** (opt-125m, max_num_seqs=64, 64 prompts, temp=0.8/top_p=0.95, cpu_sampling=False, opt0,
  TT_VISIBLE_DEVICES=0): identical IndexError in ~2:44. Confirms mesh-independent (no DP/TP/galaxy needed). Unit test:
  `tests/integrations/vllm_plugin/generative/test_sampler_batch_over_32.py`.
- **Fix (sampler.py, pure Python, no rebuild):** (a) `chunked_topk_candidates` — pad UP to next multiple of 32
  (`padded_batch=max(32, ceil(batch/32)*32)`), never the fixed `32-batch`. (b) `_ttnn_sampling_padded` — tile the batch
  into 32-row groups, pad only the final partial tile, run `tt::sampling` per tile, concat. Fixed tile count at warmup
  (sampler graph compiled at max_num_reqs) → no dynamic-shape recompile. Also incidentally fixes the greedy/random
  `torch.where` shape mismatch (both now [batch]).
- **Caveat:** fixing the SHAPE may expose the separate **#4440 token-soup** correctness issue on the 2D mesh (wrong
  tokens per replica) — distinct from this IndexError; single-chip should be correct.
- **VALIDATED:** post-fix single-chip run PASSED (1 passed, ~64s) — 64 requests each get a token, no IndexError. Pre-fix
  the same test reproduced the IndexError. Fix confirmed end-to-end on hardware.
- **Still to verify:** the DP+TP (2D-mesh) case — the SHAPE fix is mesh-independent so the IndexError should be gone
  there too, but a galaxy run is needed to confirm and to see whether #4440 token-soup then surfaces (separate). Blocked
  on a clean galaxy (no reset permission). Files changed (uncommitted): `sampler.py` (2 fns) + new test
  `test_sampler_batch_over_32.py`.

## H27 — (reserved)

## H18 — Qwen3-32B 8x4 chunked prefill batch-256: 3 tt-xla bugs found+fixed, BOTH shard-weight paths now pass
Date 2026-07-27/28. Build: tt-mlir c91bfb2d (main pin, 2026-07-24), plain — page_table row-major workaround NOT in it.
Test: `test_dptp_qwen` (new), mesh (8,4), batch 256, prefill_chunk_size 128, opt1, trace on, greedy, cpu_sampling=False.

**Test config (validated numbers, do not re-derive):**
- gpu_memory_utilization 0.2 -> 6.38 GiB -> 52,224 tokens. Confirmed on HW.
  Qwen3-32B costs 131,072 B/token (64 layers x 2 x 8 kv heads x 128 head_dim x 1 byte uint8 spec
  dtype from experimental_kv_cache_dtype=bfp_bf8). gmu 0.1 gives 26,112 tokens < 32,768 needed for
  one max-length request at max_model_len=32768 -> check_enough_kv_cache_memory refuses to start.
- max_num_batched_tokens is DERIVED: platform.py overwrites it with prefill_chunk_size * max_num_seqs
  = 128 * 256 = 32768. A user-supplied value is silently discarded.
- notimeout marker REQUIRED (full-depth run took 59 min; conftest SIGALRM fallback is 3600s).
- 32 of 256 prompts carry a ~356-token prefix so the multi-chunk cached-prefix path actually EXECUTES
  (short 13-token prompts never span a 128-token chunk -> chunking compiled but never run).

**BUG 1 — chunked SDPA result users-replicated when shard_weights_on_batch_axis=True.**
Symptom: `'ttir.chunked_scaled_dot_product_attention' op Result shape must match query shape.`
Post-SPMD: query <32x16x128x128> but result <256x16x128x128>. Heads (axis 1) matched, users (axis 0) did not.
Mechanism (traced in IR, not inferred):
  - getChunkedSdpaShardingRule (tt-mlir RegisterCustomShardingRule.cpp:526-573) adds ONE sdy factor,
    for the HEAD dim, via buildHeadShardedCustomCallRule (:506-524, exactly 1 addFactor). Axis 0 gets
    no factor, so query axis0 and result axis0 are independently shardable. The NON-chunked SDPA rule
    at :470-471 DOES add a batch factor — that asymmetry is the tt-mlir-side bug.
  - What made them diverge: partition_row_parallel_linear marks o_proj/down_proj weight
    (batch_axis, "model"). Weight is [out, in] and "model" already owns `in`, so "batch" can only land
    on the OUTPUT dim. Shardy takes the cheap local option and lets the activation inherit it ->
    "batch" sits on activation HIDDEN (640 = 5120/8). A mesh axis can shard only one dim, so users is
    evicted to replicated -> back-propagates 256 users onto the SDPA result.
  - Per-layer oscillation in the True path: QKV/gate_up have "batch" on the CONTRACTING dim, which
    leaves Shardy no choice but to all-gather the weight, so those keep users sharded (4096x5120 =
    32 users x 128 tok). o_proj flips to users-replicated (32768x640) and the MLP stays there;
    an all_to_all + all_gather converts back for the next layer. RMSNorm needs an all_reduce because
    hidden is split. This is NOT classic FSDP (which gathers weights and keeps activations sharded).
FIX (tt-xla): `_pin_users_to_batch_axis_hook` in vllm_distributed_utils.py, registered in
partition_row_parallel_linear when batch_axis is not None. Forces Shardy to all-gather hidden AFTER
the matmul instead of re-laying-out users. Could NOT reuse tt_torch sharding_constraint_hook: it
raises on non-Tensor output and o_proj returns an (output, bias) tuple; and
_normalize_partition_spec_for_rank raises on a rank-mismatched sharded spec, so the spec is built
from ndim. Pinning dim 0 is valid at rank 2 or 3 — users is the major dim either way.

**BUG 2 — query_start_loc_cpu / seq_lens_cpu sized by tokens but indexed by requests.**
Symptom (runtime, AFTER a clean compile): model_runner.py np.cumsum(..., out=...) ->
`ValueError: provided out is the wrong size for the accumulation`. Killed the False path.
Cause: both buffers sized self.max_num_tokens, but indexed query_start_loc_np[1:num_reqs+1] and
seq_lens_np[:num_reqs]. Under chunked prefill max_num_tokens is the per-seq CHUNK budget (paddings
[1,32,64,128] -> 128), the first config where tokens >= reqs (256) stops holding.
FIX: size both max(max_num_tokens, max_num_reqs). Same class as D35's cache_position fix, which
evidently never reached these two buffers. AFFECTS BOTH PATHS.

**BUG 3 — the Bug-1 constraint broke b1-prefill graphs.**
Symptom: `error: Could not compute local sharded shape for result 4` / `Failed to annotate local
shapes for results`. min_num_seqs=1 compiles graphs at num_reqs=1; dim 0 = 1 is not divisible by the
DP axis (8). torch.ops.tt.sharding_constraint has NO divisibility guard (safe_mark_sharding does).
FIX: skip the constraint when tensor.shape[0] % batch_axis_size != 0.

**RESULT (2-layer, batch 256, mesh 8x4, chunked prefill):**
- shard_weights_on_batch_axis=True  -> 1 passed, 920s, 256/256 prompts
- shard_weights_on_batch_axis=False -> 1 passed, 577s, 256/256 prompts
False is ~1.6x faster, consistent with platform.py:185-189 (False = fewer CCLs). False is the default.
CAVEAT: 2 layers => outputs are gibberish by construction and assert_output_coherent is commented out
(known condense() issue), so this proves COMPILES+RUNS, not numerical correctness. Full depth pending.

**NOT the cause (ruled out empirically):**
- partition_column_parallel_linear missing batch_axis: real inconsistency, but Qwen3 has ZERO bare
  ColumnParallelLinear. Handler counts at 2 layers: RowParallelLinear x4, QKVParallelLinear x2,
  MergedColumnParallelLinear x2, VocabParallelEmbedding x1, ParallelLMHead x1, ColumnParallelLinear x0.
  Adding batch_axis there is inert for this model (keep it anyway for models that do use it).
- page_table row-major workaround: absent from this build, but the failure was in SHLO->TTIR
  conversion, upstream of the TTNN workaround pass. Still unvalidated at full depth.

**Open / TODO:**
- Real general fix is tt-mlir: add a users factor to getChunkedSdpaShardingRule mirroring :470-471.
  Caveats: do NOT map K/V dim 0 (it is num_blocks, not users -> would shard the paged cache and cause
  silent corruption); correct mapping is query->0, key/value->kNullDim, page_table->0,
  chunk_start_idx->kNullDim, result->0. buildHeadShardedCustomCallRule cannot express two factors —
  must refactor to OpShardingRuleBuilder (the same refactor the ssalice/paged-ops branch applies to
  the three paged rules; note that branch DELETES buildHeadShardedCustomCallRule).
- PR #9079 (issues #8842/#8843) does NOT fix this and DELETES the verifier that reports it — the bad
  sharding would resurface as an opaque func.return type mismatch. Flag to the PR author.
- tt-mlir branch ssalice/devstral-07-20-2026 force-pushed, rebased onto c91bfb2d (now 1b7437ee27).
  NOT pinned in CMakeLists (pin is plain c91bfb2d).
- vllm_distributed_utils.py partition_vocab_parallel_embedding has TWO comment/code mismatches:
  comment says hidden on "model" but code marks (None,"batch"); comment says the hook uses
  ("batch",None,None) but code passes (None,None,None). Unreviewed.
- test file still has `# TEMP:` markers on shard_weights_on_batch_axis and num_hidden_layers: 2.

## H19 — Context sweep at batch 256 (2-layer, False): 1k and 16k PASS, 32k blocked by a TPU SMEM cap
| max_model_len | result | time |
| 1024  | 1 passed | 577s |
| 16384 | 1 passed | 744s |
| 32768 | 1 failed | assert max_num_scheduled_tokens_all_reqs > 0 (model_runner.py:1301) |

32k COMPILES FINE (200 chunked-SDPA ops, 0 compile errors). It dies on the first scheduling step.

Root cause: TTAttentionBackend.get_max_num_seqs (attention_impls/attention.py:138-140)
    1024*1024 // 2 // cdiv(model_len, page_size) // 4
is a TPU SMEM budget: 512 KB of scalar memory divided by page-table bytes per request (4 B/entry).
model_runner.py:558 uses it as num_reqs_max_model_len = min(that, max_num_reqs), and
_prepare_inputs:1264 truncates the batch to it, forcing the multi-pass (start_index -> end_index) path.

  ctx     pages/req  cap    vs batch 256
  1024    32         4096   fine
  4096    128        1024   fine
  8192    256        512    fine
  16384   512        256    EXACTLY fits (no margin)
  32768   1024       128    truncates -> multi-pass

At 32k the scheduler was also scheduling sparsely (dump: total_num_scheduled_tokens=124, i.e. 31 of
256 reqs at 4 tokens each). Unscheduled reqs contribute 0 via num_scheduled_tokens.get(req_id, 0),
so a 128-slot window containing no scheduled request gives max()==0 and the assert fires.
NOT fully proven: request names encode prompt index, not persistent-batch slot, so the specific
empty window was not confirmed from the dump. Mechanism fits the evidence; last link is inferred.

**Chunked prefill is working as intended — the cap is not about graph size.** Same op, two contexts:
  ctx 1024 : query <1x16x32x128>  page_table <1x32>
  ctx 32768: query <1x16x32x128>  page_table <1x1024>
The COMPUTE tensor is identical; only the page table widens (max_model_len/block_size). The cap is
triggered purely by page-table width against an imaginary 512 KB scalar memory. On TT the page table
is an ordinary DRAM/L1 tensor (256 x 1024 x 4 = 1 MB at 32k/batch256 — trivial). TT's real page_table
constraints are LAYOUT (row-major, 32 B stick alignment), not capacity.
TPU provenance is explicit in the file: the adjacent comment says "TPU has limited SREGs ... VMEM
spill", and get_page_size hardcodes `return 32` with the original TPU sizing logic left dead below.

Two fixes (neither applied yet):
 (1) Make get_max_num_seqs TT-specific (raise/remove the SMEM divisor) -> no truncation, no
     multi-pass, 32k should run. Root cause for this config.
 (2) Guard the multi-pass windowing against windows with no scheduled requests. Correct regardless,
     since sparse scheduling can arise for other reasons.
CAUTION: 16k passing is min(256,256) — zero margin. A smaller block_size or max_num_seqs > 256 tips
it straight into the 32k failure mode.

CAVEAT on all three runs: num_hidden_layers=2 inflates the KV pool 32x (4 KiB/token vs 128 KiB/token)
-> 1,671,168 tokens instead of 52,224. These validate COMPILE + page-table sizing + scheduling, NOT
full-depth KV capacity. At 64 layers, gmu 0.2 gives 52,224 tokens = 1.59x at 32k, 3.2x at 16k.
