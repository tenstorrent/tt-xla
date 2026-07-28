# Decisions log — Devstral-123B DP+TP batch-128 task

_Track forks in decisions here. Re-read on restart and after compaction. Use this to
judge whether a NEW error is genuinely new or a consequence of a PRIOR decision._

Format per entry: date · decision · rationale · advisor? · how to revert.

---

## 2026-06-26

### D1 — Pinned tt-metal to `ssalice/bh_galaxy`
- **Decision:** In `third_party/tt-mlir/src/tt-mlir/third_party/CMakeLists.txt`, set
  `TT_METAL_VERSION` from SHA `13adda80c1…` → branch `ssalice/bh_galaxy`. Committed +
  pushed on tt-mlir branch `ssalice/devstral-wip-06252026-mlir` (commit `9c7248be3`).
- **Rationale:** User instruction. Branch confirmed present on tenstorrent/tt-metal
  (`3113e9138`).
- **Advisor:** no (direct user instruction).
- **Revert:** restore the SHA line and re-commit.
- **Watch:** a moving branch pin means tt-metal can change under us; if a new build
  introduces a regression, suspect a tt-metal `ssalice/bh_galaxy` update.

### D2 — Treat `devstral-123b-galaxy-tp` as THE DP+TP batch-128 target
- **Decision:** Work the failing `test_vllm_tp_benchmark[devstral-123b-galaxy-tp]`
  (in `devstral.log`) as the headline goal, rather than separately pulling in
  `mmanzoor/vllm-data-parallel`.
- **Rationale:** That config already has `enable_data_parallel=True`, `mesh_shape=[4,8]`,
  `batch_size=128`; DP support is already merged into this branch. The given test command
  + log point here. So this test == the batch-128 DP+TP goal.
- **Advisor:** pending (will confirm before committing to a fix approach).
- **Revert:** if the real target turns out to be a different test/branch, re-scope.

### D3 — Confirmed: the 128 height-shard count is the BATCH (users), not context length
- **Decision/finding:** Advisor flagged that `128 <= 120` is ambiguous (batch_size=128 AND
  max_model_len=128). Disambiguation run (`devstral_ctx64_disambig.log`):
  `TT_BENCHMARK_MAX_MODEL_LEN=64`, batch 128, 2 layers → log shows "Using max model len 64"
  but assert is STILL `128 <= 120`. ∴ height-sharded dim = batch/users (128), independent of context.
- **Implication:** Fix direction validated — make the DP batch-split (128→32/replica) propagate
  through the decode attention path so `NLPConcatHeadsDecode` sees 32 users (≤120), not 128.
- **Advisor:** yes (advised the disambiguation; passed the gate).
- **Plan:** locate where the `("batch", …)` sharding is dropped between the marked inputs and
  the decode SDPA op; re-mark at the LEAK point (not just the symptom). Keep in tt-xla.
  Iterate empirically (pure-Python, no rebuild): success signal = assert becomes `32 <= 120`
  (or a different downstream error); still `128` = mark didn't take / leak is further upstream.
- **Guardrail (advisor):** do NOT trigger a tt-mlir rebuild during this phase — it would also
  pull the new tt-metal pin (D1) and confound which change caused any new error.

### D4 — In-graph mark_sharding is IMPOSSIBLE; unblock via chunked decode (tt-xla); true DP is a tt-mlir follow-up
- **Finding (empirical, via agent):** `xs.mark_sharding` cannot be called inside the
  `DYNAMO_TRACE_ONCE`-compiled forward — untraceable pybind init → Dynamo graph-break gb0007.
  So re-marking sharding at the decode op (D3 plan) is categorically dead. Logs:
  `devstral_fix_try1.log`, `try1b`, `try2`. Working tree reverted clean.
- **Finding (advisor + grep):** the decode path is NOT user-sharded today *by construction*:
  kv_cache is batch-*replicated* `(None,"model",None,None)`, and the decode custom op takes no
  `dp_size`/`% local_batch` (unlike sibling `paged_fill_cache`). So each device runs all 128
  users; "batch 32 works" only because 32 ≤ 120. Genuine DP for attention = a real change.
- **Finding (grep):** `stablehlo_custom_call` has no sharding arg; no tt-mlir Shardy rule for the
  decode custom-call. So a build-time sharding attribute (option A) can't partition the op — it
  would reshard around it (stay 128). (A)-alone not viable.
- **Decision:** Unblock batch 128 from **tt-xla** by CHUNKING the decode over the users dim into
  calls of ≤120 users (128 → 2×64), concatenating outputs. Attention is per-user independent;
  slice/concat are trace-safe (no mark_sharding). Each call hands ≤120 users to
  NLPConcatHeadsDecode → assert satisfied. Guard: only chunk when users > 120 (batch 32 and all
  other configs unchanged).
- **Advisor:** yes (advisor proposed chunking as the preferred tt-xla workaround).
- **Honest caveat:** chunking unblocks the assert but does NOT make attention truly DP-sharded,
  so it does not fully "maximize per-device usage" for the attention op. The perf-correct fix is
  a targeted Shardy sharding RULE for the decode custom op (tt-mlir side, NOT hacking
  `deriveCanonicalL1CoreRangeSet` grid math) — deferred follow-up after batch 128 is working.
- **Revert:** remove the chunking branch in `_compute_decode_attention`.

### D5 — Chunked decode VERIFIED; assert fired in 2 ops (cache write + decode); new lm_head CCL error
- **Result:** chunking cleared `128<=120`. Discovery: the assert fires in BOTH
  `PagedUpdateCacheOpRewritePattern` (KV-cache write) and `NLPConcatHeadsDecode` (decode SDPA),
  so BOTH `_handle_paged_attention` (decode branch) and `_compute_decode_attention` were chunked.
  (Scope grew beyond the single method — justified by the actual try1 stack trace.) Verified:
  2-layer run compiles + executes ~11m22s (`devstral_chunk_try2.log`). Diff reviewed = correct,
  guarded by `users<=120` (batch 32 / other models byte-for-byte unchanged). UNCOMMITTED.
- **New error (next fork):** lm_head/sampling postprocess AllGather: `expected TILE, got ROW_MAJOR`
  (tt-mlir runtime `debug_apis.cpp:35`) → `Bad StatusOr access: INTERNAL: 13` at
  `model_runner.py:2599`. Different graph (lm_head, `p_model_lm_head_weight`) + different axis (TP)
  than the chunking; chunked output shape == pre-chunk shape, so downstream graphs unaffected →
  judged INDEPENDENT of the chunking fix, but batch-128-specific (batch 32 passes on CI).
- **Is this caused by a prior decision?** Unlikely the chunking (shape-preserving). NOT the
  tt-metal pin (D1) — no rebuild happened. Most likely a pre-existing batch-128 TP/CCL layout
  issue unmasked by getting 11min further. To investigate next; consult advisor before committing
  to a fix direction (tt-xla workaround first per rules).

### D6 — lm_head AllGather error CONFIRMED independent of chunking; fix attempt = cpu_sampling
- **Operand confirmed from log IR (not inference):** the failing AllGathers are si32 integer
  collectives — `tensor<1x32xsi32> -> 1x128xsi32` and `tensor<32x4xsi32> -> 128x4xsi32`, both
  `cluster_axis=0` = the DP axis (replica groups [[0,8,16,24]...]). These gather sampled
  **token-IDs across the 4 DP replicas** (32/replica → 128). Integer tensors aren't auto-tiled
  to TILE layout before the CCL AllGather → runtime `expected TILE got ROW_MAJOR`.
- **Chunking EXONERATED:** my `torch.cat` output is bf16 [1,128,H,D]; the failing operand is
  si32 [1,32]/[32,4] — different dtype AND shape. Confirmed H1 (advisor-gated). Chunking
  checkpoint committed locally `ff2887db6` (NOT pushed).
- **Fix attempt (advisor's Option A, existing flag):** `TT_BENCHMARK_CPU_SAMPLING=1` →
  `cpu_sampling=True`. Sampling off-device should remove the on-device si32 token AllGather.
  Zero new code. 2-layer run in progress: `devstral_cpusample_try1.log`.
- **Killed:** Option B (cast si32→bf16 around gather) — bf16's 8 mantissa bits cannot represent
  a ~131k-vocab token ID exactly → would silently corrupt tokens. Do NOT use.
- **OPEN QUESTION (do not mark settled):** why batch 32 passes on CI. At batch 32 DP4 the same
  si32 gather exists as [1,8]→[1,32], so "integers don't tile" alone doesn't explain the split.
  Possibly the CI batch-32 config differs (e.g., not DP). Revisit if cpu_sampling doesn't clear it.
- **Diagnostic-only (if cpu_sampling fails):** `enable_trace=False` (TT_BENCH_TRACE=0) tells us
  trace-layout strictness vs a genuine lowering gap — a signal, not a shippable fix.

### D7 — cpu_sampling did NOT fix it; eth FATALs are normal (no reset); running trace=False diagnostic
- **Result:** `TT_BENCHMARK_CPU_SAMPLING=1` 2-layer run (`devstral_cpusample_try1.log`) still
  FAILED with the same si32 AllGather `expected TILE, got ROW_MAJOR` (line 9276). So the failing
  integer DP-gathers are NOT the final sampled tokens — they are intermediate integer tensors
  (`reshape`/`convert`/`subtract` post-opt op names; e.g. likely cache_position / page_table /
  logits_indices style indices) gathered across the DP axis (cluster_axis=0). tt-xla try #1 (of 3) = FAILED.
- **Device-state correction (verified, do NOT assume):** the alarming
  `Chip N logical eth core ... connects to a remote mmio device` TT_FATALs appear in ALL runs
  (44 lines each), including the HEALTHY ones (devstral.log, chunk_try2). → normal galaxy-init
  noise, NOT corruption. No tt-smi reset needed. (Also: I cannot run the user's `exit && ... tt-smi -r`
  reset from inside the Bash tool since it must run on the host; tt-smi is at /home/ssalice/tt-smi
  in-container, uv present — untested. Flag for the user if a real reset is ever needed.)
- **Next:** `TT_BENCHMARK_TRACE=0` diagnostic running (`devstral_notrace_diag.log`). If the TILE
  error VANISHES without trace → it's trace-capture layout strictness (look for a tt-xla
  trace/layout lever = try #2). If it PERSISTS → genuine integer-CCL tilization gap in
  tt-mlir/runtime → likely the 3-tries→tt-mlir escalation path (a rebuild there also pulls the
  ssalice/bh_galaxy tt-metal pin per D1 — flag to user before that commitment).

### D8 — trace=False diagnostic: error is a GENUINE integer-CCL lowering gap (not trace-strictness)
- **Result (`devstral_notrace_diag.log`, 7:18):** with `TT_BENCHMARK_TRACE=0` the run goes MUCH
  further — compiles, loads weights, reaches generation (`Processed prompts: 0/128`) — then hits
  the SAME `expected TILE, got ROW_MAJOR` si32 AllGather during execution (line 76609). So the
  error is NOT trace-capture strictness; it is a genuine integer-tensor-before-CCL tilization gap
  in tt-mlir/runtime. tt-xla try #2 = FAILED (was diagnostic).
- **Status:** 2 tt-xla angles exhausted (cpu_sampling, trace=off). The fix is almost certainly
  tt-mlir/runtime: insert a ROW_MAJOR→TILE `to_layout` before AllGather for integer operands (or
  make the CCL runtime accept ROW_MAJOR int). PENDING advisor: is there a 3rd tt-xla workaround,
  or escalate to tt-mlir? Note: this is a SEPARATE problem from the original (solved) 128>120
  blocker. Flag to user before a tt-mlir rebuild (pulls ssalice/bh_galaxy pin, long build).

### D9 — Decisive batch-32 experiment (gates the tt-mlir-vs-tt-xla decision) [PENDING]
- **Advisor:** my "integers don't tilize before CCL" theory predicts batch 32 ALSO fails (same op,
  `1x8→1x32`/`8x4→32x4`). Must resolve the batch-32-vs-128 contradiction before escalating.
- **Git fact:** commit `68c0e330b` added `devstral-123b-galaxy-tp` BORN as DP+TP **batch 32**
  (`enable_data_parallel=True`, `mesh_shape=[4,8]`). So "batch 32 CI-green" WAS this exact DP+TP
  path — not a non-DP config. ∴ the contradiction is real.
- **Experiment running (`devstral_batch32_check.log`):** `TT_BENCHMARK_BATCH_SIZE=32`, current code,
  2 layers (chunking dormant at 32≤120).
  - If batch 32 ALSO hits si32 TILE error → pre-existing DP+TP/CCL bug, batch-independent;
    "CI-green" suspect → FLAG USER; tt-mlir escalation justified.
  - If batch 32 PASSES → error is genuinely 128-specific → back to tt-xla (shape/threshold at 128),
    NOT yet tt-mlir territory; keep finding tt-xla angles.
- **RESULT: batch 32 PASSES end-to-end** (`devstral_batch32_check.log`: `1 passed in 558s`, 0 TILE
  errors, full generation — 32 reqs, ~113 tok each, TTFT~482ms, ~9.7 tok/s). ∴ the si32 TILE
  error is genuinely **128-specific** (same gather is fine at `1x8→1x32`). NOT a categorical
  integer-CCL bug → **stay in tt-xla**, do NOT escalate to tt-mlir. Chunking doubly exonerated
  (batch 32 passes with chunking dormant = no regression; the si32 gather is in the postprocess
  graph, untouched by the attention-forward chunking). "batch 32 CI-green" CONFIRMED locally.

### D10 — Next: find the tt-xla lever for the 128-specific si32 DP-gather [IN PROGRESS]
- The failing gathers (`1x32→1x128` si32 dim1, `32x4→128x4` si32 dim0, both cluster_axis=0/DP)
  are integer index tensors that work at batch 32 but fail at 128 with `expected TILE, got
  ROW_MAJOR`. Since 32 works, a tt-xla shape/padding/layout lever should exist.

### D11 — PIVOT: failing gathers are page_table/cache_position (NOT sampling); likely caused by MY chunking
- **Corrected identification (agent traced consumers in IR):** the two si32 gathers are
  `page_table` `[num_reqs,num_blocks]=[32,4]→[128,4]` (the FATAL, ROW_MAJOR `#ttnn_layout35`,
  no tile) and `cache_position` `[32]→[1,128]` (survives — a reshape retilizes it to TILE first).
  Both are `("batch",None)`-sharded at `model_runner.py:1257/1258` (real) and `2064/2065` (warmup),
  consumed by **`paged_update_cache`** in the attention/forward graph — NOT the sampling path.
  (Prior "lm_head/sampling" attribution by shape-match was WRONG.)
- **Causality (re-opens D5/D6 exoneration):** my chunking slices `page_table[start:end]` /
  `cache_position[start:end]` on the GLOBAL batch dim (chunks of 64). Each DP device owns 32 users,
  so a 64-chunk crosses DP-shard boundaries → GSPMD must ALL-GATHER the DP-sharded page_table
  (`32x4→128x4`), and that integer gather is ROW_MAJOR → FATAL. This explains "128-specific" as
  really "chunking-active" (batch>120 only; batch 32 = chunking dormant = no forced gather, passes).
  The earlier operand-based exoneration (failing tensor ≠ my bf16 concat) was necessary but
  INSUFFICIENT — chunking's SLICING introduced the gather even though the gathered tensor isn't my output.
- **Causality = INCONCLUSIVE (do not assert "chunking caused"):** si32-gather counts —
  chunked-128 `devstral_chunk_try2.log`=5; passing batch-32=0; unchunked-128 `devstral.log`=0.
  Looks like chunking-caused, BUT `devstral.log` died at the compile-time 128>120 assert with its
  TTNN IR dump incomplete (only 3 all_gather mentions), so it cannot confirm the unchunked-128
  case. Advisor's logic (unchunked-128 op processed full 128 per the assert → already needed the
  full page_table → gather inherent, chunking only UNMASKED it) is plausible and unrefuted. D11's
  "chunking caused" is OVER-stated; treat as open.
- **FIX (causality-independent, agreed w/ advisor): retilize `page_table` to TILE before its
  gather**, mirroring how `cache_position`'s reshape already retilizes (`#ttnn_layout55` = TILE)
  while page_table stays ROW_MAJOR (`#ttnn_layout35`). Works whether chunking caused or unmasked
  the gather. Caution: a bare reshape on `[32,4]` may NOT tilize (trailing dim 4 < 32 tile width)
  → fallback = pad trailing dim to tile-aligned (32). Skip per-device-local chunking (path isn't
  truly DP-sharded). Delegated to fresh agent; log `devstral_retilize_try1.log`.
- **STOP CONDITION (advisor):** if this clears the gather and reveals YET ANOTHER distinct
  128-only failure, checkpoint and HAND BACK to user — do not drill a 3rd layer in deep context.
  Primary win (`ff2887db6`) + batch-32 pass are banked = a real stopping point.

### D12 — Retilize fix WORKS (page_table TILE error gone); re-running to a verdict
- **Edit (uncommitted, working tree):** `model_runner.py` adds `_retilize_page_table()` =
  `reshape(1,-1).reshape(orig)` round-trip (routes page_table through a tile-aligned `1xN`
  intermediate so it stays TILE into the DP gather; mirrors cache_position). Applied at the real
  path (~:1272) AND warmup `_dummy_run` (~:2091) so graphs match; also handles `fill_page_table`.
- **Result:** `devstral_retilize_try1.log` → `expected TILE, got ROW_MAJOR` count = **0** (FIXED).
  Run progressed past the gather all the way to lm_head weight processing (`dot.555`), but the
  sub-agent's process was KILLED when its turn ended → no PASSED/FAILED verdict (log ends mid-exec
  with a leaked-semaphore shutdown warning). NOT a hang/crash — just truncated.
- **Now:** re-running to completion myself (`devstral_retilize_try2.log`, background) to see
  PASS vs next-failure. Apply STOP condition on the result.

### D13 — Retilize fixed page_table; SAME class recurs on cache_position in the DECODE/trace graph
- **`devstral_retilize_try2.log` (ran to completion, 6:54): FAILED.** try1's "count 0" was
  misleading (killed before reaching the failure). The page_table fix HELD (page_table enters the
  trace as `128x4` fine), but the SAME `expected TILE got ROW_MAJOR` now fires on **cache_position**
  in the DECODE/trace graph (`run_and_capture_trace_24_main`→`trace_24_main`): `%29 =
  reshape(%arg1=32xsi32)->1x32` is ROW_MAJOR at RUNTIME, feeding `%30 = all_gather->1x128` which
  expects TILE. (loc `reshape.9`.) Fails at generation (`Processed prompts 0/128`, ~53s in).
- **Notable:** the compiler ANNOTATED this gather input as TILE (`#ttnn_layout55`) but the runtime
  tensor is ROW_MAJOR — a compiler/runtime layout-inference MISMATCH. A tt-xla reshape may not
  fix this (unlike page_table, where it did). If the retilize trick doesn't make cache_position
  TILE at runtime → strong signal the real fix is tt-mlir/runtime (auto-tilize int before CCL, or
  accept ROW_MAJOR) = the escalation path (needs rebuild → pulls bh_galaxy pin → FLAG USER).
- **page_table & cache_position are THE TWO si32 DP-gathered tensors** → fixing cache_position is
  likely the FINAL piece. Delegating that extension (fresh agent, logs `devstral_retilize_try3`).
- **HARD STOP after this:** if cache_position retilize doesn't land (or reveals a genuinely new
  failure), CHECKPOINT + HAND BACK to user (deep context; secondary error is trending tt-mlir).

### D14 — ✅ WIN: cache_position retilize landed; batch 128 PASSES (2-layer); both fixes committed
- **`devstral_retilize_try3.log` (verified independently): `1 passed in 493.86s`, 0 TILE errors,
  128/128 prompts, Benchmark Results printed (14336 samples, TTFT 1461ms).** Attempt 1 sufficed
  (cache_position round-trip mirroring page_table). The tt-mlir-escalation fear (D13) did NOT
  materialize — the tt-xla retilize worked for cache_position too. Stayed entirely in tt-xla.
- **Committed:** `ff2887db6` (chunk decode) + `c6d87ae0a` (retilize page_table & cache_position).
  Both on `ssalice/devstral-wip-06252026`. tt-mlir NOT touched this session (only the earlier
  tt-metal pin D1, no rebuild needed for any of this — all fixes were pure-Python tt-xla).
- **Remaining (finish-the-task):** full 88-layer run (launching now, background); batch-32
  no-regression check (retilize is a DP-path logical no-op, low risk); finalize test config;
  pre-commit; log/branch cleanup; push.

### D15 — Batch-32 regression PASSES; full 88-layer run launched
- **Batch-32 regression (`devstral_batch32_regression.log`, WITH retilize fix): `1 passed in 400s`,
  0 TILE errors, 32/32 prompts.** → retilize does NOT regress batch 32. Both batch sizes confirmed.
- **Test config finalized for full run:** removed `num_hidden_layers=2` (use full model), kept
  batch 128. Test file change is now just `32→128` (the deliverable) — UNCOMMITTED until 88-layer passes.
- **Full 88-layer batch-128 run LAUNCHED** (`devstral_88layer_full.log`, background). Long
  (full 123B load/compile/gen). On PASS → finalize: commit test config, pre-commit, clean up
  scratch logs (`devstral_*try*.log`, `*_disambig`, `*_diag`, `*_check`, `*_regression`, focused
  copy), keep `devstral.log` + the win logs, push tt-xla. On FAIL → diagnose/hand back.

### D18 — BISECTION: gap is UPSTREAM (tt-xla propagation), not tt-mlir — likely a tt-xla-only fix, NO rebuild
- **Verdict (clean-context sub-agent, EVIDENCE-BACKED & VERIFIABLE):** the DP `("batch")` sharding
  never reaches the paged decode op's operands. In the post-Shardy StableHLO dump, the
  `tt.paged_scaled_dot_product_attention_decode` operands are REPLICATED on the batch axis:
  page_table `128x4` and cur_pos `128` carry NO `_axis_0`/`sdy.sharding`; users dim stays 128 (not 32).
- **Airtight discriminator:** `UpdateGlobalToLocalShapes` runs before the dump; the MODEL axis cut
  shapes (KV cache `11097x8x…`→`x1`, query 12 heads/device) but the BATCH axis did NOT (users 128, not
  32). So batch sharding simply isn't present on these tensors → the (correct) tt-mlir rule has nothing
  to propagate.
- **Root cause:** only `input_ids` lands its batch sharding at the SHLO boundary (it goes through
  `model_runner.py:1528 _pin_input_shardings` + `safe_mark_sharding`). The plain
  `xs.mark_sharding(page_table, ("batch",None))` / `(cache_position, ("batch",))` at
  `model_runner.py:1257-1258` (real) and `:2064-2065` (warmup) DO NOT survive torch_xla tracing.
- **Fix hypothesis (TT-XLA only; likely NO tt-mlir rebuild — the rule is already built in 87f63e914):**
  extend `_pin_input_shardings` (called `:2090` warmup / `:1662` exec) to also pin
  page_table + cache_position so their `("batch")` marks persist. The tt-mlir users-factor links
  page_table(0)/cur_pos(0) ↔ query-users(1)/output(1) as one kPassThrough factor, so once an operand is
  sharded, Shardy should back-propagate batch onto query/output. MUST VERIFY that back-propagation.
  The "force MeshWorkloadFactory for DP" half of 87f63e914 touches only paged_update_cache.cpp /
  paged_fill_cache.cpp (NOT decode) — moot until propagation lands; may matter for the cache write next.
- **PRIMARY EVIDENCE to re-verify (do NOT just trust this entry):**
  `modules/irs/shlo_compiler_*g0_g0_1782495535699.mlir` (batch-128 dump). Re-open it, find the
  `tt.paged_scaled_dot_product_attention_decode` custom_call, confirm its operands lack `_axis_0`.
  Repro cmd in the sub-agent report / task.md. Sub-agent left an uncommitted dev tweak
  (`num_hidden_layers=2` in the devstral pytest.param) on branch `ssalice/devstral-dp-sharding-rule`.

### D19 — RE-VERIFIED D18 on primary evidence; mechanism pinned via position_ids control; fix = route index tensors through _pin_input_shardings
- **Primary-evidence re-verification (own eyes, dump `…g0_g0_1782495535699.mlir`):** decode custom_call
  operands confirm batch sharding never lands. Mapped manual_computation in_shardings by tracing arg usage:
  - args_0 (`%arg36`, local **32x1**, in_sharding `[{"_axis_0"},{}]`) → feeds embedding gather (line 54) = **input_ids**, SHARDED ✓ (marked via `_pin_input_shardings`→`safe_mark_sharding`).
  - args_1 (`%arg33`, **128x1**, `[{},{}]`) → feeds rotary (line 75) = **position_ids**, REPLICATED.
  - args_2 (`%arg31`, **128**, `[{}]`) → **cache_position**, REPLICATED.
  - args_3 (`%arg30`, **128x4**, `[{},{}]`) → **page_table**, REPLICATED.
  - Also: line 56 is an **all_gather 32x1x12288→128x1x12288** right after embedding — the sharded input_ids
    is gathered back to 128 because everything downstream (positions/page_table/cur_pos) is replicated. So the
    whole backbone currently runs REPLICATED at 128 users; no real DP. Fixing the marks should keep it at 32.
- **Mechanism discriminator (advisor-proposed, FREE from existing IR):** position_ids is the clean control —
  it is a DIRECT positional arg to the model AND marked by plain `xs.mark_sharding` (model_runner.py:1676), yet
  came back REPLICATED. input_ids differs only in that it ALSO goes through `_pin_input_shardings` (1662/2090).
  ∴ the differentiator is the `_pin` path, NOT direct-arg-vs-side-channel. (Note `safe_mark_sharding` just calls
  `xs.mark_sharding` at its end (vllm_distributed_utils.py:72) — functionally identical when dims divide; and
  `_pin` is *given* position_ids but never marks it, 1551-1554. So the only tensor `_pin` actually pins is input_ids.)
- **FIX (this branch, tt-xla-only, NO rebuild — rule already built in 87f63e914):** give position_ids, page_table,
  cache_position (+fill_page_table) the SAME treatment input_ids has — mark them inside `_pin_input_shardings`
  (right before the model call) in BOTH real (1662) and warmup (2090) paths, mirroring input_ids exactly. Kept the
  pre-existing early marks (1257-1258/2064-2065) — harmless, they run before `_pin` so `_pin` wins.
- **Advisor:** yes (proposed the position_ids discriminator + the route-through-_pin first try).
- **SUCCESS SIGNAL (do NOT stop at `_axis_0` landing):** re-dump IR, confirm decode operands SHRINK —
  query `1x128x12x128`→`1x32`, cur_pos `128`→`32`, page_table `128x4`→`32x4`. Landing the mark without Shardy
  back-prop to query/output = false positive. Verify with a 2-layer (then 4-layer) batch-128 run → hand to user for CI.
- **Revert:** drop the added marks/params in `_pin_input_shardings` and its two call sites.

### D20 — FIX LANDS the decode sharding (✅ core win) but uncovers a NEW prefill-postprocess tile-slice FATAL
- **`devstral_dpfix_try1.log` (2-layer, batch 128): 1 failed in 299s.** The fix WORKED for its target:
  re-dumped IR (`…g0_g0_1782500724128.mlir`, the batch-32 warmup decode graph) shows the decode op now
  BATCH-SHARDED — query `1x8x12x128` (users 8, was 128), page_table `8x4` (was 128x4), cur_pos `8`; KV model-cut
  `11097x1x32x128`. position_ids (args_1) now carries `_axis_0` (local 8x1) — my `_pin` mark survived. page_table/
  cache_position still ENTER replicated (side-channel) but Shardy's decode users-factor rule back-slices them to
  local AT the op (12 `all_slice` composites). Run cleared the old `128>120` assert, ran 5 min executing ttnn ops.
- **NEW FAILURE (compile/precompile, `capture_model`→`_precompile_model_fused`→`_model_prefill`→
  `_model_prefill_postprocess_compiled`, config `{num_tokens:32, prefill}`):**
  `TT_FATAL: Can only slice tilized tensor with height begin index aligned to tiles`
  (`slice_device_operation.cpp:163`: `output_shape[-2] % 32 == 0 && slice_start[-2] % 32 == 0`) →
  `RuntimeError: Bad StatusOr access: INTERNAL: 13`. In const-eval `main_const_eval_9` at `dot.513` (lm_head):
  a `tensor<32xsi32>` index constant is `ttnn.mesh_partition`'d on the DP axis (cluster_axis=0) 32→**8**, then a
  tilized SLICE fails (8 % 32 ≠ 0). The 32 is a 1D index dim (NOT the 128 batch; dummy_positions `[128,32]`→`[32,32]`
  is tile-aligned and fine). At batch 128 local batch=32 is tile-aligned, but this size-32 index→8 is not.
- **Interpretation (tentative):** keeping the lm_head/sampling postprocess DP-sharded forces an si32 index tensor
  to be partitioned to a non-tile-aligned local height, and ttnn's slice requires tile alignment. Open question for
  advisor: narrow the new marks (decode-only / avoid prefill-postprocess), gather hidden states before lm_head in the
  DP prefill, or make the index slice tile-safe. The DECODE win must be preserved.
- **STATE:** edits in working tree (model_runner.py `_pin_input_shardings` + 3 call sites). Old IR dumps moved to
  modules/irs_prev/. Reference batch-128 replicated dump was `…1782495535699.mlir` (now in irs_prev).

### D21 — Two agent reviews CORRECT the diagnosis: culprit is paged_fill_cache batch_idx iota, NOT position_ids/postprocess
- **(advisor tool was unavailable; per user, used TWO independent review sub-agents instead.)**
- **BOTH agents (IR-grounded) agree the FATAL root cause:** the prefill in-graph `batch_idxs = torch.arange(
  key_for_update.shape[0])` (global size, e.g. 32) in `attention.py:496`, fed to `tt.paged_fill_cache`. Because
  key_for_update is now batch-sharded (my backbone fix), tt-mlir's paged_fill_cache rule propagates the DP split
  onto this global-32 si32 iota → in-graph `ttnn.mesh_partition` 32→8 on a TILE layout → slice needs tile-aligned
  height (8 % 32 ≠ 0) → `TT_FATAL slice_device_operation.cpp:163`. It is the prefill KV-write, NOT position_ids or
  the lm_head/sampling postprocess (my D20 framing + the "dot.513" loc were red herrings). The fix didn't add the op;
  it removed the prior accidental safety (when batch sharding didn't survive, K/V into paged_fill_cache was replicated).
- **Agents DISAGREE on the fix (empirical question):**
  - Agent-1: tt-xla-only — build `arange(local_batch)` (size 8) when dp_size>1; a tensor<8> iota has nothing to partition.
  - Agent-2 (read the tt-metal kernel `batch_idx_num_elements == input_batch[local 8]` + the rule code): predicts
    agent-1 alone hits a **sdy factor/shape mismatch** because `RegisterCustomShardingRule.cpp:~791` unconditionally
    binds batch_idx dim0 to the GLOBAL fill_value batch factor (32). Recommends a tt-mlir rule change (leave batch_idx
    REPLICATED) + tt-xla arange(local) + propagate dp_size. Cleaner: replicated arange(local)=[0..7] lands correctly
    on every device per the kernel contract.
  - Agent-2 also found a real latent bug: `_get_dummy_inputs` builds `TTMetadata` WITHOUT `dp_size` (defaults 1), so
    the `% local_batch` rebase never fires in warmup (IR had no remainder op; const was raw 0..31).
- **DECISION (pragmatic, cheapest path to truth):** test the tt-xla-only fix FIRST (no rebuild, ~5min): (1) attention.py
  build `arange(local_batch)` when dp_size>1; (2) `_get_dummy_inputs` pass `dp_size=self.dp_size` to TTMetadata. If it
  compiles past the FATAL → done cheap. If it sdy-errors (agent-2 right) → do the tt-mlir rule change (replicate
  batch_idx) — expect a rebuild (pulls bh_galaxy pin, fine on this track). Deferred: the `_get_dummy_inputs:2434`
  warmup-symmetry _pin change (agent-2: sequence it AFTER the FATAL is cleared, else warmup just hits the same wall).
- **Revert:** restore attention.py arange(global)%local_batch and drop the TTMetadata dp_size kwarg.

### D22 — tt-xla-only iota fix REFUTED (as agent-2 predicted); escalating to the tt-mlir rule change
- **`devstral_dpfix_try2.log` (arange(local) + warmup dp_size): 1 failed in 308s.** It DID reach the
  `{num_tokens:32, prefill}` config (further than try1's framing suggested). Verdict in IR + the error:
  - In-graph IR: `ttnn.mesh_partition(batch_idx) 32→8` and the sdy `all_slice tensor<8xi32> -> tensor<2xi32>`
    (`_axis_0`) — the paged_fill_cache rule STILL partitions batch_idx regardless of iota size (now 8→2).
  - Hard error (line 25259): `'ttir.paged_fill_cache' op Batch index tensor must have dim 0 equal to input
    batch (8), got 2`. EXACTLY agent-2's predicted shape-contract failure.
- **CONCLUSION:** the rule `RegisterCustomShardingRule.cpp` unconditionally binds batch_idx dim0 to the global
  batch factor → it is sliced to garbage size (2). No tt-xla-only fix can prevent this. The correct root-cause fix
  is in tt-mlir: make the paged_fill_cache sharding rule leave batch_idx REPLICATED (don't bind it to the batch
  factor). Then a replicated `arange(local_batch)=[0..7]` lands as 8 on every device = the kernel contract
  (`batch_idx_num_elements == input_batch`). Keep the tt-xla arange(local) + warmup dp_size changes (still correct
  and needed alongside the rule change). Expect a tt-mlir rebuild (pulls bh_galaxy tt-metal pin — fine on this track).
- **Status:** decode-sharding WIN still holds (the original 128>120 blocker is gone); only the prefill KV-write
  batch_idx remains, and it's now precisely localized to one tt-mlir rule.

### D23 — ABLATION: position_ids mark ALONE clears the assert; tilize error is downstream & independent
- **`devstral_dpfix_ablation_posonly.log`: 1 failed in 272s.** Only `position_ids` marked in `_pin`
  (page_table/cache_position/fill_page_table marks commented out; attention.py + dp_size reverted to HEAD).
  Result: `128<=120` assert = **0 occurrences** (GONE); decode op users = **8** (sharded); then the genuine
  tilize FATAL recurs. → CONFIRMED: Fix 1 is literally the ONE `position_ids` safe_mark_sharding line; the other
  three marks were redundant (they don't survive tracing — enter replicated, back-sliced by the decode rule).
- **Failing tensor CONFIRMED = paged_fill_cache `batch_idx`:** const-eval `main_const_eval_9` loc "dot.513":
  `ttnn.constant tensor<32xsi32>` (TILE) → `ttnn.mesh_partition 32→8` → slice tile-align FATAL. The `arange` from
  attention.py.

### D24 — Fix-options synthesis (agent, code-grounded) for the batch_idx rank-1 tilize FATAL
- **Root mechanism (NEW insight):** batch_idx FATALs not just because it's rank-1, but because it's an IN-GRAPH
  CONSTANT (arange → const-eval), so its mesh_partition runs in-graph as a real slice while still TILE-laid. tt-mlir
  ALREADY has an op-level RowMajor workaround for batch_idx (`TTNNWorkaroundsPass.cpp:514-535`) but it applies AT the
  paged_fill_cache op in the main graph — the partition/slice happens UPSTREAM in const-eval, before that layout fires.
  cache_position doesn't FATAL because it's a boundary INPUT (host pre-sharded), not an in-graph const; page_table is
  rank-2+RowMajor (safe).
- **Ranked options:** A/C5 (hoist batch_idx to a host pre-sharded input like cache_position/page_table — #5154):
  WORKS, tt-xla-only, no rebuild, low risk — deletes the in-graph rank-1 slice entirely. **Top pick.**
  C3 (build local arange + explicit `tt.sharding_constraint` replicated to override the rule's kPassThrough): MAYBE —
  the only tt-xla-only "alternative to hoisting"; gated on whether an sdy sharding_constraint overrides the custom
  OpShardingRule. C2 (2-D batch_idx): NO — verifier requires batch_idx 1-D (TTIROps.cpp:5972). C1 (force RowMajor):
  needs tt-mlir (the const-eval partition is the problem, before the op workaround). B (rule replicate batch_idx):
  tt-mlir, maybe. C4 (pad to 32): high blast radius. Plain arange(local) ALONE: already REFUTED (try2, rule re-partitions 8→2).
- **Whack-a-mole?** For CURRENT devstral (flat_model_io=False) batch_idx is the ONLY in-graph rank-1 TILE partition →
  fixing it unblocks THIS config. But flat_model_io=True (Gemma) reshapes input_ids/positions to rank-1 → SAME bug,
  different tensor. Durable cross-config fix = tt-metal **#48303** (legal rank-1 tiled slice) or a blanket tt-mlir
  policy (keep DP-partitioned rank-1 index tensors RowMajor BEFORE mesh_partition lowering).
- **Gemma 5376:** most likely a flattened rank-1 input under flat_model_io (5376 = max_num_reqs × padded_tokens, e.g.
  8×672); SAME bug class, different tensor. Her repro is bf16 → may be flattened embedding/activation, not an si32 index.

### D25 — sharding_constraint experiments on batch_idx (exploring user's "local idx" idea)
- **C3a (`devstral_dpfix_constraint.log`): `arange(local=8)` + replicated `sharding_constraint([{}])`. 1 failed 316s.**
  KEY RESULT: tilize FATAL = **0** (the constraint BROKE the const-eval TILE path → no buggy rank-1 slice), BUT
  `paged_fill_cache: Batch index dim 0 must equal input batch (8), got 2`. So the rule STILL re-shards batch_idx
  (8→2) — the replicated constraint did NOT override the op rule's COUNT. Wrong global size (8) → rule split → 2.
  Insight: the constraint is useful as a CONST-EVAL BARRIER (moves the DP partition out of the TILE const-eval into
  the row-major main graph, killing the tilize), but the global iota must be size 32 so the rule's 32→8 split lands
  the correct local count.
- **C3b (`devstral_dpfix_constraint2.log`, RUNNING): `arange(global=32) % local_batch` + SHARDED
  `sharding_constraint([{"_axis_0"}])`.** Hypothesis: global-32 → rule splits 32→8 (correct count), `% local`
  gives correct local values [0..7]/device, constraint moves partition to row-major main graph (no tilize). If it
  passes the prefill compile → tt-xla-only fix, no rebuild. If not → fall back to A (hoist batch_idx to host
  pre-sharded input) or B (tt-mlir rule replicate batch_idx).

### D26 — ✅ WIN: 2-layer batch-128 PASSES on the root-cause branch (true DP, no workarounds)
- **`devstral_dpfix_constraint3.log`: `1 passed in 350.60s`.** 0 `128<=120` asserts, 0 tilize FATALs,
  Processed prompts 100% (32/32), Benchmark Results printed (Total samples 3584, 10.52 samples/s).
- **The complete fix (3 parts, all tt-xla, NO rebuild):**
  1. `_pin_input_shardings`: mark **position_ids** ("batch", None) — clears the `128<=120` decode assert by
     keeping the backbone DP-sharded (no post-embedding all-gather). Ablation (D23) proved this single mark
     suffices; page_table/cache_position marks were redundant (commented out).
  2. `attention.py` prefill: keep GLOBAL `arange(batch) % local_batch` for batch_idx, then pin it SHARDED on
     the DP axis via `torch.ops.tt.sharding_constraint(..., '#sdy.sharding_per_value<[<@mesh,[{"_axis_0"}]>]>')`.
     The constraint acts as a const-eval barrier → the DP partition lands in the ROW_MAJOR main graph (where the
     op's batch_idx RowMajor workaround applies) instead of the TILE const-eval path → no rank-1 tilize slice
     (#48303). Clears the prefill paged_fill_cache FATAL. (User's "provide local idx" idea, made to work.)
  3. `_get_dummy_inputs`: pass `dp_size=self.dp_size` to warmup TTMetadata (was defaulting to 1, diverging from exec).
  - Plus a bugfix: reverted both `_pin` call sites to 3-arg (the real path's `attn_metadata.page_table` access
    crashed — attn_metadata is a per-layer dict there; surfaced only once we reached real generation).
- **Files:** attention.py (+23), model_runner.py (+32), test config (num_hidden_layers=2 + batch 128, uncommitted).
- **Next:** verify 4-layer; then hand to user for the CI perf run (do NOT run 88 layers locally per user). Then
  clean up dead `_pin` params/comments, finalize config, pre-commit, push.
- **Note:** the durable cross-config fix for the rank-1 tilize remains tt-metal #48303; our constraint is the
  per-tensor tt-xla unblock for this config.

### D27 — CORRECTION: D26's "passing" runs were BATCH 32, not 128 (config positional was batch_size=32)
- **Error caught:** `_tp_config(model, batch_size, ...)` — the devstral config's 2nd positional was `32`, so ALL
  this session's runs (try1…constraint3, 4layer) ran at **batch 32**, not 128 (logs say "0/32 prompts"). The
  base acba65cbf config has batch 32; the `32→128` deliverable change was never made on this branch. I misread the
  batch-32 warmup graphs (local 8 = 32/4) as "warmup uses a smaller batch" and assumed the main path was 128.
- **What's still valid:** the **tilize fix IS validated** — at batch 32 / DP4, batch_idx partitions 32→8 (local 8,
  not tile-aligned), so the rank-1 FATAL DID fire and the sharding_constraint cleared it. **What's NOT validated:**
  the position_ids / `128<=120` fix — at batch 32 the decode op sees ≤32 users (<120), so that assert never fires
  regardless of the fix. D23's "ablation proved position_ids clears the assert" is UNPROVEN (batch 32 passes either way).
- **Fix:** set config to **batch_size=128**, num_hidden_layers=2; re-running (`devstral_dpfix_batch128_real.log`).
  At batch 128 / DP4 → local 32: position_ids fix needed (else decode sees 128 → assert); batch_idx 128→32 still
  rank-1 so constraint still needed (#48303 padded-height, independent of clean split). Awaiting verdict before any
  "passes batch 128" claim.

### D28 — ✅ VALIDATED at TRUE batch 128: 2-layer PASSES (`devstral_dpfix_batch128_real.log`)
- **`1 passed in 560s`, Processed prompts 100% (128/128), 0 `128<=120`, 0 tilize, Total samples 14336, 2.99 samples/s.**
  Config now batch_size=128, num_hidden_layers=2, mesh [4,8]. This is the REAL batch-128 validation (D27 corrected
  the earlier batch-32 confusion).
- Both fixes proven at the target batch: position_ids mark clears the `128<=120` assert (fires at 128 without it);
  sharding_constraint on batch_idx clears the rank-1 tilize at the 128→32 DP partition. Pure tt-xla, no rebuild.
- **4-layer batch-128 also PASSES** (`devstral_dpfix_batch128_4layer.log`: 1 passed in 701s, 128/128 prompts,
  0 assert, 0 tilize, 14336 samples). Fix validated at the target batch across 2 and 4 layers.
- **Batch-32 no-regression CONFIRMED** (`devstral_dpfix_batch32_regression.log`: 1 passed in 378s, 32/32 prompts,
  0 assert, 0 tilize, 3584 samples, 10.54 samples/s) — fix is guarded (dp-mode + dp_size>1), batch 32 unaffected.
- **DONE:** committed `3b4481904` (attention.py + model_runner.py + test batch 32→128, full model) + pushed to
  remote `ssalice/devstral-dp-sharding-rule` (HTTPS+GH_TOKEN; SSH was down). Pre-commit passed. Working tree clean.
- **CI perf run dispatched:** Performance Benchmark run 28391993201 (galaxy-bh, sh-runner=false,
  test_filter=vllm_devstral, ref=ssalice/devstral-dp-sharding-rule). GMU 0.085 may need a TT_BENCHMARK_GMU bump for
  the full 88-layer batch-128 run (D16 KV-exhaustion) — env-only, no code change.

### D29 — New clean branch off mmanzoor/vllm-data-parallel; tt-mlir rebased to ba8aa2aca; tt-metal pin frozen to SHA
- **tt-mlir** `ssalice/devstral-wip-06252026-mlir` rebased onto `ba8aa2aca` (= upstream main w/ tilize fix #8867
  `TTNNLayoutMeshPartitionRewriter` "keep mesh_partition row-major" + users-factor paged rule) + re-applied tt-metal
  pin. Old tip 9c7248be3 in reflog. Force-pushed → tip `fb7150418`.
- **tt-metal pin CHANGED from branch → SHA:** the moving branch `ssalice/bh_galaxy` broke the build (its new tip
  points umd submodule at `ddd931ba…` = "not our ref", unfetchable). Froze TT_METAL_VERSION to
  `3113e9138aa30271b183481f3ed3705d40b9f2eb` (the known-good bh_galaxy commit "added bh galaxy", umd `3d6c4909`,
  already built). Same tt-metal, reproducible — fixes the D1 moving-branch risk. (NOTE: this edit is currently
  UNCOMMITTED in the tt-mlir checkout; commit+repush after build validates.)
- **New tt-xla branch `ssalice/devstral-dp-batch128`** off mmanzoor/vllm-data-parallel (`bfa83b49f`), 6 commits:
  fp8 hook `67b6b5e32`, devstral benchmark+CI (batch 128, repro dropped) `d66c45f70`, GMU 0.085 `8af476ca0`,
  tt-mlir pin `ca9baf70a`, **Consolidate DP input sharding into _pin (commit E)** `1c9d0f38b`, **dp_size plumbing
  (commit F)** `148666d3d`. NO sharding_constraint (the tt-mlir tilize fix replaces it). E addresses PR 4947 review
  comments (position_ids→_pin, safe_mark_sharding, drop redundant page_table/cache_position/warmup marks).
- **Validation PENDING:** rebuilding tt-mlir (fb7150418) now; then 2-layer batch 32 + batch 128 local runs to confirm
  the constraint-free path works on the new tt-mlir. The constraint-based fix is already validated on the OTHER
  branch (ssalice/devstral-dp-sharding-rule, committed+pushed 3b4481904, CI run 28391993201).

### D30 — "Redundant ttnn::ToLayoutOp" root cause: tt-mlir regression (#8867 × #6300), NOT our code
- **Diagnosis (agent, evidence-backed):** the batch-128 compile failure on rebased tt-mlir (ba8aa2aca) is a
  tt-mlir pass regression, independent of our code / the sharding_constraint / tt-metal version.
- **Mechanism:** custom-call.141 at the failure = `ttir.paged_update_cache` (its operand reconciliation), fed by
  `ttir.mesh_partition` (the DP all_slice). #8867 (`5f8c9b10d`, "keep mesh_partition row-major") forces mesh_partition
  I/O row-major in the **TTNNLayout** pass; but #6300 requires KV-cache/paged_update_cache operands **tiled**. Under
  DP the KV cache reaches the paged op THROUGH mesh_partition, so the row-major forcing collides with the tiled
  requirement → the operand reconciliation collapses to an **identity `ttnn.to_layout`**, which
  `TTNNDecomposeLayouts::isCreationValid` (TTNNDecomposeLayouts.cpp:252) rejects with "Redundant ttnn::ToLayoutOp".
  Regression is TIMING: #8867 moved the row-major forcing from TTNNWorkaroundsPass (runs AFTER DecomposeLayouts) into
  TTNNLayout (runs BEFORE), so DecomposeLayouts now sees the conflict. #8867's CI tested only TP-only configs.
- **No upstream fix:** origin/main is 4 commits past ba8aa2aca, none layout-related → bumping the target won't help.
- **Fix options (agent):** (b1) TTNNLayoutMeshPartitionRewriter: skip row-major forcing when mesh_partition feeds a
  KV-cache op (mirror #6300 carve-out); (b2) DecomposeLayouts:252 erase identity ToLayoutOp instead of erroring; OR
  (c) re-add our sharding_constraint (fast, but UNVERIFIED on the new tt-mlir given the timing change). Key files:
  TTNNDecomposeLayouts.cpp:252, TTNNLayout.cpp:495-527 (#8867) & :709-729 (shouldForceInputRowMajor),
  TTNNPipelines.cpp:190,321. Commits: 5f8c9b10d (#8867 regression), bd2477216 (#6300 tiled req), 287300946 (#6888).

### D31 — ✅ Constraint-FREE batch-128 PASSES on new tt-mlir; root cause was commit E dropping the index marks
- **`devstral_dpbatch128_marksfix2.log`: 1 passed in 417s** (128/128 prompts, 0 `128<=120`, 0 tilize, **0 ToLayout**,
  14336 samples). On rebased tt-mlir b7f43c05a (ba8aa2aca + #8867), **NO sharding_constraint**, batch 128 / 2 layers.
- **Root cause of the "Redundant ttnn::ToLayoutOp" on paged_update_cache (D30):** commit E (1c9d0f38b)
  over-eagerly removed `xs.mark_sharding(page_table/cache_position, ("batch",...))`. Without those marks the i32
  index tensors enter REPLICATED and get DP-split by an in-graph mesh_partition; #8867 forces that mesh_partition
  ROW_MAJOR → a no-op to_layout into paged_update_cache → error. (E's "didn't survive tracing" claim was wrong —
  the marks DO boundary-preshard; that was conflated with the separate position_ids issue.)
- **FIX (pure tt-xla, no constraint, no rebuild — Python):** restore the page_table/cache_position marks in ALL
  THREE paths that build them for a TTMetadata: `_prepare_inputs` (real), `_dummy_run` (warmup), AND
  `_get_dummy_inputs` (precompile — the one that actually failed; agent missed it, found empirically: failing graph
  had page_table 128x4 replicated = precompile path). All under the DP/DP+TP guard.
- **Implication:** the sharding_constraint (D21/D25) is NO LONGER NEEDED once tt-mlir has #8867 + this marks fix.
  The constraint-based branch (ssalice/devstral-dp-sharding-rule, 3b4481904) stays valid on the OLD tt-mlir; the
  clean forward path is new-tt-mlir + restored marks + no constraint.
- **TODO:** commit the marks fix on ssalice/devstral-dp-batch128; CORRECT commit E on the contribution branch
  ssalice/vllm-dp-sharding-fixes (it has the same buggy mark removal — should keep position_ids→_pin but NOT drop
  the page_table/cache_position marks).

### D32 — ✅ page_table/cache_position CAN migrate to _pin_input_shardings (earlier "can't" was wrong)
- **Experiment (user-requested):** moved page_table/cache_position/fill_page_table marks INTO _pin_input_shardings
  (passed from all 3 call sites; sample_tokens extracts via `next(iter(attn_metadata.values()))` since _prepare_inputs
  returns the per-layer dict), and COMMENTED OUT the 3 creation-site marks (false-positive guard). Agent double-checked
  the setup (one caveat: a pass could be confounded by paged-op back-prop from the still-active activation marks — but
  the marksfix control proves back-prop alone FAILS, so a pass isolates the _pin marks).
- **Result (`devstral_dpbatch128_pinexp.log`): 1 passed, 128/128, 0 ToLayout/tilize/assert.** IR is authoritative:
  page_table arg enters `local_shape 32x4` with `sdy.sharding [{"_axis_0"},{}]` → the _pin mark **SURVIVED**
  (boundary-presharded). cache_position sharded (32) in the decode/paged_update_cache graph where it matters.
- **So the earlier D19/D20 "marks don't survive in _pin" was wrong** — confounded by the attn_metadata-is-a-dict
  AttributeError bug (which made the _pin call crash, not the mark fail). With the dict handled, they survive.
  → KEEPING the _pin migration (cleaner consolidation, what mmanzoor's L1679 wanted — all input + index sharding in _pin).
- **Validation suite (user-requested), _pin-migration code, no sharding_constraint, new tt-mlir b7f43c05a:**
  2-layer batch 128 ✅ (pinexp.log); 2-layer batch 32 ✅ (pinexp_batch32.log, 32/32); full 88-layer batch 128 RUNNING
  (pinexp_full88.log, GMU bumped to 0.3 via TT_BENCHMARK_GMU for KV headroom per D16).

### D33 — ✅✅ FULL 88-layer batch-128 PASSES (constraint-free, new tt-mlir) — the complete win
- **`devstral_dpbatch128_pinexp_full88.log`: 1 passed in 6263s (1:44:23), 128/128 prompts, all requests gen 113
  tokens, Total samples 14336, 1.98 samples/s, TTFT 2359ms. 0 ToLayout/tilize/assert. NO KV exhaustion**
  (GMU bumped to 0.3 via TT_BENCHMARK_GMU resolved the D16 capacity issue — run completed, no backed-up requests).
- Full validation suite, all green, branch ssalice/devstral-dp-batch128, new tt-mlir b7f43c05a (ba8aa2aca + #8867 +
  tt-metal SHA pin), _pin-migration code, NO sharding_constraint, NO chunking/retilize:
  2-layer b128 ✅ (pinexp.log), 2-layer b32 ✅ (pinexp_batch32.log), FULL 88-layer b128 ✅ (pinexp_full88.log).
- **This is the clean true-DP solution.** The full path: (1) position_ids + page_table + cache_position + fill_page_table
  all pinned in _pin_input_shardings (clears 128<=120 assert AND keeps the paged-op index operands boundary-presharded
  so they avoid the in-graph mesh_partition that #8867 row-major-forces into a paged_update_cache ToLayout error);
  (2) attention.py prefill batch_idx arange(global)%local (NO sharding_constraint — #8867 + boundary marks suffice);
  (3) dp_size plumbed to all 3 TTMetadata build sites. tt-mlir = upstream ba8aa2aca (#8867 tilize fix) + frozen tt-metal pin.
- **TODO (cleanup):** the working tree has the _pin migration + commented-out creation marks (experiment scaffolding) —
  finalize (remove the commented-out blocks), commit on ssalice/devstral-dp-batch128, mirror the corrected E on the
  contribution branch (position_ids→_pin AND index marks→_pin, no removal), GMU decision for the committed config.

### D34 — 2026-06-30 · Committed+pushed the clean devstral solution; dispatched devstral CI; added Qwen3-32B 8x4 test
- **Cleanup DONE & PUSHED:** removed all 4 EXPERIMENT scaffolding blocks, kept the _pin migration. Committed
  `a4e822f6e` ("Pin paged index tensors in _pin_input_shardings; GMU 0.3") on `ssalice/devstral-dp-batch128`,
  pushed via HTTPS+GH_TOKEN (SSH down). Committed GMU bumped 0.085→0.3 (the value the full run needed).
- **Devstral CI dispatched:** Performance Benchmark wf 184422748, run **28422268024**, ref devstral-dp-batch128,
  test_filter=vllm_devstral, runs-on-filter=galaxy-bh, sh-runner=false, mlir_override=ba8aa2aca750... Building OK
  (didn't die on dispatch). WATCH: mlir_override=ba8aa2aca builds tt-metal pin 13adda80c1 (NOT the bh_galaxy SHA
  3113e9138 local validated against) — if CI dies at build/galaxy-init, that pin mismatch is prime suspect, not code.
- **Qwen3-32B 8x4 added** (below devstral in test_vllm_benchmarks.py, id=qwen3-32b-galaxy-tp; + perf-bench-matrix.json
  entry vllm_qwen3_32b_galaxy_tp, galaxy-bh). Config: Qwen/Qwen3-32B, batch 256, mesh [8,4] (DP8×TP4), bfp_bf8,
  enable_const_eval=True, enable_data_parallel=True, shard_weights_on_batch_axis=False, **GMU 0.45**.
  GMU derivation: PRIMARY-SOURCE from existing 8x4 config on `ssalice/qwen3-32b-bhglx-wip-06232026` (batch 32 /
  4 seq-per-rep → GMU 0.0625, has enable_const_eval); my devstral KV model GMU=0.0133+0.00896·(seq/rep) ×1.45
  (Qwen3 per-device KV: 64 layers × 2 kvh/dev vs devstral 88×1) independently reproduces 0.0625 @ 4 seq/rep and
  predicts ~0.43 @ 32 seq/rep (batch 256). 0.45 = small headroom. Same per-replica load as devstral (32 seq/rep).
- **Validating locally** (model_runner _pin fix is pure-Python, reuses last-night's build): batch 32 3-layer →
  batch 256 3-layer → full. TEMP num_hidden_layers=3 in config during bring-up (remove before commit).
- **CI budget:** user said max 8 runs. Used 1 (devstral). Qwen3 CI only if it validates locally.

### D35 — 2026-06-30 · Qwen3 batch-256 hit a PRE-EXISTING per-request-buffer bug; fixed (max(tokens,reqs))
- **Validation results:** Qwen3 8x4 batch-32 3-layer ✅ PASS (qwen3_galaxy_3layer_batch32.log, 724s, 32/32 gen 113 tok,
  0 tilize/assert/redundant). Batch-256 3-layer ❌ FAILED at runtime (compiled fine) →
  `model_runner.py:1244 cache_position_dev.copy_(cache_position): tensor a (256) != b (128)`.
- **Root cause (agent + advisor confirmed, NOT our code):** per-request scratch buffers `seq_lens_cpu` (line 491) and
  `query_start_loc_cpu` (483) were sized by `max_num_tokens` (=num_tokens_paddings[-1]=128, a per-SEQUENCE token width)
  but are indexed by REQUEST count up to `max_num_reqs` (=256). Invariant tokens>=requests breaks when
  batch_size(256) > max_model_len(128). `seq_lens_cpu[:num_reqs_max_model_len=256]` truncates to 128 rows, mismatching
  the correctly-(256,)-sized `_cache_position_dev_max`. Batch 32 passed because max_num_reqs=32 caps both to 32 (≤128).
  Pre-existing since Lewis Panos 2025-09 (`1fad912681`); our commits only touch _pin/sharding, separate from this.
- **Fix (committed-worthy, its own commit — general bug, no-op for devstral where max_num_reqs==max_num_tokens==128):**
  size both buffers to `max(self.max_num_tokens, self.max_num_reqs)` via new `self._max_per_req_or_tok`. Verified
  exhaustive: only seq_lens_cpu + query_start_loc_cpu are per-request max_num_tokens-sized (is_mm_embed/arange_np are
  per-token, left). All consumers slice by num_reqs; buffers are CPU scheduling metadata, NOT compiled-graph inputs
  (graph gets the 256-sized cache_position_dev) → growing is host-side-only, no shape/recompile/KV-corruption risk.
  Also fixes the latent decode crash at line 1173 (query_start_loc cumsum [1:257] needs len≥257; num_reqs hits 256 in decode).
- **Re-running** batch-256 3-layer (qwen3_galaxy_3layer_batch256_fix.log) — exercises prefill AND decode (the real gate).

### D36 — 2026-06-30 · ✅ Qwen3-32B 8x4 batch-256 FULL model PASSES; both commits pushed; qwen3 CI dispatched
- **qwen3_galaxy_3layer_batch256_fix.log: ✅ PASS** (502s, batch 256, 28928 samples, 0 errors) — buffer fix works end-to-end.
- **qwen3_galaxy_full_batch256.log: ✅✅ FULL 64-layer batch-256 PASS** (3323s/0:55:23, batch 256, 28928 samples,
  all 256 reqs gen 113 tok, **peak KV usage 55.8%** = 1/1.79 exactly matching the predicted 1.79x headroom → GMU 0.45
  validated, no exhaustion). const_eval makes compile ~50-60min (one-time) but runs clean.
- **Pushed:** a831a1bf8 (buffer fix), 910106046 (qwen3 config + matrix) on ssalice/devstral-dp-batch128.
- **Qwen3 CI dispatched:** Performance Benchmark wf 184422748, run **28427458622**, ref devstral-dp-batch128,
  test_filter=vllm_qwen3_32b_galaxy_tp (full name → substring-matches ONLY the galaxy entry, not the n300/qb2 ones),
  runs-on-filter=galaxy-bh, sh-runner=false, mlir_override=ba8aa2aca. CI budget: 2 of 8 used (devstral + qwen3).
- **OPEN:** devstral CI run 28422268024 perf job still running (~1h50m); qwen3 CI 28427458622 just started. Watch both;
  debug fails (devstral pin-mismatch tt-metal 13adda80c1 vs bh_galaxy is the prime suspect if devstral CI dies on hw).

### D37 — 2026-06-30 · ✅ Devstral CI GREEN on galaxy-bh; qwen3 CI building
- **Devstral CI 28422268024: conclusion=success.** perf vllm_devstral_123b_galaxy_tp (galaxy-bh) 1h46m38s (≈ local 1:44).
  The tt-metal pin-mismatch worry (13adda80c1 vs bh_galaxy 3113e9138) was UNFOUNDED — mlir_override=ba8aa2aca built +
  ran clean on galaxy-bh hw. **Devstral is DONE: local full-88 PASS + CI green.**
- **Qwen3 CI 28427458622:** still building tt-xla (release) against ba8aa2aca; watcher b05e15bih armed (~3.3h). const_eval
  ~50-60min compile means the perf job will be long (~2h). Watch for KV/const-eval issues; GMU 0.45 validated locally so
  no exhaustion expected.

### D38 — 2026-06-30 · Qwen3 CI FAILED at engine-init (topology mapping, NOT model); re-running to test transient
- **Qwen3 CI 28427458622: FAILURE in 3m48s** (build OK 23m, perf job died at engine-init in 42s, before any compile).
  Root: `TT_FATAL topology_mapper.cpp:518 mapping_result.success — Graph specified in MGD could not fit in the discovered
  physical topology. Inter-mesh mapping failed ... 1 target node(s) not mapped to global node: 0 ... run test_system_health`.
  A mesh/fabric TOPOLOGY-MAPPING failure (logical [8,4] onto physical galaxy) — NOT our model/compile/buffer code.
- **Key asymmetry:** devstral mesh [4,8] PASSED on the SAME mlir_override=ba8aa2aca (tt-metal 13adda80c1). qwen3 mesh [8,4]
  (transpose of physical 4x8) failed. Two hypotheses (advisor-confirmed): (1) TRANSIENT bad galaxy machine / 1 chip didn't
  enumerate ("check connectivity" hint), or (2) DETERMINISTIC: tt-metal 13adda80c1 lacks the bh_galaxy galaxy-mesh-solver
  fixes that the [8,4] transpose needs (local [8,4] worked on bh_galaxy 3113e9138; local log shows galaxy-specific MGD
  handling / issue #43210 / FABRIC_1D auto-override that is branch-specific).
- **Action:** re-ran IDENTICAL (run **28429226667**, watcher bp50635tm) to discriminate. PRE-COMMITTED:
  PASS → transient, done. FAIL same topology_mapper error → deterministic → re-dispatch with
  **mlir_override=b7f43c05a2af007a1d61c90e5b5eca154bbd9514** (my tt-mlir SHA = bh_galaxy tt-metal 3113e9138 + #8867,
  the exact build my local full-run passed on). DO NOT change mesh to [4,8] (user's explicit 8x4 spec; mesh is correct,
  the build's tt-metal is the variable). The branch pin ssalice/devstral-wip-06252026-mlir currently = b7f43c05a (verified).
  CI budget: 3 of 8 used.

### D39 — 2026-06-30 · qwen3 re-run build-flaked (inconclusive); went straight to bh_galaxy tt-mlir fix
- **qwen3 re-run 28429226667: FAILED at BUILD** (not perf) — transient `CPM.cmake` GitHub download flake
  (`file DOWNLOAD cannot compute hash on failed download ... CPM.cmake ... HTTP error` → `Unknown CMake command
  CPMAddPackage`). Unrelated to code/topology; never reached engine-init → INCONCLUSIVE on the topology question.
- **Decision:** rather than burn a run re-confirming ba8aa2aca's topology failure, went straight to the hypothesized fix
  (strong circumstantial evidence it's the tt-metal). Dispatched run **28429663509** with
  **mlir_override=b7f43c05a** (= bh_galaxy tt-metal 3113e9138 + #8867, my exact local-validated build). Watcher b45a0i5ko.
  Fresh build (~25min, CPM flake risk) + perf. CI budget: 4 of 8. If it PASSES → qwen3 green, tt-metal was the issue
  (evidence-backed deviation from user's ba8aa2aca, surface it). If topology fails AGAIN even on bh_galaxy → CI galaxy
  hw / [8,4]-on-CI deeper issue → escalate. If build flakes again → re-dispatch.

### D40 — 2026-06-30 · ✅✅ BOTH CIs GREEN. Task complete. (qwen3 needs bh_galaxy tt-metal; devstral fine on ba8aa2aca)
- **Qwen3 CI 28429663509 (mlir_override=b7f43c05a): conclusion=success.** perf job 58m22s, `1 passed in 3298s`, 256 reqs,
  28928 samples, KV 55.8% (= local exactly), ~337 tok/s, NO topology_mapper error. **CONFIRMS the deterministic
  hypothesis:** the [8,4] transpose mesh requires the bh_galaxy tt-metal (3113e9138 via tt-mlir b7f43c05a). ba8aa2aca's
  tt-metal (13adda80c1) lacks the galaxy mesh-solver fixes → topology map fails for [8,4] (D38). devstral [4,8] = natural
  physical orientation → fine on ba8aa2aca.
- **FINAL STATE — task complete:**
  - Devstral: local full-88 PASS + CI green (28422268024, mlir_override=ba8aa2aca). Pushed a4e822f6e.
  - Qwen3-32B 8x4 batch-256: local full-64 PASS + CI green (28429663509, mlir_override=b7f43c05a). Pushed 910106046
    (config+matrix) + a831a1bf8 (buffer fix). Branch's own tt-mlir pin already = b7f43c05a, so default (no-override)
    qwen3 CI would also work; ONLY mlir_override=ba8aa2aca breaks qwen3's [8,4].
  - CI budget: 4 of 8 used (devstral ×1, qwen3 ×3: fail-topology, build-flake, pass).
- **For user to ratify:** qwen3 CI uses mlir_override=b7f43c05a (NOT the originally-requested ba8aa2aca) — evidence-backed:
  ba8aa2aca's tt-metal can't map [8,4] on the galaxy; mesh stays 8x4 per spec, only the build's tt-metal changed.

### D41 — 2026-06-30 · Added multi-prompt COHERENCE tests (devstral 4x8 + qwen3 8x4); validating qwen3
- User asked for multi-prompt generation tests (the throughput benchmark runs 1 repeated prompt + ignore_eos, no
  coherence check). Added 2 nightly bh_galaxy tests in
  tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py (committed+pushed 157bbbe51):
  test_data_tensor_parallel_generation_devstral_123b (mesh [4,8], 8 prompts=2/replica) and ..._qwen3_32b (mesh [8,4],
  16 prompts=2/replica). Both: llm.chat distinct prompts, temp=0, max_tokens=32, assert_output_coherent, bfp_bf8,
  const_eval, shard_weights_on_batch_axis=False, GMU 0.1, **cpu_sampling=True** (REQUIRED: 2D-mesh on-device sampler =
  token-soup for >1 sample/device, issue #4440 — per existing llmbox tests).
- COHERENCE is the NEW unknown (benchmark never checked output quality). Validating qwen3 locally first
  (qwen3_gen_coherence.log, ~1h full+const_eval) — de-risks both (devstral same DP+TP path). bg task b940zc0m9.
  If coherent → optionally run devstral (123B, longer). If soup → revisit cpu_sampling / per-replica count.

### D42 — 2026-06-30 · Gen tests: 1st run actually WORKED (grep artifact); refined to generate+completion; const_eval REQUIRED
- **FALSE ALARM on "garbage":** qwen3_gen_coherence.log (chat+cpu_sampling=True+greedy+const_eval=True) was actually
  15/16 COHERENT — my `grep '^prompt:'` was line-based and hid the multi-line `<think>\n<reasoning>` (e.g. [1]="Okay,
  the user wants a one-sentence explanation of a neural network..."). Only [0] "Describe Tenstorrent" degenerated
  (greedy + OOD). So the DP+TP galaxy forward + cpu_sampling=True are NUMERICALLY SOUND (advisor caught my grep error).
  (Also confirms the throughput benchmark's output was real coherent text, not masked soup.)
- **Refined both tests** (working tree, supersedes committed 157bbbe51 chat version): chat→**llm.generate + "Continue in
  English:" completion prompts + temp=0.8 + cpu_sampling=True** (mirrors the file's proven llmbox DP+TP coherence tests;
  avoids Qwen3 thinking-mode truncation + the vacuous-pass on short <think> outputs). const_eval=True kept.
- **const_eval is REQUIRED** for this Qwen3-32B galaxy config: const_eval=False fails at `_precompile_backbone`
  torch_xla.sync() → `Bad StatusOr access: INTERNAL: Error code 13` (qwen3_gen_coherence_v2.log, 3m56s). So NO fast
  (const_eval-off) iteration path — must validate with const_eval=True (~50min).
- **Validating** the rewrite fully (qwen3_gen_coherence_v3.log, const_eval=True, bg bbrbh3bj2). Keeping working chat
  version committed until the rewrite passes. If 16/16 coherent → commit rewrite (supersede), optionally run devstral.

### D43 — 2026-06-30 · ✅ Qwen3 gen test 16/16 coherent; refined tests committed+pushed; devstral validating
- **qwen3_gen_coherence_v3.log: PASS (3088s/51:28), 16/16 COHERENT** completions (generate+completion+temp0.8+
  cpu_sampling=True+const_eval=True), e.g. "winter, because I like making snowmen and going sledding", "The Alchemist
  by Paulo Coelho...", "Paris. It is a big, historical and beautiful city in the north of France". No <think>, no soup,
  no vacuous pass. Refined gen-test config VALIDATED for qwen3-32b 8x4.
- **Committed+pushed 67864403f** (generate+completion supersedes the chat version 157bbbe51). Branch tip = 67864403f.
- **Devstral 4x8 coherence validating** (devstral_gen_coherence.log, bg bqxxrapfs, ~1h+ for 123B). Same proven pattern;
  benchmark already proved devstral 4x8 compiles+runs (CI green) so low risk — only confirming output coherence.

### D44 — 2026-06-30 · ✅ Devstral gen test PASS (7/8 coherent); BOTH gen tests done. One triage item for user.
- **devstral_gen_coherence.log: PASS (6253s/1:44:13).** 7/8 completions coherent ("add print statements in key places",
  "a stack follows the LIFO principle", "to have a history of changes in a project", etc.). Prompt **[0]** produced
  token-soup (":ide youth a Tome or{ the the be|...") that PASSED assert_output_coherent because the heuristic is
  weak: _MIN_WORDS=5, _MIN_STOPWORD_RATIO=0.10 — the soup had 12 words / ~0.5 stopword ratio (lots of "the/a/or/be/more")
  so it cleared the 0.10 floor.
- **HONEST framing (corrected from earlier overfit):** this is **observed ONCE on devstral [0]** on the committed config
  (generate+completion+temp0.8). NOT systematic — qwen3 [0] on the same config was coherent ("woods... I am trying to
  figure out"), and the benchmark [0] (" \n! This file contains... GameMaker") was actually coherent too. The earlier
  qwen3-CHAT [0] soup had 3 confounds (greedy + thinking-mode + OOD "Tenstorrent"). With temp=0.8 a single soup draw on
  the first token (flattest dist, bfp8 123B) is the boring explanation. Cause undetermined; confirming systematic-vs-
  variance needs repeated full-model runs (NOT done — user is away + resource-conscious).
- **BOTH gen tests committed+pushed (67864403f) and PASS.** Multi-prompt DP+TP generation works (qwen3 16/16, devstral
  7/8). FOR USER TRIAGE: the coherence heuristic can pass on stopword-heavy soup → a nightly could green-light a bad
  [0]. Options (their call): strengthen assert_output_coherent (shared helper, risky), add a warmup/throwaway prompt,
  or investigate a possible position-0/first-token issue. Did NOT decide solo. TASK otherwise COMPLETE.

### D17 — CORRECTION: tt-mlir branch ALREADY has paged-op batch sharding rules (87f63e914); they're WIP/not landing
- **My earlier "no sharding rule exists" (D4) was WRONG.** `tt-mlir/lib/Dialect/StableHLO/Transforms/
  RegisterCustomShardingRule.cpp` registers Shardy rules for `tt.paged_scaled_dot_product_attention_decode`,
  `tt.paged_update_cache`, `tt.paged_fill_cache` (+chunked_sdpa, flash_mla). The decode rule adds a
  **users factor (kPassThrough)** linking query-users(dim1)/page_table(dim0)/cur_pos(dim0)/output(dim1);
  update_cache links fill_value/update_indices/page_table. Pass IS in pipeline (StableHLOPipelines.cpp:76).
- **Tip commit of the branch the user gave me:** `87f63e914 "[SHLO][TTNN] Add users-dim factor to paged
  ops + force MeshWorkloadFactory for DP (#8781)"` (2026-06-25) — the team's IN-PROGRESS true-DP work.
- **Yet base run still hit 128>120** (decode op saw full 128 users). So the rules exist but the DP
  sharding is NOT actually landing on the paged ops at batch 128. **Reframed task = debug WHY**, not "add rules."
- **Two hypotheses to bisect:** (a) the `("batch")` sharding doesn't PROPAGATE to the decode op's
  operands (upstream gap — check StableHLO IR after Shardy in existing logs, NO rebuild needed); or
  (b) it propagates in StableHLO but the lowering / "force MeshWorkloadFactory for DP" path doesn't
  honor it (downstream). Note "force MeshWorkloadFactory for DP" in the commit title is a clue.
- New branch for this work: `ssalice/devstral-dp-sharding-rule` (off base acba65cbf, no workarounds).
  Workaround branch `ssalice/devstral-wip-06252026` pushed to remote (6e8e33427).

### D16 — 88-layer run: OUR FIXES HELD; failed on KV-cache CAPACITY (config, not our code) → HAND BACK
- **`devstral_88layer_full.log`: FAILED after 2:10:16**, BUT: **0 TILE errors, 0 `128>120` asserts**
  across the full 2-hour full-model run. Our two fixes (chunk decode, retilize) are validated at
  full scale: compiled all 88 layers, ran 2h, generated tokens for 62/128 prompts (48%).
- **Root cause = KV-cache exhaustion at full-model batch 128 (capacity/config):** `GPU KV cache
  usage` climbed 51% → **99.6%** (peak, first hit 09:50:51), requests backed up (Running 128→63,
  Waiting 0→65), generation throughput collapsed (186→4→92 tok/s, one 638s/it spike), then
  EngineDeadError at `worker.py:312`. No alloc/OOM exception text — manifested as scheduling
  collapse under KV saturation. NOT related to chunking/retilize (those don't allocate KV blocks).
- **Why:** `gpu_memory_utilization=0.085` was committed (`422bea04b`) as "~2x KV headroom" sized
  for **batch 32** (8 seq/replica). batch 128 = 32 seq/replica = ~4x KV demand → 0.085 insufficient.
- **Recommended next step (USER decision):** raise `gpu_memory_utilization` for the batch-128
  config (e.g. try ~0.2–0.34) to give KV headroom; re-run full 88-layer (~2h/iter). Env lever:
  `TT_BENCHMARK_GMU=<float>`. This is a deliberate tuning param the user set — handing back rather
  than burning blind 2h runs on a guessed value. CODE fixes are done + committed.
- **STATE:** test config (working tree, UNCOMMITTED) = batch 128, num_hidden_layers removed
  (full model). 2-layer batch-128 PASS + batch-32 no-regression both verified & committed.

---

## 2026-07-13 — SESSION: TT-inference-server integration + CHUNKED PREFILL + torch-2.11/vLLM-0.20.2 env uplift

New branch **`ssalice/devstral-qwen-wip-07-13-2026`** (off origin/main; DP+TP merged via #4947).
Cherry-picked 5 commits (bench, gen tests, fp8 hook, tt-mlir pin, examples) + WIP.
GOAL this session: get CHUNKED PREFILL working for devstral 4x8 / qwen3 8x4 at opt>=1 on BH galaxy.
Companion `report.md` has the live state ladder. User steer: **do NOT chase the "1D-mesh fabric hang"
theory** — CCLs work in other galaxy tests; treat hangs as specific op/lowering issues.

### D45 — 2026-07-13 · fp8 dequant version-skew fix (torch 2.11 / vLLM 0.20.2)
- **File:** `integrations/vllm_plugin/vllm_tt/fp8_dequant.py`. (a) `__init__` sets `activation_quant_key=None`,
  `weight_quant_key=None`, `input_dtype=self.out_dtype`; (b) added `create_weights` override that no-ops
  `init_fp8_linear_kernel` around the base call.
- **Why:** vLLM 0.20.2 base `Fp8LinearMethod.__init__` now sets those 3 attrs (subclass skips super().__init__)
  and moved `init_fp8_linear_kernel()` (KeyError: OOT) into `create_weights`. Crash `AttributeError:
  ...activation_quant_key` at model load. Verified against venv fp8.py:270-386.
- **Result:** model load succeeds. **Revert:** `git checkout -- integrations/vllm_plugin/vllm_tt/fp8_dequant.py` (pure Python).
- **STATUS: KEEP.**

### D46 — 2026-07-13 · tt-mlir: enable SDPA index-tensor row-major workarounds at opt>=1  [REBUILT]
- **File (tt-mlir submodule, UNCOMMITTED):** `.../lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`
  — added `ChunkedScaledDotProductAttentionOp`, `PagedScaledDotProductAttentionDecodeOp`,
  `PagedFlashMultiLatentAttentionDecodeOp` to `enabledOpsForWorkaroundWithOptimizer` (~line 591 after ArgMaxOp).
- **Why:** `chunked_scaled_dot_product_attention` `TT_FATAL: Page table must be row major`. The row-major workaround
  already exists (`TTNNWorkaroundsPass.cpp:1202`, forces RowMajor on page_table + chunk_start_idx) but is gated off
  at opt>=1 because those ops were missing from the enabled set. (opt0 = all ops workaround'd = single-device works.)
- **BUILT:** `ninja -C third_party/tt-mlir/src/tt-mlir/build TTMLIRCompiler` then copied
  `build/lib/libTTMLIRCompiler.so` → `third_party/tt-mlir/install/lib/libTTMLIRCompiler.so` (the copy the plugin
  dlopens; ldd-confirmed). No tt-xla rebuild (function-body change, no ABI change).
- **Result:** row-major FATAL CLEARED — chunked SDPA op now executes on device (verified `devstral_test_trace_on_rerun_v2.log`).
- **Revert:** `git -C third_party/tt-mlir/src/tt-mlir checkout -- <file>` + rebuild TTMLIRCompiler + recopy to install.
- **STATUS: KEEP** (verified). Commit to the tt-mlir branch when the whole path is green.

### D47 — 2026-07-13 · CURRENT BLOCKER: fused ttnn.all_reduce hangs in trace on galaxy (uplift regression)
- **Symptom:** after D45+D46, `test_dptp_devstral[...True...]` runs the chunked SDPA fine, then HANGS at
  `ttnn.end_trace_capture` of a graph containing a fused `ttnn.all_reduce` (cluster_axis=1, TP axis). Device
  timeout, cores 15-3/15-2 (`devstral_test_trace_on_rerun_v2.log:15865`).
- **Evidence (decisive):** pre-uplift PASS run (`devstral_1024_bench_PASS.log`) lowered the SAME TP reduction as
  **DECOMPOSED** `reduce_scatter`+`all_gather` (120 reduce_scatter, 0 all_reduce) and traced fine (48
  end_trace_capture OK, generated output). Post-uplift rerun: **32 `ttnn.all_reduce`, 0 reduce_scatter** → hang.
  ∴ the uplift changed the all-reduce lowering from decomposed→fused, and fused all_reduce doesn't trace on galaxy.
- **NOT a "1D-mesh fabric" problem** (per user; CCLs work in other galaxy tests) — it's a specific CCL-lowering regression.
- **Fix direction (UNDER INVESTIGATION, agent a66ec12fa2952d54f):** restore decomposed reduce_scatter+all_gather
  lowering for all_reduce (a tt-mlir pass/pipeline/flag). Candidate change NOT yet made.
- **Also to try (empirical, cheap):** rerun with `enable_trace=False` (hang is at end_trace_capture; trace-off may
  execute straight through). And a smaller-mesh repro (8 chips via TT_VISIBLE_DEVICES=0,4,8,...,28) to isolate
  chunked-prefill CCL from galaxy scale.

### D48 — 2026-07-13 · Non-blocker artifacts this session
- `tests/integrations/vllm_plugin/generative/test_prefill.py` (NEW): single-device prefill sanity + chunked control
  + DP+TP bisection matrix {chunked on/off}×{trace on/off}. Revert: rm.
- `test_dptp_devstral` rewritten to production config (opt1, trace, bfp8 kv+weights, chunked prefill=128, b1-prefill,
  min_context_len=32, num_hidden_layers=2, prompts×8=128, cpu_sampling=True). Revert: git checkout the test file.
- `TT_INFERENCE_SERVER_INTEGRATION.md`: appended precompiled-graph-inventory appendix.

### D49 — 2026-07-13 · ROOT CAUSE of D47 PINNED: tt-mlir commit 1d91fcf556 (#8961) dropped all_reduce decomposition
- **Cause (agent, evidence-backed):** tt-mlir `1d91fcf556` (#8961, 2026-07-06, "Reshape sub-4D all_reduce to 4D;
  drop reduce_scatter decomposition") — ANCESTOR of tt-mlir HEAD (e26227dd86) — deleted the
  `TTNNAllReduceWorkarounds` pattern (decomposed `ttnn.all_reduce`→`reduce_scatter`+`all_gather`) and its
  `enable-all-reduce-workaround` option. Post-commit, all_reduce lowers FUSED → hangs `end_trace_capture` on galaxy.
  The torch-2.11/vLLM-0.20.2 uplift pulled this commit = the regression.
- **NOT opt-level gated** (unlike D46). It was a pass-composition/pattern-registration knob (default-on bool option).
- **FIX PLAN (Variant A, validated for galaxy by the PASS run):** re-add `TTNNAllReduceWorkarounds` to
  `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` (copy from `git show
  1d91fcf556^:<file>`, class ~line 346 + helper ~471), register after current line ~483 (unconditionally for a
  fast test, or behind a re-added `allReduceWorkaroundEnabled` option). KEEP #8961's `TTNNCollectiveReshapeWorkaround`
  (no conflict — our op is rank-4, reshape pattern is a no-op there). Then rebuild via the D46 loop (TTMLIRCompiler → install).
- **CAVEAT (from OUR prior commit ca019aa82a):** reduce_scatter deadlocks on >2-chip axes — but that was a 1×4 LINEAR
  mesh; the galaxy 2D-mesh PASS run used reduce_scatter on the 8-wide TP axis fine. So Variant A is the right pick for
  galaxy. Variant B (ca019aa82a: all_gather+local-reduce on >2-chip axes) = fallback if reduce_scatter misbehaves.
- **STATUS: fix NOT yet applied** (waiting for the running trace-on test to finish + device-health check before rebuild).
- Refs: #8961 (the drop); tt-metal #13835 (native all_reduce); our branch `ssalice/fix-allreduce-no-reduce-scatter`
  (commits ca019aa82a, e41ad7b387).

### D50 — 2026-07-13 · all_reduce decomposition APPLIED + BUILT + INSTALLED; validating on 8 chips
- **Applied (agent a1f1c…):** re-added `TTNNAllReduceWorkarounds` class + `rewriteAsAllGatherLocalReduce` helper
  VERBATIM from `1d91fcf556^` into `TTNNWorkaroundsPatterns.cpp` (before `GatherSi32Workaround`); registered
  UNCONDITIONALLY inside the `decompositionWorkaroundsEnabled` block (after `LinearOpRewritePattern`). No include/API
  changes needed (ReduceScatterOp/AllGatherOp builders match arg-for-arg). Kept #8961's `TTNNCollectiveReshapeWorkaround`.
- **Built + installed:** `ninja -C build TTMLIRCompiler` linked cleanly (build lib 22:20). Installed via ATOMIC RENAME
  (cp→.new→mv) to `install/lib/libTTMLIRCompiler.so` (22:21, +~10KB, decomposition symbol present) — mmap-safe.
- **WATCH (agent flag):** TWO patterns now match `AllReduceOp` (retained reshape `TTNNCollectiveReshapeWorkaround` +
  re-added decomposition). Our op is rank-4 → reshape is a no-op there, decomposition should win. Novel coexistence —
  validate on HW. Revert-all: `git -C third_party/tt-mlir/src/tt-mlir checkout -- <file>` + rebuild + reinstall.
- **Confirmed hang (this rerun, `devstral_test_trace_on_rerun_v2.log`):** row-major fix held, chunked SDPA executed,
  hang at `end_trace_capture` line ~15864 → TIMEOUT. KILLED the hung pytest+EngineCore (was grinding per-device 60s
  timeouts); all 32 /dev/tenstorrent now FREE. **Fabric may still be wedged → host `tt-smi -glx_reset` may be needed
  (cannot run from container).**
- **VALIDATION RUNNING (fast, 8 chips):** `test_prefill_dptp_chunked_smallmesh -k "chunked-on and trace-on"`
  (Qwen3-0.6B, mesh [2,4], TT_VISIBLE_DEVICES=0,4,8,12,16,20,24,28) → `smallmesh_chunkedon_traceon.log`. PASS =
  all_reduce fix works + hang not scale-specific. HANG = fix failed OR fabric wedged (→ reset + retry to disambiguate).

### D51 — 2026-07-13 · smallmesh carve-out FAILED at init (bad topology, NOT the fix); validating on full galaxy
- **`smallmesh_chunkedon_traceon.log`: 1 failed in 70s at EngineCore init** — `TT_FATAL: Chip N logical eth core
  connects to a remote mmio device`. Root cause: carving NON-CONTIGUOUS chips (0,4,8,…,28) makes the excluded
  chips' eth links "remote" → the 8-chip [2,4] submesh is not a valid connected topology on the galaxy without a
  matching TT_MESH_GRAPH_DESC_PATH. **This is a bad carve-out (the ad-hoc chip IDs), NOT the all_reduce fix** — it
  never reached compile/trace (0 CCL ops, 0 end_trace_capture). Inconclusive for the fix. Don't rabbit-hole 8-chip
  topology; if a small repro is wanted later it needs the galaxy's real connected-submesh chip IDs + mesh descriptor.
- **Positive signal:** device ran init + exited CLEANLY in 70s (no hang) → fabric is RESPONSIVE, likely not wedged.
- **VALIDATING ON THE REAL TARGET (full galaxy):** rerunning `test_dptp_devstral[...True...]` (mesh [4,8], all 32
  chips, valid topology) with the all_reduce fix → `devstral_test_allreduce_fix.log`. This reaches end_trace_capture
  (always has, even post-hang). PASS/continue-past-end_trace_capture = fix WORKS. Hang at end_trace_capture = fix
  didn't take (decomposition not eliminating the fused all_reduce) — since full-galaxy reliably reached that point
  before, a wedge is unlikely to be the cause. Was pre-fix reference: `devstral_test_trace_on_rerun_v2.log`.

### D52 — 2026-07-13 · ✅ all_reduce fix WORKS (end_trace_capture hang GONE); new stall is likely DEVICE WEDGE → needs host reset
- **`devstral_test_allreduce_fix.log` (full galaxy [4,8], 191s): the D47/D49 hang is RESOLVED.**
  `ttnn.all_reduce = 0` → decomposed to `reduce_scatter = 8` + `all_gather = 12` (fix active). Warmup/compile
  progressed PAST the old `end_trace_capture` wall (hung twice pre-fix) into RUNTIME execution. reduce_scatter on the
  8-wide TP axis did NOT deadlock (ca019aa82a fear does not bite on this galaxy 2D mesh — matches PASS run).
- **NEW stall (different failure):** at runtime, `main_const_eval_0` → `ttnn.to_device`(embedding 131072x1536) →
  `TIMEOUT: device timeout IN FETCH QUEUE WAIT`. DISPATCH fetch-queue stall — DIFFERENT timeout type than the pre-fix
  "waiting for physical cores 15-3,15-2" CCL/trace hang. 0 CCLs executed at runtime before it. Basic weight-to-device
  stalling on the fetch queue, on a device that took the earlier hang (killed mid-teardown, NO host reset since) =
  most likely ACCUMULATED DEVICE WEDGE, not a new op bug.
- **BLOCKED ON HOST RESET (user-only):** `cd /data/ssalice/tt-smi && uv run tt-smi -glx_reset` (host; cannot run from
  container). Then rerun `test_dptp_devstral[...True...]` on a CLEAN device. If fetch-queue stall RECURS after fresh
  reset → genuine next issue (const-eval weight load); if GONE → it was wedge and the all_reduce fix fully unblocks
  the trace hang.
- **STATE:** all /dev/tenstorrent free, no orphan procs. all_reduce fix in WORKING TREE (tt-mlir uncommitted;
  libTTMLIRCompiler.so rebuilt+installed). Do NOT revert the all_reduce fix — validated effective on the target hang.

### D53 — 2026-07-13 · Sharding analysis (skill) + embedding DP-round-trip fix APPLIED (user directive: "location matters, nothing in embedding/lm_head")
- **`sharding_analysis.md` written (skill-grounded, cited).** Conclusions: (1) the o_proj/down_proj cross-TP
  all-reduce is FUNDAMENTAL to dense Megatron TP — no sharding eliminates it (it's the correct location); only its
  OP FORM is changeable. (2) Decomposition (shipped, D50/D52) already makes end_trace_capture succeed. (3) Sequence
  parallelism emits the SAME 2 collective types → no help for the hang + reduce_scatter→rms_norm hits a tt-mlir bug
  → REJECTED. (4) column-parallel-2nd-matmul reform → 8× weight blowup → REJECTED.
- **APPLIED — embedding DP round-trip fix (Real Win #1):** `partition_vocab_parallel_embedding`
  (vllm_distributed_utils.py ~347) hook `(None,None,None)` → `("batch",None,None)`. Confirmed via IR (post-fix log):
  the embedding output was being all_gather'd batch 32→128 (cluster_axis=0) + mesh_partition 128→32 every forward —
  a pure DP round-trip from the forced full-replication. New spec keeps batch DP-sharded (matches batch-pinned
  inputs at model_runner.py:1770), leaving only the legit TP hidden gather (cluster_axis=1, 1536→12288). Pure Python,
  NO rebuild. Parses OK. Revert: `git checkout -- integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` (note:
  file also has the user's lm_head revert; revert selectively if needed).
- **Real Win #2 (KV-cache DP-batch shard) — BLOCKED upstream:** paged layout has no batch axis → a DP shard falls on
  blocks/block_size and breaks `ttir.paged_update_cache` (model_runner.py:3446). Deferred (tt-mlir follow-up).
- **lm_head:** column-parallel → final gather for sampling only, NO all-reduce (matches user's expectation). The only
  all-reduce is at o_proj/down_proj (correct).
- **Collective inventory (post-decomp run, devstral_test_allreduce_fix.log):** all_reduce=0; TP(cluster_axis=1)
  reduce_scatter=8 + all_gather=10 (legit o_proj/down_proj + TP hidden gather); DP(cluster_axis=0) all_gather=2 +
  mesh_partition=2 (the embedding round-trip — this fix targets those).
- **NEXT:** after host `tt-smi -glx_reset`, rerun target test with BOTH fixes (decomposition + embedding) → expect
  fewer DP-axis CCLs; check whether the fetch-queue stall (D52) was device-wedge (gone on clean device) or real.

### D54 — 2026-07-14 · CLEAN-DEVICE rerun: embedding fix VALIDATED, but trace-on STILL hangs at end_trace_capture (CCL-in-trace, NOT sharding). D52 was over-claimed.
- **`devstral_test_bothfixes.log` (clean device, both fixes): still HUNG.** Got MUCH further (15758 lines, past
  trace capture) then TIMEOUT "waiting for physical cores 15-3,15-2" at **`end_trace_capture`** of a decode graph
  whose trace contains `%143 reduce_scatter` + `%144 all_gather` (the DECOMPOSED all_reduce). So the decomposition
  changed the CCL form but did NOT fix the trace-capture hang on a clean device.
- **CORRECTION to D52:** D52 read `allreduce_fix.log` (WEDGED device) and concluded "end_trace_capture succeeds / fix
  works" — that was WRONG (wedged run behaved differently, showed a fetch-queue stall). On a CLEAN device the trace
  hang persists. Do not treat the all_reduce decomposition as a complete fix for the hang.
- **embedding fix VALIDATED ✅:** cluster_axis=0 (DP) all_gathers EXECUTED at runtime = **0** (was 2 — the embedding
  round-trip is GONE). 36 reduce_scatter/all_gather executed eagerly FINE. So: (a) embedding DP round-trip fixed,
  (b) TP CCLs work EAGERLY, (c) only TRACE CAPTURE of them hangs.
- **ROOT of remaining blocker = CCL-in-trace hang (uplift regression), NOT sharding, NOT all_reduce form:** capturing
  ANY TP collective (fused all_reduce OR decomposed reduce_scatter+all_gather) in a trace hangs end_trace_capture on
  the uplifted stack. Pre-uplift PASS run had CCLs-in-trace + 48 end_trace_capture that WORKED → it's a regression in
  the uplift (tt-mlir and/or the frozen tt-metal pin's trace+CCL path), same class as the all_reduce-decomposition
  regression. Sharding CANNOT avoid it — the TP reduction CCLs are fundamental and legitimately inside the traced graph.
- **PLAN (two tracks):**
  1. **trace-OFF pragmatic validation** (needs reset): `TT_DEVSTRAL_TRACE=0` (env override just added to
     test_dptp_devstral, enable_trace line). Trace-off = no end_trace_capture; CCLs run eagerly (proven fine); the
     embedding fix removes the DP all_gather that the OLD trace-off run (devstral_test_trace_off.log) hung on. Expect
     chunked prefill to run end-to-end (slower decode, no trace replay) = functional WIN.
  2. **Restore trace-on (production perf):** find/revert the CCL-in-trace uplift regression (deep tt-mlir/tt-metal),
     mirroring the all_reduce-decomposition fix. Higher value (keeps trace perf) but harder.
- **KV-CACHE finding (agent) — answers user's question:** effective DP-sharding of the KV cache (each replica
  writes/reads only its own users via global `batch_idx` arange @ model_runner.py:742-757 + in-graph `% local_batch`
  @ attention.py:492-511; cache replicated + kNullDim by tt-mlir rule design) is ALREADY IMPLEMENTED + working on this
  branch. The user's "arange hoisted out" memory = the old tilize CRASH (now neutralized by #8867 keep-mesh_partition-
  row-major), NOT a semantic loss. So D53's "KV-shard blocked" was about the wrong sense: CORRECTNESS DP-sharding is
  DONE; only PHYSICAL DRAM de-replication is open (cache holds global block pool ×dp_size) — fixable by local-sizing
  the block pool in the vLLM KV-manager (NOT a sharding/tt-mlir change; entry model_runner.py:1021-1069). No tt-mlir
  rule change warranted. Full detail in the agent report / sharding_analysis.md.
- **STATE:** device WEDGED again (this run hung) → needs another host reset. Orphan EngineCore killed. Both fixes
  remain in tree (validated-helpful, do NOT revert). enable_trace + max_model_len now env-overridable in the test.

### D55 — 2026-07-14 · PRECISE root cause of the trace hang: stale CCL GlobalSemaphore on program-cache HIT between the two chunked-path traces (tt-metal #44408/#45332 class; unported to reduce_scatter/all_gather)
- **NOT "CCL-in-trace broken" (too strong).** From `devstral_test_bothfixes.log`: trace_0 (standard prefill,
  prefix_chunk=False) contains 5 all_gather + 4 reduce_scatter and its `end_trace_capture` SUCCEEDS (L8750-9059).
  The hang is trace_1 (the chunked cached-prefix graph, prefix_chunk=True) at `end_trace_capture` (L15745-15746),
  cores 15-3/15-2.
- **Mechanism (leading, strong circumstantial):** trace_1's reduce_scatter/all_gather are BYTE-IDENTICAL to trace_0's
  (same shapes/dtype/cluster_axis → same device program hash; the hash keys on specs NOT buffer addrs —
  reduce_scatter_device_operation.cpp:100-122, all_gather_device_operation.cpp:141-164). So trace_1's CCLs
  DEVICE-PROGRAM-CACHE-HIT trace_0's programs and REUSE trace_0's stale baked GlobalSemaphore L1 addresses → hang.
  This is exactly the #44408 stale-RTA-on-cache-hit bug; #45332 FIXED it for all_to_all_combine (hash input buffer
  addrs to force a miss — all_to_all_combine_device_operation.cpp:117-133) but the fix was NEVER ported to
  reduce_scatter / all_gather. (Caveat: not 100% source-proven that the classic op framework fails to re-patch the
  semaphore on hit — validate on HW.)
- **Trigger = chunked path emits TWO traces of the SAME bucket.** `_chunked_sdpa_active=True` (min_context_len=32 +
  prefill_chunk_size=128, model_runner.py:432-434) → `prefix_chunk_options=[False,True]` (model_runner.py:2565) →
  standard + cached-prefix graphs of the SAME 4096-tok bucket, whose CCLs collide on the program hash. PASS run had
  `_chunked_sdpa_active=False` (min_context_len=128, no chunk) → only [False] → distinct-shaped buckets → distinct
  hashes → fresh semaphores each → no collision → worked. Both runs use decomposed reduce_scatter (D50 fix) + same
  tt-metal tree; the differentiator is the vLLM-0.20.2 config enabling the chunked path, NOT a revertable commit.
- **Options (ranked):**
  1. trace-OFF (pragmatic, being tested; `TT_DEVSTRAL_TRACE=0`) — no traces, no collision. Functional, slower.
  2. **tt-metal ROOT FIX (keeps chunked prefill + trace-on = production):** port #45332 to reduce_scatter &
     all_gather `compute_program_hash` (hash input buffer addrs → trace_1 cache-MISSES → fresh semaphores). ~2 files,
     needs a tt-metal rebuild + HW validation (agent caveat on classic-framework re-patch).
  3. plugin selective-trace (trace prefix_chunk=False, run prefix_chunk=True eagerly) — avoids the colliding 2nd
     trace; medium effort, `enable_trace` is a global tt-mlir pass so needs a plugin branch/separate compile.
  4. config-revert (raise min_context_len / drop chunk) — REJECTED: disables chunked prefill = defeats the goal.
- Citations: bothfixes.log L8417/15111 (both trace-cache MISS), L8750-9059 (trace_0 OK), L15436-15746 (trace_1 hang);
  tt-metal pin 3113e9138 reduce_scatter/all_gather device ops + all_to_all_combine #45332; model_runner.py:432-434,2565.

### D56 — 2026-07-14 · SELF-RESET capability fixed (tt-smi works in-container); user AFK, full autonomy (no PRs)
- **Fixed tt-smi under Python 3.12:** `uv pip install -e .` from /home/ssalice/tt-smi (into the ACTIVE tt-xla venv)
  pulled **tt-umd 0.9.5** (3.12-built), replacing the broken cpython-310 tt_umd. Now `import tt_umd` works,
  `tt-smi` CLI loads with `-glx_reset` / `--galaxy_6u_trays_reset`, and `import vllm_tt.platform` still OK (venv intact).
  ⇒ **I can now reset the galaxy myself:** `tt-smi -glx_reset` from the tt-xla venv (no host/container-exit needed).
  (Note: installed into the tt-xla venv, not tt-smi/.venv — harmless; tt-umd 0.9.5 is an extra pip pkg tt-xla doesn't import.)
- **User AFK — full autonomy granted, ONLY constraint: do NOT open PRs.** Loop now self-sufficient: run → if hang,
  `tt-smi -glx_reset` → fix → rerun.
- **In flight:** trace-off validation `devstral_test_traceoff.log` (TT_DEVSTRAL_TRACE=0, bl0a4gc80). If PASS →
  chunked prefill works functionally → then sweep max_model_len 4096 & 8192 (TT_DEVSTRAL_MAX_MODEL_LEN, 2 layers).
  Then tackle the durable trace-on fix (tt-metal #45332 port to reduce_scatter/all_gather, D55 option 2).

### D57 — 2026-07-14 · trace-off ALSO hangs (device-instability-like); partial self-reset; PIVOT to tt-metal #45332 port (building)
- **`devstral_test_traceoff.log` (TT_DEVSTRAL_TRACE=0): FAILED after 36:48.** 0 end_trace_capture (trace correctly
  off) but 34 TIMEOUTs. Hung at `%19 ttnn.all_gather all_gather_dim=2 cluster_axis=1` (the LEGIT TP hidden-gather
  1536→12288 after embedding) in the num_tokens=64 warmup bucket — AFTER the 128-bucket's structurally-identical CCLs
  ran fine. embedding fix confirmed here too (0 cluster_axis=0 DP all_gathers). A legit TP all_gather that worked
  moments earlier hanging 36min into an eager warmup ⇒ looks like device/fabric INSTABILITY over a long eager run,
  not a clean op bug. (Trace-off warmup runs every bucket eagerly = far more device ops than trace-on = more exposure.)
- **Self-reset is PARTIAL from the container:** `tt-smi -glx_reset` triggers but "POST_RESET failed for device
  /dev/tenstorrent/2..31" — the 6U tray reset needs host/BMC access the container lacks. Chips still enumerate (32,
  Blackhole), NOT bricked, but fabric-clean state uncertain. ⇒ I can TRIGGER but not COMPLETE a galaxy reset; a truly
  clean device may still need the user's HOST `tt-smi -glx_reset`. (tt-smi CLI itself now works in-container — D56.)
- **PIVOT (device-independent, the durable production fix): tt-metal #45332 port** (agent a873…, building). Add input
  buffer-address hashing to `reduce_scatter`/`all_gather` `compute_program_hash` (mirror
  all_to_all_combine_device_operation.cpp:117-133) so the chunked path's 2nd trace cache-MISSES → fresh GlobalSemaphores
  → no end_trace_capture hang, WITH trace-on (production perf). tt-metal source at
  third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal; agent figuring out the incremental build+install
  into third_party/tt-mlir/install/lib (the runtime the plugin dlopens).
- **STATE:** device uncertain (partial reset). Both prior fixes (all_reduce decomp in libTTMLIRCompiler.so + embedding
  sharding) intact + validated-helpful. Next: after tt-metal build, validate trace-ON (short warmup = less fabric
  exposure than trace-off); if device init fails, request host reset from user.

### D58 — 2026-07-14 · tt-metal #45332 port APPLIED + BUILT + INSTALLED; validating trace-ON
- **Applied (agent a873…):** added input-buffer-address hashing to `compute_program_hash` in
  `reduce_scatter_device_operation.cpp` (~L100) and `all_gather_device_operation.cpp` (~L141), mirroring
  all_to_all_combine (#44408/#45332). Appended `tensor_args.input_tensor.buffer()->address()` to each op's
  `hash_operation<T>(...)`. Confirmed create_global_semaphore is inside the factory's cache-miss create() path
  (reduce_scatter_program_factory.cpp:41-49 / all_gather_program_factory.cpp:36-44) → a hash-miss mints FRESH
  semaphores. Only the 2 .cpp files changed (git status clean otherwise).
- **Built + installed:** tt-metal nested build at .../tt-metal/src/tt-metal/build_Release. `ninja -C build_Release
  _ttnncpp.so` recompiled ttnn_op_ccl unity_1/unity_3 + relinked `ttnn/_ttnncpp.so` (exit 0, real recompile ~16s
  ccache). Atomic-copied to install/lib/_ttnncpp.so (the plugin's load-bearing copy via $ORIGIN RUNPATH). Both copies
  sha256 e520037d, ts 05:02:25, 39676088 bytes. (clangd "file not found" diagnostics = IDE noise, missing nested
  include config; ninja build was clean.)
- **Symbol lives in _ttnncpp.so** (not libtt_metal.so). Plugin loads install/lib/_ttnncpp.so.
- **VALIDATING (bbzjmnsdq → devstral_test_ttmetalfix.log):** trace-ON, all 3 fixes active. PASS/past-end_trace_capture
  = the whole chain works (chunked prefill + trace on galaxy). Hang at end_trace_capture = #45332 mechanism wrong /
  didn't take. Init failure = device needs host reset (partial self-reset left it dirty). Device state uncertain (D57).

### D59 — 2026-07-14 · ✅ #45332 fix VALIDATED: end_trace_capture NOW SUCCEEDS (trace-capture hang GONE). Remaining hang = DIRTY DEVICE.
- **`devstral_test_ttmetalfix.log` (trace-ON, all 3 fixes): the end_trace_capture hang is RESOLVED.** Sequence before
  the failure: num_tokens=128 → begin_trace_capture → **end_trace_capture ✓** → begin_trace_capture → **end_trace_capture ✓**
  → Starting main → main_const_eval_0 → to_device → TIMEOUT **"in fetch queue wait"**. So the two chunked-path traces
  CAPTURED SUCCESSFULLY — the exact op (end_trace_capture) that hung on every prior trace-on run (D47/D49/D55/D58 clean
  run). The tt-metal buffer-address hash (D58) forced the 2nd trace's CCLs to cache-miss → fresh GlobalSemaphores → no
  hang. **The core blocker of this whole effort is fixed.**
- **Remaining failure = DIRTY DEVICE, not the fix.** The hang moved to "fetch queue wait" at the FIRST const-eval
  to_device (a trivial weight load) — the device-WEDGE signature (same as D52). This run's device was dirty: 35
  `remote mmio` FATALs + 28 POST_RESET/hang/fabric lines (my partial in-container reset in D57 couldn't clean the 6U
  trays; the prior trace-off hang wedged it). A wedged dispatch can't fetch commands → fetch-queue timeout.
- **To finish: need a CLEAN device.** In-container `tt-smi -glx_reset` is PARTIAL (POST_RESET fails). Retrying + the
  `-glx_reset_auto` variant (bfvh24j6r). If those don't fully clean it → the ONE remaining blocker is the user's HOST
  `tt-smi -glx_reset`. On a clean device, all 3 fixes should carry chunked prefill + trace-on end-to-end (the
  const-eval fetch-queue hang is expected to clear — it's a wedge symptom; if it RECURS on a truly clean device, it's
  a genuine next issue to chase).
- **THE 3 FIXES (all validated-effective, do NOT revert):** (1) all_reduce decomposition (libTTMLIRCompiler.so),
  (2) embedding ("batch",None,None) (vllm_distributed_utils.py), (3) reduce_scatter/all_gather buffer-addr hash
  (_ttnncpp.so). Plus test env knobs (TT_DEVSTRAL_TRACE, TT_DEVSTRAL_MAX_MODEL_LEN).
