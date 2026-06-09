# Debug journey: tt-xla #5116 — batched prefill divergence

Live log of the deep-dive to root-cause and fix the batched-prefill divergence
(identical prompts in one batch produce different greedy tokens). Updated as I go.

## Stack (stock, matches #5116)

- tt-xla: branch `kmabee/issue-5132-...` but **`integrations/vllm_plugin/` is byte-identical to `origin/main`** (only diff vs main is an unrelated new test). pjrt C++ unchanged.
- tt-mlir: **`c5f398432`** (stock; the version main / `8a54156b5` pin). Built `libTTMLIRCompiler.so` md5 `b54ed57c195a24442581ce91e1d723cc`.
- tt-metal: stock pin.
- Only additions: untracked repro scripts `tmp/batch_determinism.py`, `tmp/ttnn_prod_shapes.py`. No chunked-prefill / paged-fill WIP.

Single-device run env: `TT_MESH_GRAPH_DESC_PATH=<install>/tt-metal/.../p150_mesh_graph_descriptor.textproto`, `TT_VISIBLE_DEVICES=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

## What's already established (from prior sessions, re-confirmed on this stock stack)

- **Reproduced here:** batch 32, 1-layer, `max_tokens=1`, greedy, identical prompts, `fp32_dest_acc_en=False` → rows 0–15 emit token 25, rows 16–31 emit token 0 (`min_context_len=32` config). Deterministic.
- **Contiguous-group split; boundary shape-dependent:** k=16 with `min_context_len=32`; k=4 without. Tracks the prefill program-config/grid, not a fixed number.
- **Not the sampler:** reproduces with `cpu_sampling=True` → wrong values are in the logits off the device.
- **`fp32_dest_acc_en=True` fully fixes it.** → bf16 destination-accumulation effect.
- **No single op reproduces it in isolation** even at exact production shapes AND the exact tt-mlir compute config (`hifi4, fp32_dest_acc_en=False`): standalone ttnn matmul / dense SDPA / rms_norm with 32 identical rows are **bit-identical** across rows. ⇒ the divergence is **compositional**.
- All graph intermediates are **interleaved DRAM** (no sharded layouts); matmuls carry only `compute_config = hifi4`, **no explicit program_config** (grid chosen at runtime from shape).

## Logical deduction driving the plan

Composing row-invariant functions is row-invariant. The full graph diverges on
identical input rows, yet every op was row-invariant in isolation on random data.
⇒ **some op is non-row-invariant as executed in-graph** — i.e. on its actual
inputs / memory-config / runtime-selected program-config, which my standalone
reproductions didn't match. The only way to find it is a **per-op activation dump
of the real graph**: run batch-32 with identical rows, dump every op's output,
diff the row-0 block vs the row-16 block, find the first op that diverges.

Leading mechanism hypothesis: a matmul (or its chain) whose output rows are split
across two core groups by the runtime grid, with bf16 K-accumulation rounding
differently per group. Boundary at user 16 = row 512 (= 16 users × 32 tokens),
i.e. the [1024,N] output split in half. To be confirmed/refuted by the dump.

## Tooling note

Runtime per-op callbacks (`debug::Hooks::getPostOperatorCallback`) exist but are
gated behind `TT_RUNTIME_DEBUG==1` (OFF in this build). Plan: add a small
env-gated dump directly in the runtime op-execution path and rebuild (incremental
~1 min), OR build ttrt with the golden/intermediate-dump feature.

## Log

- (start) Stock stack confirmed; bug reproduced; IR dumped via `TTXLA_LOGGER_LEVEL=DEBUG` + `~/scripts/extract_mlir_graphs.py` (graphs in `generated/`, `/tmp/graph_*`). Next: per-op activation dump.
- Patched `runtime/lib/ttnn/program_executor.cpp` with an env-gated (`TT_DUMP_INTERMEDIATES`) per-op check: after each op, pull host output, compare user-0 block vs users {1,4,8,15,16,17,31} (handles `[32,…]` and flattened `[1024,…]`), log `[ROWDIV]`. Incremental runtime rebuild ~1 min.

## ROOT CAUSE FOUND (per-op dump, decisive)

Per-op dump of the batch-32 / 1-layer / `min_context_len=32` run:
- **All 44 `[1024,…]` decoder ops are row-invariant** (matmuls, SDPA, rms_norm → `differ_from_user0: none`). The decoder forward is NOT the bug.
- **First op to diverge (only users ≥16) = `gather.34`** (`ttnn.embedding` over a `[1024,3072]` table) — the per-user **last-token gather** feeding the LM head (`dot.39`) → `argmax` (`reduce.101`).

The SHLO (graph 9) shows a 2-D-indexed gather of each user's last-token hidden:
```
%c_2 = [[0],[1],...,[31]]                    # user indices [32,1]
%5 = concatenate(%c_2, %4, dim=1) : [32,2]   # start indices [user, token_offset]
%6 = stablehlo.gather(%arg1 [32,32,3072], %5) -> [32,3072]
```
tt-mlir lowers this gather to a flat linear index `user*32 + offset` computed by a
**`ttnn.matmul` `[32,2]@[[32],[1]]` with `hifi4` / no `fp32_dest_acc_en`** (bf16
destination accumulation), then an `embedding` lookup.

**bf16 can only represent integers exactly up to 256; step 2 in [256,512); step 4
in [512,1024).** The per-user flat index `u*32+14 ≡ 2 (mod 4)`, so:
- users 0–15: index ≤ 494, even ⇒ exactly representable ⇒ correct.
- users ≥16: index ≥ 526 ≡ 2 (mod 4) in the step-4 region ⇒ **rounds to a wrong
  row** ⇒ gathers a different token's hidden ⇒ wrong logits ⇒ wrong greedy token.

This explains everything: boundary exactly at user 16 (index crosses 512);
deterministic; `fp32_dest_acc_en=True` fixes it (f32 indexes exactly); and the
boundary tracks the per-user stride (stride 128 ⇒ user 4 hits 512 ⇒ boundary 4,
matching the minimal-config repro).

**It is NOT a model-compute precision bug — it's integer-index arithmetic done in
bf16 in tt-mlir's `stablehlo.gather` → embedding lowering.** Fix: that index
matmul must accumulate exactly (fp32 / integer), never bf16.

Next: locate the tt-mlir gather-lowering pass, force fp32 dest-acc on the index
matmul, rebuild compiler, confirm divergence vanishes + no regression.

## THE FIX (implemented + validated)

The bad matmul is created in tt-mlir's StableHLO→TTIR gather lowering
(`lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`, ~line 4990–5015):
it typecasts integer start-indices to **f32**, builds a stride constant
(`[[32],[1]]` here), and emits a `ttir.matmul` to compute
`flat_index = indices @ strides`, then an embedding lookup. Downstream this
matmul gets `hifi4` with **no fp32 dest accumulation**, so flat indices ≥512
round to the wrong value.

**Fix (targeted, in `lib/Dialect/TTNN/Transforms/TTNNSetComputeKernelConfig.cpp`):**
force `fp32_dest_acc_en = true` on any matmul whose result feeds an embedding's
index operand (traced back through `typecast`/`reshape`/`to_layout`). Index
arithmetic must be exact; these matmuls are tiny so the perf cost is negligible.
Independent of the global `fp32_dest_acc_en` flag.

```cpp
// collect index matmuls (result feeds an embedding's indices)
llvm::DenseSet<Operation *> indexMatmuls;
moduleOp->walk([&](EmbeddingOp emb) {
  Operation *def = emb.getInput().getDefiningOp();
  while (def && isa<TypecastOp, ReshapeOp, ToLayoutOp>(def))
    def = def->getOperand(0).getDefiningOp();
  if (def && isa<MatmulOp>(def)) indexMatmuls.insert(def);
});
// ...in the per-op config walk:
if (indexMatmuls.contains(op)) config = config.withFp32DestAccEn(true);
```

### Validation (all with global `fp32_dest_acc_en` OFF)

| case | before | after |
|---|---|---|
| batch 32, 1-layer, `max_tokens=1`, `min_context_len=32` | 16/16 split (25 / 0) | **all 25, unique=1** |
| batch 32, full 28-layer | 16/16 split (720 / 320) | **all 720, unique=1** |
| batch 32, minimal config (no `min_context_len`, boundary 4) | 4/28 split (720 / 320) | **all 720, unique=1** |
| batch 32, full model, `max_tokens=16` (decode) | divergent | **all identical, unique=1** |
| op-level: `gather.34` row-divergence | users ≥16 differ | **none** |

Output is **correct** (matches user-0 / the fp32 reference), not just consistent.
The decode test confirms there is **no separate decode bug** — this single fix
resolves the full generation.

### Files changed
- `lib/Dialect/TTNN/Transforms/TTNNSetComputeKernelConfig.cpp` — the fix.
- `runtime/lib/ttnn/program_executor.cpp` — **debug-only** env-gated per-op
  row-divergence dump (`TT_DUMP_INTERMEDIATES`); used to find + verify. Revert
  before shipping the fix (kept for now for re-verification).

### Remaining — all done

- ✅ decode path (max_tokens 16 & 32): `unique=1`.
- ✅ correct output (matches user-0 / fp32 reference), not just consistent.
- ✅ generality: different prompt (31 tok, stride changes) `unique=1`.
- ✅ batch-1 sanity: coherent output, no regression.
- ✅ reverted the runtime debug patch (`program_executor.cpp`); rebuilt clean
  runtime; final clean-stack run (batch 32, full model, fp32 OFF) → `unique=1`.

### Final state of the workspace

- **Fix (only change):** `third_party/tt-mlir/.../lib/Dialect/TTNN/Transforms/TTNNSetComputeKernelConfig.cpp`. Compiler rebuilt + installed (`libTTMLIRCompiler.so` md5 `033a07254ef6ec82ee13ce641654c974`).
- Runtime back to stock (debug patch reverted, rebuilt, reinstalled).
- Fix saved as a patch: `perf_investigation/FIX_5116_gather_index_fp32.patch`.
- tt-xla branch unchanged; repro scripts in `tmp/` (untracked).

## TL;DR

**Root cause:** tt-mlir lowers a multi-dim `stablehlo.gather` (here: per-user
last-token hidden-state gather feeding the LM head) into `flat_index = indices @
strides` computed by a **float `ttnn.matmul` with bf16 destination accumulation**.
bf16 can't exactly represent integers ≥512 (step becomes 4), so any user whose
flat last-token index ≥512 gathers the **wrong row** → wrong logits → wrong greedy
token. Boundary = first user whose index hits 512 (user 16 at stride 32; user 4 at
stride 128). Deterministic; global `fp32_dest_acc_en=True` masked it by making the
index matmul exact.

**Fix:** force `fp32_dest_acc_en` on matmuls feeding embedding indices (index
arithmetic must be exact). Targeted, negligible perf cost, no global flag needed.

It is **not** a model-compute precision issue — the whole decoder backbone is
bit-row-invariant. It's integer-index arithmetic done in bf16.

---

# Revisiting chunked prefill (#4986) with the #5116 fix applied

Stack: chunked branch `kmabee/chunked_prefill_isue_4986_explore` + tt-mlir `cc7e160f3` (= `66d2edc` uplift + cherry-picked #5116 fix).

## Differentiation (the key question): pre-existing vs our chunked code

Test matrix (Llama-3.2-3B, identical prompts, greedy, fp32 OFF):

| config | result |
|---|---|
| batch 1, multi-chunk | ✅ == single-chunk reference |
| batch>1, single-shot (no chunk) | ✅ all identical (so pre-existing #5116 batch>1 prefill is fixed) |
| batch>1, multi-chunk | ❌ diverges |
| batch>1, multi-chunk, **fp32 ON** | ❌ still diverges (logic bug, not precision) |

So the remaining divergence is **only** in chunking × batch>1, is **fp32-immune** (logic, not the tt-mlir accumulation class), and is therefore in **our Stage-B chunked code** — not pre-existing infra. (Also ruled out the batched `paged_fill_cache` `c8cc0e739`: per-user fill loop diverges identically.)

## BUG #1 found + fixed (our code): non-block-aligned chunk

`ascend_scheduler._block_aligned_chunk` had:
```python
chunk = (token_budget // block_size) * block_size
if chunk <= 0: chunk = token_budget   # BUG
```
When a user gets the **budget remainder** after another user is packed into the same step and that remainder is `< block_size` (e.g. 21 with block_size 32), it scheduled a **non-block-aligned** chunk. `num_computed_tokens` then lands mid-block, and the block-granular `fill_page_table` roll (`num_computed // block_size`) misplaces that user's chunk in the KV cache → corrupted output. The docstring's own invariant ("non-final chunks must be block-aligned") was violated.

**Fix:** return 0 when a block-aligned chunk can't fit the remaining budget, and have the caller `skip_cur_request()` (defer to next step). Validated: budget-64 batch-4 multi-chunk → **all 4 users identical** (was divergent); batch-1 regression-clean. (in `ascend_scheduler.py`, Python-only.)

## BUG #2 still open (our code, likely): decode drift at larger budgets

With BUG #1 fixed, budget-256 batch-4 still diverges — but **later**: prefill + first ~4 tokens correct, then users 1–3 drift apart in **decode** (mismatch at tokens 4–8). Single-shot batch-4 decode is clean (Control 1), so it's specific to decode **after chunked prefill**, and it's per-user (users diverge from each other, not just from the reference). fp32 doesn't fix it. Suspect: residual KV-cache state / mixing of a final partial chunk with a new user's chunk at different offsets in one step. Needs a per-op dump on the budget-256 case to pin. (open)

## Net

- The pre-existing batch>1 prefill bug (#5116) is fixed in tt-mlir.
- Chunked prefill: batch=1 correct; batch>1 had a real **scheduler alignment bug in our code** (fixed), plus a second **decode-drift issue in our code** (open). Both are feature bugs, not infra — high-confidence differentiation achieved.

## BUG #2 ROOT-CAUSED (confirmed): mixing different-stage prefill chunks

The chunked scheduler packs the per-step token budget greedily: after a request's
chunk is scheduled and budget remains, it pops the next waiting request. When
prefills are **staggered** (budget < batch × prompt, so users start at different
times), this packs a **fresh** request (num_computed=0) into the same step as a
**continuation** (num_computed>0). Schedule for budget-256 batch-4 (prompt 566):

```
step3: req0 computed=512 sched=54 (user A FINAL)  +  req1 computed=0 sched=192 (user B FRESH)
step5: req0 computed=448 sched=118 (user B FINAL) +  req1 computed=0 sched=128 (user C FRESH)
step7: req0 computed=384 sched=182 (user C FINAL) +  req1 computed=0 sched=64  (user D FRESH)
```

Result: user0 (starts alone, step1) correct; users 1/2/3 (each **starts its first
chunk in a mixed step**) wrong. In a mixed step the continuation req is correct
and the **fresh req (computed=0) is corrupted** — it's forced into the
all-or-nothing `prefix_chunk_mode` (gather + masked SDPA) because another req in
the batch carries a mask.

**Decisive confirmation:** serializing prefill (one prefill request per step, no
mixing) makes budget-256 batch-4 **all-identical and == reference**. So mixing
different-stage prefill chunks is the cause. It is **our chunked code** (scheduler
packing + the all-or-nothing attention path), a logic bug, fp32-immune.

### Fix options

**Option 1 — Scheduler (CHOSEN for now; simple/safe).** Don't pack prefill
requests at different `num_computed_tokens` stages into one step. All-fresh
batches (all computed=0) and all-same-stage continuations still batch; only
staggered/mixed prefills serialize. Simple and correct. **Cost:** may reduce
prefill throughput when prefills are staggered (mixed-stage steps serialize).
Needs benchmark validation (throughput + no-recompilation).

**Option 2 — Attention path (DEFERRED; potentially higher perf).** Make the
cached-prefix path (`prefix_chunk_mode`) correct for a computed=0 user batched
with computed>0 users, so mixed-stage steps stay batched and throughput is
preserved. Needs a per-op dump of a mixed step to pin the exact mechanism (why
the fresh user is corrupted) before implementing.

> **⚠️ ACTION ITEM (do not lose track):** ship **Option 1** first for correctness.
> If chunked-prefill throughput proves insufficient because staggered prefills get
> serialized, **revisit Option 2** to keep mixed-stage steps batched. Re-evaluate
> once we have chunked-prefill perf numbers. The per-op-dump tooling for pinning
> the mixed-step mechanism is recorded in this journey (env-gated `TT_DUMP_USERS`
> patch to `runtime/lib/ttnn/program_executor.cpp`).

### Option 2 assessment (done): not cleaner — defer

Looked into whether Option 2 would be cleaner/simpler than Option 1. Conclusion:
**no — keep Option 1.**
- Option 1 is a few localized scheduler lines, proven correct at batch 4/8/16.
- Option 2's scope is **uncertain**: the exact reason a fresh (computed=0) user is
  corrupted when batched with a continuation isn't pinned yet (the batched *fill*
  was ruled out — per-user fill still diverges — so it's the gather / SDPA /
  positions for the non-first user against a very-different-offset peer). Pinning
  it needs a per-op dump of a mixed step, then delicate cached-prefix
  attention-path surgery — strictly more work and risk.
- Option 1's throughput cost is **small**: block-aligned chunks tend to fill the
  per-step budget, so chunked prefill already runs ~one user/step; mixing only
  happens in the staggered tail. So Option 2's upside is modest.
- **Decision:** Option 1 stands. Pursue Option 2 only if profiling shows the
  staggered-tail serialization measurably hurts throughput; start by per-op
  dumping a mixed step to pin the mechanism.

## Status summary
- #5116 (pre-existing, tt-mlir gather-index bf16): FIXED + committed in tt-mlir.
- BUG #1 (our code, non-block-aligned chunk): FIXED + committed (`c2c3a22c2`).
- BUG #2 (our code, mixed-stage prefill packing): root-caused + confirmed.
  Plan: **Option 1 (scheduler)** now; **Option 2 (attention path)** flagged as a
  perf follow-up — see the ACTION ITEM above.
