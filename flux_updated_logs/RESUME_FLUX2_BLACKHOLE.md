# FLUX.2-dev Transformer — Blackhole QuietBox Bringup (RESUME / HANDOFF)

> For the NEXT Claude session, started on the Blackhole QB box. The dev-box `~/.claude`
> memory does NOT travel to the QB machine — **this file is the context.** Read it first.

## TL;DR
We are pivoting FLUX.2-dev **transformer** bringup from the Galaxy 32-chip setup to a
**Blackhole QuietBox (QB)** to fit **bf16** and dodge two upstream 32-device bugs. On the QB,
run the transformer on a **1-D mesh** (`(1,8)` or `(1,4)`) in **bf16** — that avoids both bugs.
Confirm the box's chip count + DRAM, run, report whether bf16 fits.

## Why we pivoted (bug context)
On 32-chip Galaxy the transformer is blocked by BOTH mesh options:
- `(1,32)` 1-D ring → **runtime deadlock** in `completion_queue_wait`; the 1-D {1,N}->(4,8)
  reshape fails in the MGD solver → **tt-metal#43210**.
- `(4,8)` 2-D mesh → **compile error**: `sdy.collective_permute` lowering not implemented →
  **tt-mlir#3370**. Ours is the *unequal-axis* (model=8 != batch=4) swap, which the #3370
  assignee (hkwonTT) says is unsupported on non-square meshes (also throws
  `'sdy.collective_permute' op requires the same type for all operands and results`).
  Real example posted to tt-mlir#3370 (comment 4727471787).

## Why a Blackhole QB solves it
- Transformer mesh table (`TRANSFORMER_MESH_SHAPES`): **8->(1,8), 4->(1,4)** — both **1-D**.
  A 1-D mesh emits only all-reduce / all-gather / reduce-scatter (NO `collective_permute`)
  → **#3370 N/A**; and no 32->(4,8) reshape → **#43210 N/A**.
- This exact `(1,8)` path ALREADY ran e2e on the 8x n300 LLMBox; the ONLY blocker there was
  **bf16 OOM** (8.06 GB/chip on a 12.85 GB n300). Blackhole's larger DRAM (~28-32 GB/chip)
  should let bf16 fit.

## Exact footprint (from cached safetensors index)
- Transformer: **60.0 GB bf16** (~32.2B params; 8 dual + 48 single blocks, hidden 6144, FFN x3).
- Text encoder: **44.7 GB bf16** (~24B). VAE: small.

## Fit math
weights/chip = 60/N + ~1-2 GB replicated; peak = weights + transient (acts + CCL + tilize spike,
~5 GB at 8-way, larger at 4-way).

| Config | weights/chip | est. peak | DRAM/chip | verdict |
|---|---|---|---|---|
| n300 8-chip (OOM anchor) | ~8.1 GB | ~13 GB | 12.85 GB | **OOM (101%)** |
| **Blackhole 4-chip (1,4)** | ~16 GB | ~24-25 GB | ~32 GB | **likely fits (~75%); tight on 28 GB** |
| Blackhole 8-chip (1,8) | ~8 GB | ~13 GB | ~32 GB | trivially fits |

Encoder at 4-way ~11 GB/chip -> fits. **Transformer is the binding constraint.**

## DO THIS on the QB
0. Set env (gated HF repo + cache; see "Env" below). Reserve/confirm the box.
1. **Confirm box specs:** `tt-smi` -> chip count (4 or 8) + DRAM/chip (28 or 32 GB). This decides
   comfortable vs tight. Report it.
2. Sanity: `python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('tt'))"`.
3. **Run transformer, bf16, 1-D mesh (do NOT set FLUX_2D):**
   - Quick full-model bf16 repro via the isolate harness (uses `(1,N)` + bf16 by default):
     ```
     FLUX_NL=999 FLUX_NS=999 FLUX_SHARDED=1 \
       timeout 5400 pytest -svv tests/torch/models/flux2/test_transformer_realwt_isolate.py::test_realwt 2>&1 | tee bh_transformer_bf16.log
     ```
   - Or the component test in bf16: use `akannan/bringup_flux2_updated`'s `test_transformer.py`
     (already bf16 + xfail-#3370; on a 1-D mesh it should NOT xfail). On THIS debug branch
     `test_transformer.py::test_transformer_sharded` still uses `bfp_bf8` — drop that for bf16.
   - ALWAYS wrap with `timeout` + `tee`; never leave a bare detached pytest. If it hangs at high
     CPU with frozen logs, gdb the busy thread; `completion_queue_wait_front` = device hang ->
     kill + `tt-smi -r`.
4. Outcome: PCC printed = success; DRAM bank-alloc `TT_FATAL` = OOM -> fallback.

## Fallback ladder
1. bf16 on QB (goal — also escapes the bfp8 PCC question).
2. If OOM -> **bfp8 on QB**: `FLUX_WDTYPE=bfp_bf8` (isolate) or
   `CompilerConfig(experimental_weight_dtype="bfp_bf8")`. Halves weights (~8 GB/chip), fits;
   reopens bfp8 PCC (see `TRANSFORMER_PCC_DIAGNOSIS.md`).
3. If QB unavailable -> back to 32-device Galaxy and fix **tt-metal#43210** (1-D ring; cleaner,
   no collective_permute) rather than tt-mlir#3370.

## Branch map
- `akannan/bringup_flux2_updated` — CLEAN final component tests (transformer bf16 + xfail #3370,
  encoder, VAE). Submodule pointer intentionally NOT bumped (uplift after the OOM-fix PRs merge).
  **For merge.**
- `akannan/fix_flux2_transformer_oom` (tt-forge-models, @161b631558) — 2-D `(4,8)` transformer
  specs; the #732 encoder-OOM fix (`ac2212ffdf`) is an ancestor. **For PR.**
- `akannan/bringup_flux2` (THIS branch, @d5cf1b692) — debug: isolate harnesses + logs + submodule
  pinned to the 2-D specs. **For repro.**
- tt-forge-models **#732** = encoder OOM fix.

## Env (gated HF + cache)
- `black-forest-labs/FLUX.2-dev` is GATED -> need `HF_TOKEN=<token>` (ask the user; never commit it).
- Set `HF_HOME` and `TT_METAL_CACHE` onto a big disk (NOT the tiny /home). On the dev box these
  redirect to /proj_sw via `~/.bashrc`.
- **The QB is a different machine:** confirm /proj_sw + the HF cache are reachable there, else the
  gated model re-downloads. Dev-box cache: `/proj_sw/user_dev/ctr-akannan/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev`.

## Gotchas
- The transformer test runs the **text encoder** on CPU (via `load_inputs -> make_prompt_embeds`)
  to generate `encoder_hidden_states`; that's input generation, not a separate test. To isolate,
  swap it for a synthetic random tensor of shape `(1, MAX_SEQUENCE_LENGTH, 15360)`.
- `run_graph_test` runs **CPU golden first**, then TT compile. On a big-RAM host the 32B CPU
  forward is ~tens of seconds; on a small-RAM box it may be the bottleneck.

## Repro evidence (full bf16 transformer on (4,8) 2-D -> #3370, dev box, 49s)
```
Created device mesh: (4, 8) with 32 devices.
loc("reshape.8") : error: ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet: .../issues/3370.
loc("reshape.34"): error: ShardyToStableHLO lowering for CollectivePermuteOp is not implemented yet: .../issues/3370.
loc("reshape.8") : error: 'sdy.collective_permute' op requires the same type for all operands and results
module_builder.cc:749  ERR| Failed to run stablehlo pipeline
```
Logs preserved in this dir: `flux2_2d_FULL_transformer.log` (full model), `flux2_2d_nl2_ns2.log` (NL=2/NS=2).
