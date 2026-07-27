---
name: mixed-precision-bringup
description: Use when bringing up or tuning mixed precision (MP) on a vLLM-hosted LLM on Tenstorrent hardware — lowering weight / KV-cache / activation dtypes to bfp8/bfp4 while holding token accuracy. Triggers include "MP bringup", weight_dtype_overrides, experimental_kv_cache_dtype, enable_activation_dtype_lowering, bfp8/bfp4 weight lowering, TOP1/TOP5 accuracy drop after quantization, chisel op-by-op analysis, a pattern/pass not triggering in the ttnn IR.
---

# Mixed Precision (MP) Bringup

## ⚠️ THIS SKILL IS WORK IN PROGRESS — read this first

This skill is being refined empirically through real model bringups. One fresh
agent brings up ONE model per session; each subsequent agent should inherit a
better skill.

**You (the bringup agent) may edit this skill ONLY when ALL of these hold:**
1. The MP bringup for your model is **finished** (accuracy resolved, or you've
   documented why a feature was dropped).
2. You have written a **root-cause analysis of what went wrong** during the
   bringup (in your log file — see [Logging](#logging-mandatory)).
3. The **user has explicitly approved** the specific edit you propose.

Never edit this skill mid-bringup, never "improve it while you work", and never
edit it without user sign-off. If you learned something, put it in your log and
*propose* the skill change at the very end. **Letter and spirit both apply** —
"I'm only adding a helpful note" is still a mid-session edit; don't.

## Overview

MP bringup optimizes an **already-working** model by applying tt-mlir mixed-
precision features that lower tensor dtypes to save memory/bandwidth. Every
lowering loses information; the objective is to lose the **least** information
(smallest TOP1/TOP5 drop) while saving the **most** memory.

The three MP features (all implemented as tt-mlir graph patterns):

| Feature | vLLM knob (`additional_config`) | Values |
|---|---|---|
| Weight dtype lowering | `experimental_weight_dtype` (global) + `weight_dtype_overrides` (per-layer `{glob: dtype}` + `"default"`) | `bfp_bf8`, `bfp_bf4` |
| KV cache dtype lowering | `experimental_kv_cache_dtype` | `bfp_bf8` (`bfp_bf4` NOT impl, #5011) |
| Activation dtype lowering (around CCLs) | `enable_activation_dtype_lowering` | `true`/`false` |

All 3 can each be fine in isolation but **drop accuracy when combined**. If so:
debug the cause, or drop one feature. Which to drop depends on the task — e.g.
for long sequences KV-cache dtype dominates memory, so prefer keeping it.

**Single-chip caveat:** activation-dtype lowering matches subgraphs *around CCL
ops* (reduce_scatter/all_gather), which don't exist on one chip — so it is a
**no-op on single-chip (n150) model tests**. Do NOT set
`enable_activation_dtype_lowering` for single-chip configs (it just adds a
misleading knob); it only matters on multi-device TP.

## Dictionary

- **TOP1 accuracy** — % of decode positions where the device's argmax token
  equals the CPU reference model's argmax.
- **TOP5 accuracy** — % of positions where the device's argmax is in the CPU
  reference model's top-5.
- **"Full precision" / starting point** — the model's *current working* config,
  whatever dtypes it already has (e.g. weights bfp8, KV bf16, activations bf16).
  This is where THIS bringup starts.
- **baseline_acc** — TOP1 (the **p5** percentile across users, not the mean —
  per-user variance is real) of the full-precision starting point = the model's
  **current `_config(...)` as-is**. Note most single-device configs already
  default to `experimental_weight_dtype="bfp_bf8"`, so happy-path step 2
  ("lower to bfp8") is often already done. Optionally also record a true-bf16
  ceiling (`TT_BENCHMARK_WEIGHT_DTYPE=""`) — but `baseline_acc` for the
  threshold is the current-config number.
- **threshold** — user-specified, else **90% of baseline_acc** by default.

## Prerequisites (environment)

- Work on branch `dgolubovic/mp-agentic-bringup` (has the vLLM teacher-forced
  accuracy harness + the `enable_activation_dtype_lowering` vLLM knob).
- Bringup order & per-model targets: `mixed_precision/MP_BRINGUP_ORDER.md`.
- All single-device vLLM configs run on **n150** (current machine). TP models
  need qb2-blackhole/galaxy and vLLM TP accuracy is unsupported (CPU reference
  too large) — use `test_llms.py` TP accuracy jobs for those.
- Create the log dir once: `mkdir -p mixed_precision/logs`.
- **For the step-B diagnosis path you also need** (confirm before starting, else
  ask the user): (a) the plugin built with **`TT_RUNTIME_DEBUG` +
  `TTMLIR_ENABLE_BINDINGS_PYTHON`** — without it `import chisel` fails and
  chisel/emit-ttnn are impossible; (b) `ttmlir-opt` on PATH and a tt-mlir source
  checkout (`third_party/tt-mlir/src/tt-mlir/`) for lit repros. The happy path
  (steps 1–3, accuracy only) does **not** need these.

## Running accuracy

Teacher-forced decode accuracy vs a CPU HF reference (auto-generated `.refpt` on
first run). Reports TOP1/TOP5 min/**p5**/median/mean/max; `evaluation_score` = TOP1 p5.

```bash
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[<id>]" --accuracy-testing
```
`<id>` is the pytest id from `SINGLE_DEVICE_CONFIGS` (e.g. `qwen2.5-0.5b-instruct`).

**Setting MP knobs without editing configs** (fast A/B — env vars, see the header
of `test_vllm_benchmarks.py`):
- `TT_BENCHMARK_WEIGHT_DTYPE=bfp_bf8|bfp_bf4|""`
- `TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8`
- `TT_BENCHMARK_WEIGHT_OVERRIDES=/path/to/overrides.json` (`{glob: dtype}` + `"default"`)

Activation lowering and layer-count have **no env var** — you must **edit the
model's checked-in `_config(...)` entry** in `SINGLE_DEVICE_CONFIGS`
(`tests/benchmark/test_vllm_benchmarks.py`), adding
`enable_activation_dtype_lowering=True` and/or `num_hidden_layers=1`. Revert the
edit when done. For a true bf16 weight baseline, set `TT_BENCHMARK_WEIGHT_DTYPE=""`.

**`_config(...)` accepts ANY `additional_config` key as a kwarg** (they flow
through `**additional_config_extra`). So `_config(model, experimental_kv_cache_dtype="bfp_bf8")`
works — this is the clean way to **bake a final tuned config** into the checked-in
entry (kv dtype, weight_dtype_overrides, num_hidden_layers, …). The env vars above
are just for fast A/B without editing code.

**Limit layers for fast IR inspection:** vLLM uses `num_hidden_layers` in
`additional_config` (0 = full model); a 1-layer model has all components but a
tiny graph — ideal for verifying patterns trigger (below). **Trap:** the
`--num-layers` pytest CLI option exists but the vLLM benchmark test does **not**
consume it (it's silently ignored) — set `num_hidden_layers` in `_config(...)`,
not `--num-layers`.

## The happy-path bringup

**REQUIRED BACKGROUND:** if you hit an accuracy drop or a pattern that won't
trigger, use `superpowers:systematic-debugging` — don't guess-and-check.

**Step 0 — environment sanity-check (do this FIRST, before the slow baseline run).**
A stale venv silently breaks the very first run and looks nothing like an MP
issue. The branch inherits main's PyTorch/vLLM uplifts, so the installed vLLM
must match the branch. Cheap pre-flight:
```bash
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import vllm; from vllm.v1.worker.worker_base import CompilationTimes"  # smoke: plugin API present?
```
If the second line raises `ImportError` (e.g. `cannot import name 'CompilationTimes'`),
the venv **predates the branch's uplift** — STOP. **Do NOT `pip install vllm==<pin>`
from public PyPI to fix it:** that resolves to CUDA torch + toolkit and destroys
the CPU/TT env. The env is re-provisioned from **TT release wheels** (`vllm-tt`,
`xla-torch`, pjrt) via `pypi.eng.aws.tenstorrent.com` (see CI `call-perf-test.yml`:
`source venv/activate && pip install wheels/*.whl`) — that's a coordinated
multi-wheel uplift, **out of scope for a bringup: escalate to the user.**

1. **Baseline.** Run accuracy on the current config → record `baseline_acc`
   (TOP1 p5). Set `threshold` (user value, else 0.90 × baseline_acc).
2. **All features on, weights → bfp8.** Turn on KV-cache and activation lowering
   (both true), and lower all weights to `bfp_bf8` if not already. Run accuracy.
3. **Weights → bfp4 on MLP/MoE.** With everything at bfp8, push MLP/MoE matmul
   weights to `bfp_bf4` via `weight_dtype_overrides`
   (`{"default":"bfp_bf8","model.layers.*.mlp.*proj.weight":"bfp_bf4"}`). These
   layers are the largest and usually most quantization-robust vs attention
   weights. Run accuracy.
   - **Globs match the FULL parameter path**, which for standard Linear layers
     ends in `.weight` — omit it (`...*proj`) and the pattern matches nothing
     and silently does not lower (you'll see a "did not match any model
     parameters" warning). MoE experts differ (e.g. gpt-oss uses bare
     `...mlp.experts.gate_up_proj`, no `.weight`). **Confirm the real parameter
     names first** (from the model or the IR) — don't trust the example glob.

Record every run (config + TOP1/TOP5 p5) in your log. Stop lowering a tensor
class when it pushes below `threshold`.

## What can go wrong

### A. A pattern / pass is not triggering

Marking weights bfp4 (or enabling KV/activation lowering) may not propagate —
the tt-mlir pattern didn't match. Verify in the dumped IR:

- IRs dump under `modules/irs/` (the benchmark sets `export_path`), as
  **timestamped per-graph files that ACCUMULATE** across runs — a naive
  before/after diff is awkward. Instead, on the newest largest-prefill graph
  (`ttnn_..._g6_*.mlir` — the one with the matmuls), COUNT dtype tokens:
  `grep -c bfp_bf4 <file>` and `grep -c bfp_bf8 <file>`, before vs after applying
  the override. It took effect if the bfp4 count jumps (e.g. 1 enum-legend-only →
  hundreds) and bfp8 drops correspondingly.
- Fast first-line positive signal: the **absence** of a "did not match any model
  parameters" warning in the run output means the override glob matched ≥1 param.
  (Still confirm in the IR — a glob matching a param ≠ the pass actually applying.)
- Do this on a **1-layer** model (`num_hidden_layers=1`) — small graph, all
  components.
- The three passes and where they live (all under `third_party/tt-mlir/src/tt-mlir/`):

| Feature | Pass class | `ttmlir-opt` flag | Lit tests dir |
|---|---|---|---|
| Weight dtype | `TTNNWeightDtypeConversion` (+ TTIR `TTIRPropagateWeightDtype` for per-tensor) | `--ttnn-weight-dtype-conversion="target-dtype=bfp_bf8"` (`--ttir-propagate-weight-dtype`) | `test/ttmlir/Dialect/TTNN/weight_conversion/` |
| Activation (CCL) | `TTNNCCLActivationDtypeLowering` | `--ttnn-ccl-activation-dtype-lowering` | `test/ttmlir/Dialect/TTNN/activation_dtype_lowering/` |
| KV cache | `TTNNKVCacheDtypeConversion` (+ TTIR `TTIRInferKVCacheArgumentTypes`) | `--ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8"` | `test/ttmlir/Dialect/TTNN/kv_cache_conversion/` |

Impl: `lib/Dialect/TTNN/Transforms/TTNN{WeightDtypeConversion,ActivationDtypeLowering,KVCacheDtypeConversion}.cpp`.
Per-tensor overrides flow: tt-xla `tt.weight_dtype_override` custom_call →
`ttcore.weight_dtype` func-arg annotation → `TTIRPropagateWeightDtype`.

**When a pattern should trigger but doesn't:**
1. Create a minimal **ttnn IR lit repro** targeting only that pass (copy a RUN
   line + input from the lit-tests dir above; `ttmlir-opt <flag> repro.mlir`).
2. Confirm the repro reproduces the miss in tt-mlir.
3. Write a tt-mlir fix; re-run the lit test + confirm the pattern now triggers
   AND applies in the full ttnn IR dump.
4. Continue the bringup, and open a **draft PR** on tt-mlir for the fix.

### B. E2E accuracy below threshold

Could be information loss (expected), a kernel bug, or an un-maxed compute-kernel
config (`math_fidelity`, `fp32_dest_acc_en`, packer L1 acc, math approx mode).
Diagnose op-by-op — do not guess:

1. **chisel** — op-by-op accuracy analysis to find the ops regressing accuracy
   most. It's the autouse `chisel_context` fixture in `tests/conftest.py` (gated
   on `--enable-chisel`), inherited by `tests/benchmark/`; results land in
   `chisel_results/`. Requires the debug build (see Prerequisites) or
   `import chisel` fails. Candidate invocation:
   `pytest --enable-chisel "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[<id>]" --accuracy-testing`.
   ⚠️ **UNVERIFIED for vLLM:** the fixture opens in the pytest process while
   vLLM runs the model in **worker subprocesses**, so it may not capture the
   model's ops. **First bringup that reaches step B must verify whether chisel
   captures vLLM-worker ops, read the `chisel_results/` format, and document how
   to rank ops** — then propose that as a skill edit.
2. **emit ttnn (codegen)** — emit standalone ttnn Python during a vLLM run, edit
   it, then reload instead of compiling. Driven by env vars
   `TTXLA_CODEGEN_EXPORT_DIR` (emit) / `TTXLA_CODEGEN_LOAD_DIR` (load); see
   `tests/integrations/vllm_plugin/codegen/test_codegen_emit_load.py` for the
   exact pattern. (`python_package/tt_torch/codegen.py` is the torch-XLA
   nn.Module path — NOT the vLLM path.)
3. **Bypass bad ops to CPU.** In the emitted+reloaded ttnn code, route the worst
   ops to CPU equivalents; start with the top p% worst ops, shrink until only
   the truly-hurting ops remain. ⚠️ **UNVERIFIED:** the exact edit to route one
   op to CPU is not yet documented — first bringup to use this must establish
   and document the mechanism.
4. **Single-op repro.** For each culprit, write a single-op test reproducing the
   bad accuracy, then try: raise compute-kernel-config knobs, raise that op's
   weight dtype, or debug the kernel. Recover accuracy while keeping the rest of
   the model at the lowest precision possible.

## Logging (MANDATORY)

Keep **one** log file for the whole session:
`mixed_precision/logs/<model-id>-bringup.log` (see `bringup-log-template.md`).
Record, in order: baseline_acc + threshold; every config tried with its exact
knobs and TOP1/TOP5 p5; every IR/lit-test/chisel/single-op investigation and its
outcome; the final config + which features were kept/dropped and why; and a
closing **root-cause analysis of what went wrong**. This log is the basis for any
skill edit you later propose.

## Red flags — STOP

- About to edit this skill before the bringup is done, or without user approval → don't (see WIP header).
- Concluding "accuracy is just quantization loss" without a chisel op-by-op run → run chisel first.
- Assuming an override applied without checking the `modules/` ttnn IR → verify in IR.
- Guess-and-check on knobs without a log of what you tried → log every run.
