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
  ask the user): (a) a plugin rebuilt with **`--build-type explorer`** (turns on
  `TT_RUNTIME_DEBUG` + python bindings + installs the chisel/golden/ttmlir
  packages) — without it `import chisel` fails or records nothing; (b) `ttmlir-opt`
  on PATH and a tt-mlir source checkout (`third_party/tt-mlir/src/tt-mlir/`) for
  lit repros. The happy path (steps 1–3, accuracy only) does **not** need these.

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
   - **The committed config may FAIL TO COMPILE, not just under-perform.** Seen on
     n150: `optimization_level=2` (the `_config` default) hits a typecast
     `L1SpillManagement` / DRAM-demotion layout `TT_FATAL` during
     `_xla_warm_up_cache`. This is a compile error, **not OOM** (do NOT halve
     batch size for it). Fix: retry the baseline at `_BENCH_OPTIMIZATION_LEVEL=0`
     (precedent: the `llama-3.1-8b` config pins `optimization_level=0`) and pin
     `optimization_level=0` in the final config. Only start MP work once the
     baseline compiles.
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
   - **Globs match the vLLM parameter names, which are FUSED** for Llama-family
     models (incl. Falcon3): `gate_proj`+`up_proj` → `gate_up_proj`, and
     q/k/v → `qkv_proj`. Separate `gate_proj`/`up_proj` globs silently miss (only
     `down_proj` keeps its name). Use the fused glob, e.g.
     `{"default":"bfp_bf8","model.layers.*.mlp.gate_up_proj.weight":"bfp_bf4","model.layers.*.mlp.down_proj.weight":"bfp_bf4"}`.
     The "did not match any model parameters" warning is your tell that a glob
     used the wrong (HF) name.

Record every run (config + TOP1/TOP5 p5) in your log. Stop lowering a tensor
class when it pushes below `threshold`.

## Full-precision investigation (when baseline TOP1 < 90%)

A low baseline is often NOT a mixed-precision problem. Before any op-level MP
debugging, find out whether full precision is even better:

1. **Full-precision e2e.** Run the whole model in true bf16 — weights bf16 + kv
   bf16 + activation off: `TT_BENCHMARK_WEIGHT_DTYPE=""` against a config with no
   `experimental_kv_cache_dtype` and no `enable_activation_dtype_lowering`
   (temp-edit the `_config(...)` to plain if it bakes those in; revert after).
   Report TOP1/TOP5 p5.
   - **bf16 ≈ baseline** → **MP is NOT the cause**; the ceiling is inherent
     (small model, teacher-forced-vs-CPU reference, per-user outliers — use p5).
     Stop chasing quantization; lowering further won't recover it. (Qwen2.5-0.5B:
     68.75% TOP1 p5 in *both* bf16 and bfp8.)
   - **bf16 ≫ baseline** → quantization is hurting; go op-level (chisel, below).
   - **Whole model won't fit in bf16 on one chip (≈7B+ on n150):** SKIP the bf16
     e2e entirely — you can't run it. Use the **baseline bfp8** run as
     `baseline_acc`, and run **chisel on one layer in bfp8** (below) to catch a
     broken kernel. You lose the bf16-vs-bfp8 attribution, but that A/B is not
     informative via chisel anyway (chisel's golden bakes in the device dtype).

   **Note on model size:** very small models (≲1B) are NOT robust to quantization
   and are noisy references (Qwen2.5-0.5B: 68.75% TOP1 p5 in *both* bf16 and
   bfp8, with a 14% outlier user) — MP work is more meaningful on 7–8B models.
   Those don't fit full bf16 on n150, so follow the bfp8-only branch above.
2. **One decoder layer under chisel** to catch a broken/regressing *kernel* op —
   see the chisel recipe in "What can go wrong → B". (This is diagnostic; chisel
   measures kernel correctness, not quantization loss — the e2e A/B above is what
   decides whether MP is the culprit.)

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

1. **chisel** — per-op kernel numerics (PCC vs a torch golden). Validated recipe:
   - **Build:** rebuild the plugin with `python setup.py bdist_wheel --build-type explorer`
     (auto-enables `TT_RUNTIME_DEBUG` + python bindings and installs the
     chisel/golden/ttmlir packages; without it `import chisel` fails or records
     nothing). ~20–45 min; install the resulting wheel (glob carefully — pick the
     newest wheel, not a stale one).
   - **Trace OFF, always with a timeout:** run with `trace_enabled=False`. Trace-on
     raises `TT_FATAL: Reads are not supported during trace capture` and HANGS for
     hours (chisel reads each op's output). Wrap every run in `timeout 900`.
   - **Run ONE layer IN-PROCESS via `test_llms.py`** (chisel captures PJRT
     in-process; vLLM runs the model in a worker subprocess and is NOT captured):
     `timeout 900 pytest -svv --enable-chisel --num-layers 1 tests/benchmark/test_llms.py::test_<model>`.
     Set the run's weight/kv dtype by temp-editing that test's `test_llm(...)` call
     (`experimental_weight_dtype=""` for full precision, `"bfp_bf8"` for baseline);
     `--num-layers` works here (it's ignored by the vLLM benchmark). Copy
     `chisel_results/*.jsonl` aside after each run (the next run overwrites it).
   - **Rank:** load the JSONL, keep `check == "numerics"`, sort ascending by
     `payload.pcc`.
   - **EXIT=124 is expected:** even with trace off, the pytest process often hangs
     in teardown *after* chisel has written a complete JSONL, so the `timeout` kills
     it (exit 124). Always inspect `chisel_results/*.jsonl` before concluding the
     run failed — the records are usually all there.
   - **Interpretation (critical):** chisel's golden is *promoted from device
     tensors*, so it measures **kernel correctness (device vs torch-of-its-inputs),
     NOT quantization-vs-fp32** — and it may not score matmul/linear (goldens get
     evicted). Use it to catch a *broken/regressing kernel op*, NOT to quantify
     weight-quantization loss (the full-precision e2e A/B does that). **Ignore
     degenerate ops:** `ttnn.max`/`ttnn.eq`/`ttnn.argmax` PCC≈0 is expected
     (argmax/boolean outputs); `ttnn.fill_cache` low PCC is an in-place-cache
     accounting artifact. Records are accumulated-mode via the fixture (isolated
     mode may be empty).
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
- Baseline TOP1 < 90% and you start lowering dtypes without the full-precision e2e A/B → run bf16 first; MP may not be the cause at all.
- Running chisel with trace enabled → it hangs for hours (reads unsupported during trace capture); set `trace_enabled=False` + `timeout`.
- Treating chisel PCC as a quantization metric → it measures kernel correctness, not quant-vs-fp32; ignore degenerate `max`/`eq`/`argmax`.
- Assuming an override applied without checking the `modules/` ttnn IR → verify in IR.
- Guess-and-check on knobs without a log of what you tried → log every run.
