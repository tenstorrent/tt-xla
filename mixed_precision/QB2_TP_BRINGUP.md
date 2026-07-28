# QB2 / 4-chip TP Mixed-Precision Bringup Runbook

Instructions for a fresh agent doing MP bringup of the **tensor-parallel (TP) LLMs**
on a **4×p150** machine. Read the `mixed-precision-bringup` skill first — it is the
authority on the MP method (baseline → all-features-on → bfp4 MLP, chisel, IR
verification). This file only covers what's **different for the 4-chip TP path**.

## Scope

Six untuned (bf16) LLM TP configs. Bring them up in this order (cheapest CPU
golden first — the 24–32B goldens need a lot of host RAM):

| # | model | ~params | vLLM perf config (bf16) | `test_llms.py` accuracy entry |
|---|---|---|---|---|
| 1 | Falcon3-7B-Base | 7B | `falcon3-7b-qb2-tp` | `test_falcon3_7b_tp_qb2` |
| 2 | Llama-3.1-8B-Instruct | 8B | `llama-3.1-8b-qb2-tp` | `test_llama_3_1_8b_instruct_tp_qb2` |
| 3 | Falcon3-10B-Base | 10B | `falcon3-10b-qb2-tp` | `test_falcon3_10b_tp_qb2` |
| 4 | Mistral-Small-24B-Instruct-2501 | 24B | `mistral-small-24b-instruct-2501-qb2-tp` | `test_mistral_small_24b_instruct_2501_tp_qb2` |
| 5 | Qwen2.5-Coder-32B-Instruct | 32B | `qwen2.5-coder-32b-instruct-qb2-tp` | `test_qwen_2_5_coder_32b_instruct_tp_qb2` |
| 6 | Qwen3-32B | 32B | `qwen3-32b-qb2-tp` | `test_qwen_3_32b_tp_qb2` |

One agent per model, strictly sequential (each inherits an improved skill).

## Caveats (READ FIRST)

1. **This machine is 4×p150 (blackhole), NOT the real qb2 (2×p300).** It's close
   enough for MP tuning (same dtypes/passes), but it is a *different* machine, so
   **before any long golden generation, first run the plain vLLM TP e2e to confirm
   the model even compiles/runs here** (Step 0). Don't burn hours generating a
   32B CPU reference for a model that doesn't come up on this box.

2. **Accuracy uses a CPU golden — that's expected, not a blocker.** vLLM TP
   accuracy is guarded off, so accuracy comes from the **`test_llms.py` TP entry**,
   which CPU-generates the full (multi-billion-param) reference. This is slow and
   host-RAM-heavy (bf16 ≈ 2 GB/B params; a 32B ≈ 64 GB RAM) but it is the required
   method. **The reference is cached to `tests/benchmark/llm_utils/reference_outputs/<model>.refpt`
   and auto-reused** (`needs_regeneration` checks it). Generate it ONCE per model
   (at the baseline) — every later MP variant reuses the same refpt, because the
   golden is the CPU model and is independent of the device dtype. **Keep the
   `.refpt` files** (consider committing them) so you never regenerate.

3. **Activation-dtype lowering IS live on TP** (unlike single-chip) — the pass
   matches subgraphs around CCL ops, which exist across 4 devices. So all three
   MP knobs are in play here.

4. **opt-level compile risk** (seen on n150 for 7B): the committed default
   `optimization_level=2` can fail to *compile* the bfp8 typecast path. If a run
   fails to compile (typecast/L1-spill `TT_FATAL`, not OOM), drop to opt-level 0.

## Environment

- **Check out the branch first:**
  ```bash
  git fetch origin
  git checkout dgolubovic/mp-agentic-bringup
  git pull --ff-only origin dgolubovic/mp-agentic-bringup
  ```
  All the MP work (skill, tuned configs, accuracy harness, this runbook) lives on
  `dgolubovic/mp-agentic-bringup` — NOT on `main`. The skill is at
  `.claude/skills/mixed-precision-bringup/`; the bringup order at
  `mixed_precision/MP_BRINGUP_ORDER.md`.
- Activate every shell:
  `export SYSTEM_DIST_PACKAGES="${SYSTEM_DIST_PACKAGES:-}"; source venv/activate`.
- The plugin is already an `--build-type explorer` (debug) build → `import chisel`
  works (verify: `python -c "import chisel"`). If it's a fresh reservation, the
  env/deps may need rebuilding — see the skill's Step 0 + `mixed_precision/logs/env/`.
- Confirm 4 devices: `python -c "import jax; print(jax.devices('tt'))"` → expect 4.
- The TP mesh auto-sizes to the available device count (mesh_shape=None / the
  model loader's `get_mesh_config`), so the same tests use all 4 chips here.

## Per-model workflow

Wrap every device run in `timeout` (e.g. `timeout 3600`); log to
`mixed_precision/logs/<vllm-id>-bringup.log` (use the skill's template).

**Step 0 — Sanity e2e on this machine (NO accuracy, ~minutes).** Runs the
committed bf16 vLLM TP config; confirms the model compiles + generates tokens on
4×p150 (and that bf16 fits across 4 chips):
```bash
timeout 3600 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_tp_benchmark[<vllm-id>]"
```
If this fails to compile → try `_BENCH_OPTIMIZATION_LEVEL=0`. Do NOT proceed to
golden generation until Step 0 passes.

**Step 1 — Baseline accuracy (generates + caches the refpt ONCE).** Full-precision
bf16 baseline via the `test_llms.py` TP entry. NOTE: `test_llms.py` DEFAULTS to
`experimental_weight_dtype="bfp_bf8"` + `experimental_kv_cache_dtype="bfp_bf8"`, so
for a true bf16 baseline temp-edit the entry's `test_llm_tp(...)` call to add
`experimental_weight_dtype=""`, `experimental_kv_cache_dtype=None`,
`enable_activation_dtype_lowering=False` (these flow via `**kwargs`):
```bash
timeout 14400 pytest -svv tests/benchmark/test_llms.py::<accuracy-entry> --accuracy-testing
```
First run CPU-generates `<model>.refpt` (slow); record baseline_acc = TOP1 p5,
threshold = 0.90 × baseline_acc. Keep the refpt.

**Step 2 — Apply MP (each step reuses the cached refpt; gate at threshold).**
Temp-edit the same `test_llm_tp(...)` call per config and re-run --accuracy-testing:
1. weights → `bfp_bf8` (+ `experimental_kv_cache_dtype="bfp_bf8"`, and
   `enable_activation_dtype_lowering=True` since CCLs are live on TP).
2. bfp4 on MLP via `weight_dtype_overrides` — **confirm the vLLM FUSED param names
   first** (Llama-family incl. Falcon3 fuse `gate_up_proj`, `qkv_proj`); verify the
   override applied in the ttnn IR (`grep -c bfp_bf4`).
Record every run's TOP1/TOP5 p5.

**Step 3 — chisel (only if a config drops below threshold).** Per the skill:
explorer build (already present), `trace_enabled=False`, `timeout`, one layer via
`--num-layers 1` on the `test_llms.py` TP entry. chisel = kernel correctness, not
quant-vs-fp32; ignore degenerate `max`/`eq`/`argmax`; EXIT=124 teardown hang is
expected (JSONL still complete).

**Step 4 — Bake the result.** Mirror the winning dtypes into BOTH the `test_llms.py`
TP entry (accuracy/perf) and the vLLM `_tp_config(...)` entry (for the vLLM perf
matrix), as uncommitted edits. Document kept/dropped features + why in the log.
Do NOT edit the skill mid-session; propose edits at the end, only with user approval.

## Reporting

At the end: baseline_acc + threshold, table of configs → TOP1/TOP5 p5, final
recommended MP config (both files), chisel findings if any, proposed skill edits,
and the log path.
