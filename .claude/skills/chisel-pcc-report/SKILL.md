---
name: chisel-pcc-report
description: Turn a chisel op-by-op numerics report into a human-readable PCC-failure report. Use when analyzing a chisel PCC/numerics report, investigating which TTNN ops failed numerics validation, or when given a `*_report.jsonl` chisel report (optionally with one or more `ttnn_runtime_*.mlir` graphs). Produces a markdown report (which ops failed, their shapes/config, min & max PCC across devices, a quick suspected cause) plus a filtered JSONL of the collapsed failures. Trigger on phrases like "pcc report", "chisel report", "which ops failed numerics", or when handed a chisel report jsonl.
argument-hint: <report.jsonl> [graph.mlir ...] [save]
---

# Chisel PCC Report

Chisel (tt-mlir instrumentation, `third_party/tt-mlir/src/tt-mlir/tools/chisel/`)
runs alongside a model and checks every TTNN op's output against a golden,
emitting one JSONL record per op-per-device. This skill collapses that report to
the worst failures, enriches each with shapes/config from the serialized graph,
and writes a scannable markdown report — shallow by default, so the user can ask
for a deeper dive on specific ops.

## Inputs

| Input | Required | Notes |
|-------|----------|-------|
| Raw chisel report JSONL | yes | e.g. `llm_benchmark_deepseek_v3_1_report.jsonl`. The **raw** report — the skill does the filtering. |
| `ttnn_runtime_*.mlir` graph(s) | optional | Under `modules/irs/`. Used to enrich failing ops with shapes/config via SSA cross-ref. A report may span multiple `binary_id`s, each potentially a different graph. |
| `save` token | optional | If omitted, still write the markdown (default is to write). Present for consistency with other skills. |

## Record schema (so you don't re-discover it)

Each JSONL line: `op` (e.g. `ttnn.softmax`), `check` (`numerics`,
`mlir_vs_runtime_tensor`, `golden_*`, `_default_pre_op`…), `ssa` (e.g. `%200`,
`%indices`), `op_asm` (full MLIR asm — **present only on non-numerics checks**),
`binary_id` (int), `program_name` (`main`, `main_const_eval_0`…),
`program_index` (int), `payload`.

For `check == "numerics"` the payload has: `status` (`ok` | `numerics_fail`),
`mode` (`isolated` | `accumulated`), `pcc` (float | null), `atol`, `rtol`,
`device_id`. **Numerics records carry no `op_asm`** — so shapes/config for a
failing op must come from the MLIR graph (cross-ref by SSA), which is why the
graph prompt in step 3 matters.

A failure's identity is the tuple
`(op, check, ssa, binary_id, program_name, program_index)`; it recurs once per
`device_id`, so PCC varies per device → report **min and max** across devices.
Default PCC pass threshold is `0.99` (see chisel `validators.py`).

## Output location

**All artifacts go under `chisel_analysis/`** (created in the repo root) — keep the repo root and
the tt-mlir tree clean. Reports, the filtered/summary JSONL, and any generated repro tests live
here. Create it once with `mkdir -p chisel_analysis chisel_analysis/repros`.

## Workflow

### 1. Collapse the report (deterministic)

Derive a `<model-tag>` from the report filename (e.g.
`llm_benchmark_deepseek_v3_1_report.jsonl` → `deepseek_v3_1`). Run the bundled
helper — never hand-parse the raw JSONL (it can be 100k+ lines):

```bash
mkdir -p chisel_analysis
python3 .claude/skills/chisel-pcc-report/scripts/collapse_minmax.py <report.jsonl> \
    --filtered-out chisel_analysis/pcc_filtered_<model-tag>.jsonl \
    --summary-out  chisel_analysis/pcc_summary_<model-tag>.json
```

> Python must run inside the project's docker container — if a bare
> `python3 …` is refused or fails to import, run it inside the container.

This keeps only `numerics_fail` + `isolated` records, groups by identity, and
emits: `chisel_analysis/pcc_filtered_<model-tag>.jsonl` (verbatim min/max lines,
for reference) and `chisel_analysis/pcc_summary_<model-tag>.json` — a compact,
**worst-first** summary with
per-op `min`/`max` `{pcc, atol, rtol, device_id}`, `num_devices`,
`binary_ids_with_failures`, and a separate `harness_issues` list
(`shape_mismatch` / `dtype_mismatch` / `chisel_bug`). Read the summary JSON.

### 2. Read the summary, not the raw report

Work from `pcc_summary_<model-tag>.json`. The `failures` array is already sorted
worst (lowest min PCC) first. Note the distinct `binary_ids_with_failures`.

### 3. Resolve a graph per binary_id

For **each** `binary_id` that has failures, you need its `ttnn_runtime_*.mlir`
graph to enrich the ops. If the user supplied graph(s), match them. If a failing
`binary_id` has **no** graph yet, **ask the user to provide the graph for that
binary_id** (point them at `modules/irs/ttnn_runtime_*.mlir`). Only if they
confirm none is available do you proceed without it for that binary_id — say so
explicitly in the report; do not silently skip.

### 4. Enrich each failing op from the graph

For each failing `ssa`, find its definition in the matching graph, e.g.:

```bash
grep -n '%200 =' modules/irs/ttnn_runtime_deepseek_v3_1_..._g0_....mlir
```

```
%200 = "ttnn.softmax"(%199) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, dimension = 3 : si32, numericStable = true}> : (tensor<16x16x16x128xf32, #ttnn_layout179>) -> tensor<16x16x16x128xf32, ...>
```

Extract: input/output tensor shapes & dtypes, and key config attributes
(`dimension`, `math_fidelity`, `numericStable`, layout/memory config, etc.).
Named SSAs (`%indices`, `%indices_1`) define multi-result ops — grep the bare
name. If no graph is available for a binary_id, fall back to whatever `op_asm`
exists for that op elsewhere in the report; otherwise report shapes as unknown.

### 5. Group near-duplicate failures

Collapse failures that are virtually the same — **same op name, same input
shapes, same config, and similar PCC** — into a single issue, even across
different SSAs or programs. List **every** contributing SSA and its
corresponding graph line so each occurrence stays traceable. Do not merge ops
that differ in shape or config.

### 6. Per-issue suspected cause (shallow)

Give **one short line** per issue from simple heuristics — do not deep-dive:

| Pattern | Quick suspicion to surface |
|---------|----------------------------|
| `softmax` / reductions over a large axis | accumulation precision / reduction along `dimension`; check `numericStable`, math fidelity, fp32 accumulation |
| `topk` / `argmax` (index outputs) | index PCC is inherently noisy on ties; low PCC on `%indices` may be benign — check atol on values instead |
| `to_layout` / `to_memory_config` / reshape | tilization / sharding / memory-layout mismatch rather than math |
| `matmul` / `linear`, wide-K, bf16 | bf16 accumulation precision over large contraction dim |
| failures only on a subset of `device_id`s | per-shard issue (sharding boundary / mesh dim), not a uniform op bug |

### 7. Write outputs

Write the markdown report `chisel_analysis/pcc_report_<model-tag>.md` and print a
concise console summary (total failing ops/issues, worst PCC, top offending op
types, and any binary_ids left un-enriched). Artifacts produced (all under
`chisel_analysis/`): `pcc_report_<model-tag>.md`, `pcc_filtered_<model-tag>.jsonl`,
`pcc_summary_<model-tag>.json`.

## Drill-down: generate a tt-mlir op repro (on request)

This is **not** part of the default report. Do it only when the user asks to dig into a
**specific op** AND the evidence points at the **op kernel itself** being buggy. The decisive
signal is **isolated‑mode** failure: chisel's isolated mode feeds the *same device input tensor*
into both the device op and the torch golden, so an isolated PCC failure means the op and
`torch` disagree on identical input — i.e. a kernel bug, not an upstream/mask/golden problem.
(If the op is an index output like `topk`, a pure data‑movement op, or a near‑constant reduction,
suspect a metric artifact instead — say so rather than generating a repro.)

When that bar is met, write a standalone tt-mlir golden test the user can drop into a tt-mlir
checkout. **You only author the file and give instructions — the user copies it in, builds, and
runs.** Do not try to run it here, and **do not write it into the tt-mlir tree.**

**Where you write it:** `chisel_analysis/repros/test_<op>_<symptom>_repro.py` (alongside the other
analysis artifacts). **Where the user copies it:** into their cloned tt-mlir at
`test/python/golden/` (sibling of `test_compute_kernel_config.py`). Tell them this path explicitly.

tt-xla vendors tt-mlir at `third_party/tt-mlir/src/tt-mlir/`, so read existing tests *there* for the
current API before writing (verify `builder.<op>` signature and `compile_and_execute_ttir` kwargs —
they drift) — but **save your generated file under `chisel_analysis/repros/`, not in that tree.**

**How to author it (pattern):**
1. Enrich the failing op from the graph (shapes, dtype, config) as in steps 3–4.
2. Reconstruct the **input distribution that triggers the failure** — in isolated mode the input
   is what matters. If the op consumes a masked/special tensor, replicate it exactly (e.g. the
   `softmax` case: masked entries == the model's `ttnn.full` sentinel value, traced from the graph).
3. Inject the input verbatim and provide a torch golden via `builder.set_goldens({in0: data},
   {out: golden})` inside a `@builder.func([shape], [dtype])` module; call `compile_and_execute_ttir(
   module, **get_request_kwargs(request), device=device, pipeline_options=[...], target="ttnn")`.
   PCC is checked by default (`pcc=0.99`); a divergent kernel makes the test fail.
4. Mirror the model's compute config via `pipeline_options`, e.g.
   `["compute-cfg-math-fidelity=hifi4", "compute-cfg-fp32-dest-acc-en=true"]`.
5. **Parametrize a control** that isolates the hypothesis (e.g. the suspected sentinel value vs a
   milder one), so a pass/fail split confirms root cause. Document expected pass/fail per param in
   the docstring.
6. **Scope = single chip** unless the failing op has a CCL/all‑reduce/`mesh_shard` on its
   reduction axis. Check the per‑device graph: if the op's layout is local (`<1x1>` grid, reduction
   dim fully on-device) the bug reproduces single‑chip and the default mesh is correct — failures
   across many `device_id`s just mean the same local kernel ran on every shard. Only build a
   multi‑device variant if the op genuinely depends on cross‑chip data.

**Tell the user how to run it** (put these in the test docstring too):
```bash
# copy the generated repro into your cloned + built tt-mlir checkout, then:
cp chisel_analysis/repros/test_<op>_<symptom>_repro.py <tt-mlir>/test/python/golden/
cd <tt-mlir>
source env/activate
ttrt query --save-artifacts
export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys
pytest -svv test/python/golden/test_<op>_<symptom>_repro.py --sys-desc=$SYSTEM_DESC_PATH
```
State plainly that you have not executed it (no built tt-mlir + silicon here), and what result
confirms vs refutes the hypothesis (e.g. "the `bf16_min` params fail with low PCC while the milder
control passes → kernel bug on the sentinel").

## Report template (`pcc_report_<model-tag>.md`)

```markdown
# PCC Failure Report — <model-tag>

**Source:** `<report.jsonl>`  •  **Failing ops:** N (M issues after grouping)
**Worst PCC:** <pcc> on `<op>` `<ssa>`  •  **PCC threshold:** 0.99
**Graphs used:** binary_id 1 → `<graph>`; binary_id 3 → *(none provided — confirmed unavailable)*

## Summary
- One-paragraph overview: dominant failing op types, how many issues, anything
  notable (e.g. all failures isolated to certain devices).

## Failing Ops
> Sorted worst-first by min PCC.

### 1. `ttnn.softmax` — softmax over dim 3, 16×16×16×128 f32  (min PCC 0.005)
| | |
|---|---|
| SSAs | `%470`, `%605`, … (grouped — same shape/config) |
| Program / binary | `main` / binary_id 1 |
| Input → output | `tensor<16x16x16x128xf32>` → `tensor<16x16x16x128xf32>` |
| Config | `dimension=3`, `math_fidelity=hifi4`, `numericStable=true` |
| PCC (min / max) | 0.005 (dev 17) / 0.188 (dev 20) |
| atol (min / max) | 0.99 / 0.97 |
| Devices failing | 32 |

**Graph lines**
```
%470 = "ttnn.softmax"(%469) <{...}> : (tensor<16x16x16x128xf32, ...>) -> ...
%605 = "ttnn.softmax"(%604) <{...}> : (tensor<16x16x16x128xf32, ...>) -> ...
```

**Suspected cause:** Reduction along dim 3 — likely softmax accumulation
precision; verify fp32 dest accumulation and `numericStable` are effective.

### 2. … (next issue)

## Non-Numerics / Harness Issues
> Shape/dtype/chisel-bug records — compile/harness problems, not numerics drift.

| Op | Check | Status | Count |
|----|-------|--------|-------|
| `ttnn.to_device` | mlir_vs_runtime_tensor | shape_mismatch | 4 |

## Drill down
Ask me to dig deeper on any op (by SSA or name) and I'll trace its inputs through
the graph, compare isolated vs accumulated PCC, or inspect per-device behavior.
For an op whose kernel looks buggy (isolated-mode divergence), I can also generate
a standalone tt-mlir repro test for you to run.
```

## Notes
- Keep the human-facing markdown free of raw JSON — use tables and short prose.
- Be honest about gaps: if a binary_id had no graph, say shapes/config are
  unenriched for those ops rather than guessing.
- Stay shallow by default. The "Drill down" section is the contract — depth comes
  only when the user asks for a specific op.
