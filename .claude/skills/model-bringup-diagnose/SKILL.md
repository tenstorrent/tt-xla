---
name: model-bringup-diagnose
description: Root cause diagnosis stage of the model bringup pipeline. Reads a pytest log, matches failure patterns (graph_break, oom, pcc_low, missing_op, runtime_mismatch, import_error), and outputs a structured JSON diagnosis with confidence level. Can recommend the runtime-failure-debugger skill as the repair strategy when standard pattern-matching is insufficient. Invoked by the model-bringup orchestrator at the DIAGNOSE stage.
allowed-tools: Read Bash
---

# Model Bringup — Root Cause Diagnosis

You are the **root cause diagnosis** stage of the model bringup pipeline.

## Invocation
`/model-bringup-diagnose <log_path> [--json-report <path>] [--iteration <N>]`

## Responsibility
Read the pytest log (and structured JSON report if available) and produce a
diagnosis that the repair stage can act on.

## Input precedence

Always prefer the JSON report when present (model-bringup-run writes one by
default at `logs/iter_<N>_result.json`). If `--json-report` is supplied or
a file exists at the conventional path:

1. Load the JSON.
2. Read `tests[0].metadata.tags.bringup_status` — this is the runner's own
   categorized verdict (`FAILED_TTMLIR_COMPILATION`, `FAILED_RUNTIME`,
   `INCORRECT_RESULT`, etc.). Map it directly to `root_cause_category`
   without regex matching:
   - `FAILED_TTMLIR_COMPILATION` + traceback mentions `failed to legalize`
     → `missing_op`
   - `FAILED_TTMLIR_COMPILATION` + mentions `Can't convert shape rank` →
     `runtime_mismatch` (compile-time shape error)
   - `FAILED_RUNTIME` + log shows `Out of Memory` → `oom`
   - `INCORRECT_RESULT` → `pcc_low`
   - `FAILED_FE_COMPILATION` + Dynamo trace → `graph_break`
3. Read `tests[0].metadata.tags.failing_reason.description` for the
   one-line root cause string; copy it verbatim into the diagnosis output.
4. Read `tests[0].metadata.tags.pcc` / `pcc_threshold` for PCC cases —
   surface both numbers in the diagnosis details.

Fall through to the stdout regex table below **only** if the JSON report is
missing or malformed (record `details.json_report: "missing"` in state).

## Failure Pattern Matching (stdout fallback)
Scan the log for these patterns in order (first match wins):

| Root Cause | Indicators | Default Strategy |
|---|---|---|
| `graph_break` | `torch._dynamo.exc.Unsupported`, `graph break`, `Unsupported: ` | `monkey_patch` |
| `oom` | `DRAM OOM`, `Out of memory`, `Killed`, `ResourceExhaustedError` | `adjust_oom_config` |
| `pcc_low` | `PCC=<value> (required=`, `pcc.*FAIL`, `AssertionError.*pcc` | `lower_pcc_threshold` |
| `missing_op` | `NotImplementedError`, `not supported by the tt backend`, `Unsupported node: aten.` | `escalate` |
| `runtime_mismatch` | `shape mismatch`, `RuntimeError.*size`, `Expected.*got.*tensor` | `fix_output_handling` |
| `import_error` | `ImportError`, `ModuleNotFoundError` | `escalate` |
| `unknown` | none of the above | `escalate` |

## Confidence Rules
- `high`: pattern is unambiguous and unique in the log
- `medium`: pattern found but log contains multiple candidate causes
- `low`: no known pattern matched, or multiple conflicting patterns

## Escalating to a specialist skill

Two specialist skills can take over when the default one-shot strategies are
unlikely to resolve the failure. Pick at most one per diagnosis.

### `runtime-failure-debugger` (for runtime faults)
Override `suggested_repair_strategy: "runtime_debug"` and set
`escalation_skill: "runtime-failure-debugger"` when **any** of these hold:

1. Confidence would otherwise be `low` AND the failure is a runtime fault
   (`oom`, `pcc_low`, `runtime_mismatch`, or `unknown` with a Python/runtime
   traceback) — i.e. there is real signal to debug, just not enough to pick
   a one-shot fix.
2. `state.json` history shows the previous iteration already applied the
   default strategy for this same `root_cause_category` without resolving it
   (e.g. `adjust_oom_config` was tried and the new log shows the same OOM
   byte-count, or `lower_pcc_threshold` was tried and PCC is still failing).
3. The matched pattern is `oom` or `pcc_low` and the log indicates the
   failure is op-config-driven rather than input-driven (e.g. byte-identical
   L1 budget regardless of input size, or PCC drop on a single intermediate
   tensor).

Do **not** use `runtime_debug` for `missing_op` or `import_error` — those
still escalate to a human (the debugger cannot synthesise a missing op).

### `graph-break-analysis` (for compile-time / Dynamo breaks)
Override `suggested_repair_strategy: "graph_break_analysis"` and set
`escalation_skill: "graph-break-analysis"` when:

1. `root_cause_category` is `graph_break` AND any of:
   - The log shows **more than one** graph-break source (multi-break — a
     single `monkey_patch` won't cover them).
   - The break is from `dynamic_shape`, mutation, or control-flow guards
     (these need flag/region fixes, not a wrapped call).
   - `monkey_patch` was tried in a prior iteration and the same break
     persists.

Do **not** use `graph_break_analysis` for single, well-localized graph
breaks where the unsupported op is named in the traceback — `monkey_patch`
is the right tool there.

## Required Output
Print the diagnosis as JSON conforming exactly to this schema:
```json
{
  "root_cause_category": "<value>",
  "suggested_repair_strategy": "<value>",
  "confidence": "high | medium | low",
  "escalation_skill": "runtime-failure-debugger" | null
}
```
Set `escalation_skill` to:
- `"runtime-failure-debugger"` when `suggested_repair_strategy` is `runtime_debug`
- `"graph-break-analysis"` when `suggested_repair_strategy` is `graph_break_analysis`
- `null` otherwise

Then print a human-readable summary:
```
[diagnose] iter=<N>
  root_cause:  <category>
  strategy:    <strategy>
  confidence:  <level>
  excerpt:     <first 3 lines of the matching log region>
```

## State Update
Append to `state.json` history:
```json
{
  "stage": "diagnose",
  "result": "<category>",
  "details": {
    "confidence": "<level>",
    "strategy": "<strategy>",
    "escalation_skill": "<runtime-failure-debugger or null>",
    "log": "<log_path>",
    "json_report": "<path or 'missing'>",
    "source": "json_report | stdout_fallback",
    "tt_xla_sha": "<short sha of tt-xla HEAD>",
    "tt_foundry_sha": "<short sha if tt-foundry submodule is present, else null>"
  }
}
```

`source` records whether the verdict came from the structured JSON report
or from the stdout regex fallback — useful when triaging spurious
classifications later.

Also append `<category>:<confidence>` to `failure_reasons` in state.json.

## Bringup Steps Log
Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Diagnose (model-bringup-diagnose, iteration <N>)
--------------------------------------------------------------------------------
Log analysed  : <log_path>
Root cause    : <category>
Strategy      : <suggested_repair_strategy>
Escalation    : <runtime-failure-debugger or 'none'>
Confidence    : <level>
Excerpt       :
  <first 3 lines of the matching log region>

DIAGNOSE RESULT: <category>:<confidence>
```
