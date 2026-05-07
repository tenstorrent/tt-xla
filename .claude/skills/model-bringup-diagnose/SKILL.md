---
name: model-bringup-diagnose
description: Root cause diagnosis stage of the model bringup pipeline. Reads a pytest log, matches failure patterns (graph_break, oom, pcc_low, missing_op, runtime_mismatch, import_error), and outputs a structured JSON diagnosis with confidence level. Can recommend the runtime-failure-debugger skill as the repair strategy when standard pattern-matching is insufficient. Invoked by the model-bringup orchestrator at the DIAGNOSE stage.
allowed-tools: Read Bash
---

# Model Bringup — Root Cause Diagnosis

You are the **root cause diagnosis** stage of the model bringup pipeline.

## Invocation
`/model-bringup-diagnose <log_path> [--iteration <N>]`

## Responsibility
Read the pytest log and produce a structured diagnosis that the repair stage can act on.

## Failure Pattern Matching
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

## Escalating to runtime-failure-debugger
The default strategy table above maps each pattern to a cheap, one-shot fix.
When that is unlikely to help, override `suggested_repair_strategy` to
`runtime_debug` so the repair stage delegates to the `runtime-failure-debugger`
skill (systematic op-level bisect → minimal sanity → TTNN repro → tt-metal
replication).

Set `runtime_debug` when **any** of these hold:
1. Confidence would otherwise be `low` AND the failure is a runtime fault
   (`oom`, `pcc_low`, `runtime_mismatch`, or `unknown` with a Python/runtime
   traceback) — i.e. there is real signal to debug, just not enough to pick
   a one-shot fix.
2. `state.json` history shows the previous iteration already applied the
   default strategy for this same `root_cause_category` without resolving it
   (e.g. `adjust_oom_config` was tried and the new log shows the same OOM
   byte-count, or `lower_pcc_threshold` was tried and PCC is still failing).
   Read prior `repair` entries before deciding.
3. The matched pattern is `oom` or `pcc_low` and the log indicates the failure
   is op-config-driven rather than input-driven (e.g. byte-identical L1 budget
   regardless of input size, or PCC drop on a single intermediate tensor).

Do **not** use `runtime_debug` for `missing_op` or `import_error` — those
still escalate to a human (the debugger cannot synthesise a missing op).

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
Set `escalation_skill` to `"runtime-failure-debugger"` whenever
`suggested_repair_strategy` is `runtime_debug`; otherwise `null`.

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
    "log": "<log_path>"
  }
}
```

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
