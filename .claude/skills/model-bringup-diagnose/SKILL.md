---
name: model-bringup-diagnose
description: Root cause diagnosis stage of the model bringup pipeline. Reads a pytest log, matches failure patterns (graph_break, oom, pcc_low, missing_op, runtime_mismatch, import_error), and outputs a structured JSON diagnosis with confidence level. Invoked by the model-bringup orchestrator at the DIAGNOSE stage.
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

## Required Output
Print the diagnosis as JSON conforming exactly to this schema:
```json
{
  "root_cause_category": "<value>",
  "suggested_repair_strategy": "<value>",
  "confidence": "high | medium | low"
}
```

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
  "details": { "confidence": "<level>", "strategy": "<strategy>", "log": "<log_path>" }
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
Confidence    : <level>
Excerpt       :
  <first 3 lines of the matching log region>

DIAGNOSE RESULT: <category>:<confidence>
```
