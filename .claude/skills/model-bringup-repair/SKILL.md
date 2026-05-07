---
name: model-bringup-repair
description: Patch and repair stage of the model bringup pipeline. Applies the repair strategy from diagnosis (monkey_patch, lower_pcc_threshold, adjust_oom_config, fix_output_handling, runtime_debug, or escalate). The runtime_debug strategy delegates to the runtime-failure-debugger skill for systematic op-level analysis. Runs code-reviewer skill before finalizing any patch. Invoked by the model-bringup orchestrator at the REPAIR stage.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — Patch & Repair

You are the **patch and repair** stage of the model bringup pipeline.

## Invocation
`/model-bringup-repair <model_key> --diagnosis '<JSON>' [--iteration <N>]`

The `--diagnosis` value is the JSON object output by `model-bringup-diagnose`.

## Responsibility
Apply the repair strategy indicated by the diagnosis. Write any patch to
`.claude/bringup/<safe_key>/patches/iter_<N>_<strategy>.py` (or `.txt` for config changes).

## Strategy Implementations

### `monkey_patch` (graph break)
1. Extract the unsupported op name from the log excerpt in the diagnosis.
2. Identify the exact call site in the model source or wrapper using grep.
3. Write a Python patch file that monkey-patches or wraps the problematic call
   so it does not trigger a graph break (e.g. mark it with `torch.compiler.disable`,
   replace it with a graph-break-safe equivalent, or remove the unsupported path).
4. Inject the import of the patch at the top of `model_utils.py` in the test directory.
5. Set `requires_human_review: true` — do not apply without confirmation.

### `lower_pcc_threshold`
1. Extract the measured PCC from the log (pattern: `PCC=<value>`).
2. Compute `suggested = max(0.90, measured_pcc - 0.01)`.
3. Locate the `pcc=` kwarg in the pytest fixture for this variant.
4. Edit the value in the test file directly.
5. Record the old and new values in the patch log.

### `adjust_oom_config`
Try strategies in order, stopping when one resolves the OOM:
1. Halve `DEFAULT_HEIGHT` and `DEFAULT_WIDTH` in `src/utils.py` (or the loader constants).
2. Reduce `DEFAULT_NUM_FRAMES` to 9 (minimum: 8k+1).
3. If still OOM after both: add `@pytest.mark.skip(reason=_OOM_SKIP_REASON)` to
   the failing variant and note it for escalation.

### `fix_output_handling`
1. Inspect the wrapper's `forward()` in `model_utils.py`.
2. Check if `output.sample`, `output[0]`, or a dict key is the correct extraction path.
3. Patch the wrapper to match the actual output structure.

### `runtime_debug` (delegates to runtime-failure-debugger)
Use when diagnosis sets `escalation_skill: "runtime-failure-debugger"` — i.e.
the failure is a runtime fault (OOM / L1 overflow / PCC drop / NaN PCC /
runtime exception) where the cheap one-shot strategies are unlikely to help
or have already been tried without progress.

This strategy **delegates** to the `runtime-failure-debugger` skill. Repair
itself does not edit any source files; it sets up arguments, invokes the
skill, and records the result.

#### Step 1 — Derive arguments

Compute the values that the debugger's Phase 0 ("Ask for Prerequisites")
expects, so it does not have to ask the user for them:

| Field | Source |
|---|---|
| `tt_xla_repo` | The current tt-xla working directory (parent of `.claude/`). |
| `tt_xla_branch` | `git -C <tt_xla_repo> branch --show-current` |
| `test_command` | The pytest invocation captured in the failing log header (`<log_path>` from the diagnosis). |
| `failure_type` | One of PCC drop / NaN PCC / OOM / L1 / other — derived from `root_cause_category` in the diagnosis (`oom` → OOM; `pcc_low` → PCC drop; `runtime_mismatch` → other). |
| `model_name` | `model_key.split("/")[0]` (e.g. `qwen_2_5_vl/pytorch-3B_Instruct-single_device-inference` → `qwen_2_5_vl`). This is what the debugger uses for `claude_logs_<model_name>/` and `tests/torch/ops/<model_name>/`. |
| `failing_log_path` | `<log_path>` from the diagnosis (passed so the debugger can extract failure type and op from it instead of running an extra pass). |

Items the debugger needs that **cannot** be auto-derived:
- `tt_metal_machine`, `tt_metal_repo`, `tt_metal_branch` (Phase 5).

If those three are absent, do **not** pre-fill — the debugger will pause on
its own Phase 0 and ask. The orchestrator should treat that pause as a
human-input gate (see `runtime_debug` notes in `model-bringup`).

#### Step 2 — Invoke the skill

Call `runtime-failure-debugger` (the skill's directory slug; note the
file's frontmatter currently declares `name: debug_runtime_failures`, but
the harness registers it under the directory name). Pass the derived fields
as a single, self-contained brief — the skill is autonomous from Phase 1
onward and runs the full pipeline (op identification → bisect → block
sanity → minimal sanity → TTNN codegen → tt-metal replication) without
prompting between phases.

#### Step 3 — Record artefacts

The debugger creates persistent files outside `.claude/bringup/`:
- `<tt_xla_repo>/claude_logs_<model_name>/` — per-phase logs and the final
  `debug_report.md`
- `<tt_xla_repo>/tests/torch/ops/<model_name>/test_<op>_sanity.py` — block
  sanity (Phase 3)
- `<tt_xla_repo>/tests/torch/ops/<model_name>/test_<op>_minimal_sanity.py`
  — minimal single-op sanity (Phase 3B)
- `<tt_xla_repo>/examples/pytorch/codegen/python/<model_name>_<op>_repro.py`
  — codegen driver (Phase 4)
- `<tt_xla_repo>/<model_name>_<op>_export/` — generated TTNN code (Phase 4)

Record the **debug report path** in the repair `details` as
`debug_report_path: "<tt_xla_repo>/claude_logs_<model_name>/debug_report.md"`.

Required: also record the **minimal sanity path** as
`sanity_test_path: "<tt_xla_repo>/tests/torch/ops/<model_name>/test_<op>_minimal_sanity.py"`
(falling back to the block sanity at `test_<op>_sanity.py` if Phase 3B did
not converge on a minimal). The orchestrator uses this path to gate VERIFY
— the full model test is only re-run after the sanity passes.

Optionally also list other sanity / codegen paths in `details.artifacts[]`
for human reference.

The debugger may transiently edit model files (loader.py, package files in
`venv/lib/.../transformers/...`) for bisect cuts and `logger.info`
instrumentation. Per its own rules it reverts those before phase
transitions; verify with `git status -s` after the skill returns. If
edits remain, surface that in the repair output as a follow-up cleanup
item — do not silently leave them.

#### Step 4 — Outcome mapping

| Debugger result | Repair result |
|---|---|
| Wrote `debug_report.md` with all phases (3, 3B, 4, 5) populated. | `applied: false`, `requires_human_review: true`. The report is the deliverable; a human applies the fix. |
| Phase 3 could not reproduce the failure within the 5-op chain (debug_report.md has Phases 3B/4/5 marked **skipped**). | `blocked: true` with `block_reason: "runtime-failure-debugger could not isolate the failure in a 5-op sanity — see debug_report.md"`. |
| Phase 5 reports a delta between TTNN repro and tt-metal (different result after fixes). | `requires_human_review: true`; record the delta in `details.tt_metal_delta`. |
| Skill aborted before producing `debug_report.md` (e.g. user-cancelled, environment error). | `blocked: true` with `block_reason: "runtime-failure-debugger aborted before producing a report"`. |

The Code Review Gate **does not apply** to `runtime_debug` because no
project source files are intentionally modified by this stage. (The
debugger's transient edits are a debugger-internal concern; verify cleanup,
do not re-review them.)

### `escalate` (missing_op, import_error, unknown)
Do not attempt a repair. Return:
```
blocked: true
block_reason: "<root_cause_category> requires human intervention"
```

## Code Review Gate
Before finalising any patch, run the `code-reviewer` skill on the changed files.
If the review raises a blocking issue, set `blocked: true` with the review finding as `block_reason`.

The gate is **skipped** for `runtime_debug` because that strategy modifies no
source files — it only delegates to the runtime-failure-debugger skill and
records the analysis artefact path.

## State Update
Append to `state.json` history:
```json
{
  "stage": "repair",
  "result": "applied | blocked",
  "details": {
    "strategy": "<strategy>",
    "patch_path": "<path or null>",
    "debug_report_path": "<path or null>",
    "sanity_test_path": "<path or null>",
    "requires_human_review": true | false,
    "block_reason": "<string or null>"
  }
}
```
Add the patch path to `applied_patches` if applied. For `runtime_debug`, leave
`patch_path: null` and populate both `debug_report_path` and
`sanity_test_path` with the paths written by the runtime-failure-debugger.
The orchestrator reads `sanity_test_path` to gate VERIFY (sanity passes
before the full model test re-runs).

## Bringup Steps Log
Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Repair (model-bringup-repair, iteration <N>)
--------------------------------------------------------------------------------
Strategy        : <strategy>
Patch file      : <patch_path or 'none'>
Debug report    : <debug_report_path or 'none'>     # runtime_debug only
Files modified  : <list of files changed>
Requires review : yes | no
Code review     : passed | blocked | n/a — <finding if blocked>

[Brief description of what the patch does — for runtime_debug,
 summarise the runtime-failure-debugger findings instead.]

REPAIR RESULT: applied | blocked | requires_review
[If blocked:]  block_reason: <reason>
```

## Output
```
[repair] iter=<N>  strategy=<strategy>
  patch: <path or 'none'>
  status: applied | blocked | requires_review
  <block_reason if blocked>
```
