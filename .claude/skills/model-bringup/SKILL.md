---
name: model-bringup
description: E2E model bringup pipeline orchestrator for Tenstorrent hardware. Drives the full VALIDATE → FIRST_RUN → DIAGNOSE → REPAIR → VERIFY → CONFIG_UPDATE FSM for bringing up a new model in tt-forge-models. Use when the user says "bringup <model>", invokes /model-bringup, or wants to run the full bringup pipeline on a model.
allowed-tools: Bash Read Write Edit Grep Glob Task Agent
---

# Model Bringup Pipeline — Orchestrator

You are the E2E Model Bringup Pipeline orchestrator for Tenstorrent hardware.

## Invocation
`/model-bringup <model_key> [--arch <arch>] [--resume]`

Example: `/model-bringup ltx2/pytorch-Fast-single_device-inference --arch n150`

## Argument Parsing
Parse `$ARGUMENTS`:
- `model_key` (required): first positional argument
- `--arch` (optional, default `n150`): target architecture
- `--resume` (optional flag): resume from existing state.json instead of starting fresh

## State Location
All state lives at `.claude/bringup/<model_key_with_slashes_replaced_by_double_underscore>/state.json`.

## Startup
1. If `--resume` is set and `state.json` exists, load it and resume from the recorded stage.
2. Otherwise, if `state.json` exists, ask the user whether to resume or restart.
3. If no state exists, create a new one by invoking the `model-bringup-scaffold` skill.

## FSM Loop
Run the following loop (max 5 iterations total across repair cycles):

```
VALIDATE  →  FIRST_RUN
                ↓ pass → CONFIG_UPDATE → PASSED
                ↓ timeout (300s) → MANUAL-RUN PAUSE  (handled inside model-bringup-run)
                                       ↓ user provides log, tail = passed → CONFIG_UPDATE → PASSED
                                       ↓ user provides log, tail = failed → DIAGNOSE
                                       ↓ user replies "skip" or log inconclusive → CONFIG_UPDATE(TIMEOUT) → STOP
                ↓ fail → DIAGNOSE
                            ↓ low confidence (iter >= 2) → ESCALATE
                            ↓ runtime_debug → REPAIR (delegate to runtime-failure-debugger)
                                                ↓ debug_report.md → PAUSE for human review
                                                ↓ user fix → VERIFY (sanity gate first)
                            ↓ → REPAIR
                                  ↓ blocked → ESCALATE
                                  ↓ → VERIFY
                                        ↓ sanity gate (runtime_debug only)
                                            ↓ sanity fails → DIAGNOSE (skip full model)
                                            ↓ sanity passes → run full model
                                        ↓ pass → CONFIG_UPDATE → PASSED
                                        ↓ fail → DIAGNOSE (next iteration)
                                        ↓ no progress → ESCALATE
```

### Stage Execution

**VALIDATE**: Invoke `model-bringup-scaffold` skill with the model_key. On failure → ESCALATED.

**FIRST_RUN / VERIFY**: Invoke `model-bringup-run` skill with model_key and arch.

For **VERIFY** specifically: if the previous repair stage was `runtime_debug`
and recorded a `sanity_test_path` in state, **gate the full-model run on the
sanity test passing first**. Sanity tests are single-op and complete in
seconds, so this avoids spending several minutes on a full-model rerun when
the candidate fix did not actually resolve the failing op.

Sanity-gate procedure (runtime_debug only):
1. Look up `sanity_test_path` from the most recent `repair` history entry.
2. Run `pytest -svv <sanity_test_path> 2>&1 | tee
   .claude/bringup/<safe_key>/logs/iter_<N>_sanity.log`.
3. If the sanity exits **non-zero** (fails or errors) → do **not** invoke the
   full model run. Save the sanity log path to state and transition to
   DIAGNOSE for the next iteration. Append a note to `failure_reasons`
   indicating the sanity gate did not pass (e.g. `sanity_failed:<exit_code>`).
4. If the sanity exits **zero** (passes) → proceed to invoke
   `model-bringup-run` for the full model as below.

For all other repair strategies, skip the sanity gate and invoke
`model-bringup-run` directly.

Then, regardless of how VERIFY arrives at the full-model invocation:
- On pass → transition to CONFIG_UPDATE.
- On timeout → the run skill pauses internally and asks the user for a manual
  longer-budget run. The orchestrator should not transition until the run skill
  returns a final verdict. The verdict can be:
    - `passed` (manual log shows pytest pass) → CONFIG_UPDATE.
    - `failed` (manual log shows pytest fail / traceback) → DIAGNOSE, with
      `details.source: "manual_run"` on the history entry so DIAGNOSE knows the
      log came from a longer-budget run.
    - `timeout` (user replied "skip" or manual log was still inconclusive) →
      CONFIG_UPDATE with result=TIMEOUT, then STOP.
- On fail → save log path to state, transition to DIAGNOSE.

**DIAGNOSE**: Invoke `model-bringup-diagnose` skill with the log from the last run.
- If confidence is `low` and iteration >= 2 → ESCALATED.
- If diagnose sets `escalation_skill: "runtime-failure-debugger"` (i.e.
  `suggested_repair_strategy: "runtime_debug"`) → REPAIR with that strategy.
  This path is taken when standard pattern-matching is insufficient or when a
  prior cheap strategy did not resolve the same root cause.
- Otherwise → REPAIR.

**REPAIR**: Invoke `model-bringup-repair` skill with diagnosis and model_key.
- If strategy is `runtime_debug`, repair delegates to the
  `runtime-failure-debugger` skill. Important characteristics of this path:
  - **Long-running**: the debugger does an architecture-print pass, several
    bisect runs (each a full pytest), a block-sanity run, a Phase 3B
    minimal-sanity bisect (more pytest runs), a Phase 4 codegen + TTNN run,
    and a Phase 5 tt-metal run. Budget tens of minutes to a few hours and do
    not enforce the FIRST_RUN/VERIFY 5-minute timeout on it.
  - **Human-input gate**: Phase 5 needs `tt_metal_machine`, `tt_metal_repo`,
    and `tt_metal_branch`. The orchestrator does not have these from
    bringup state. Before invoking, prompt the user for them (or accept
    "skip Phase 5" — the debugger still produces a useful report up to
    Phase 4). Pass the values through to the repair stage so it can
    pre-fill the debugger's Phase 0.
  - **No automated patch**: the deliverable is `debug_report.md` at
    `<tt_xla_repo>/claude_logs_<model_name>/debug_report.md` plus the block
    and minimal sanity files under `tests/torch/ops/<model_name>/`. Treat
    the repair result as `requires_human_review: true`: pause, surface the
    report and sanity paths, and wait for the user to either supply a fix
    (continue to VERIFY) or escalate.
  - **Cleanup check**: after the debugger returns, run `git status -s`. If
    the debugger left transient edits in model files (its own rules say it
    must revert), flag this in the orchestrator's pause message so the user
    can clean up before re-running.
- If blocked → ESCALATED.
- If requires_human_review → pause and show the generated patch/instructions, wait for user confirmation before continuing.
- On proceed → increment iteration, transition to VERIFY.

**CONFIG_UPDATE**: Invoke `model-bringup-config-update` skill with model_key and result.

**ESCALATE**: Generate `escalation_report.md` (see below), invoke `model-bringup-config-update` skill with result=ESCALATED.

## Escalation Conditions
Escalate immediately when any of the following is true:
- Iteration count reaches 5 with no PASSED result
- Diagnosis confidence is `low` after iteration 2
- Repair is `blocked`
- The same failure_reason repeats across two consecutive iterations (no progress)
- Scaffold/validate fails

## Escalation Report
Write `.claude/bringup/<model_key>/escalation_report.md` containing:
- model_key, arch, total iterations
- Each iteration: stage, diagnosis, repair attempted
- Final failure category and confidence
- Recommended next human action

## Progress Display
After each stage transition, print a one-line status:
`[model_key] stage=<STAGE> iteration=<N> → <result>`

## Bringup Steps Log
Maintain `.claude/bringup/<safe_key>/bringup_steps.txt` throughout the pipeline run.
Append one section per stage as each completes (do not write the whole file at the end).
The log is the human-readable audit trail and must survive partial runs.

Each section follows this template:
```
--------------------------------------------------------------------------------
STEP <N> — <Stage Name> (<skill name>)
--------------------------------------------------------------------------------
<key facts: what was done, what was decided, what was found>
<any commands run and their one-line result>
<files created or modified>
<stage result: PASSED | FAILED | TIMEOUT | ESCALATED>
```

Open the file with a header block when the pipeline starts:
```
================================================================================
MODEL BRINGUP LOG
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
================================================================================
```

Close it with a summary block when the pipeline ends:
```
================================================================================
FINAL RESULT
================================================================================
<✓|✗> <model_key> — <PASSED|ESCALATED> after <N> repair iteration(s)
  Loader created  : yes | no
  Applied patches : <list or 'none'>
  Duration        : <total seconds>s
  YAML entry      : <key added to YAML or 'none'>
================================================================================
```

## Terminal Output
On PASSED:
```
✓ <model_key> — PASSED after <N> iteration(s)
  Applied patches: <list or 'none'>
  Steps log: .claude/bringup/<safe_key>/bringup_steps.txt
```
On ESCALATED:
```
✗ <model_key> — ESCALATED
  Reason: <last failure_reason>
  Report: .claude/bringup/<model_key>/escalation_report.md
  Steps log: .claude/bringup/<safe_key>/bringup_steps.txt
```
