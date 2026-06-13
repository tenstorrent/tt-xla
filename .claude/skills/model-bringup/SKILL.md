---
name: model-bringup
description: E2E model bringup pipeline orchestrator for Tenstorrent hardware. Drives the full VALIDATE → FIRST_RUN → DIAGNOSE → REPAIR → VERIFY → CONFIG_UPDATE FSM for bringing up a new model in tt-forge-models. Use when the user says "bringup <model>", invokes /model-bringup, or wants to run the full bringup pipeline on a model.
allowed-tools: Bash Read Write Edit Grep Glob Task Agent
---

# Model Bringup Pipeline — Orchestrator

You are the E2E Model Bringup Pipeline orchestrator for Tenstorrent hardware.

## Invocation
`/model-bringup [model_key] [--mode auto|bringup|retriage] [--arch <arch>] [--resume]`

Examples:
- `/model-bringup ltx2/pytorch-Fast-single_device-inference --arch n150`
- `/model-bringup perceiverio_vision/pytorch-Vision_Perceiver_Conv-single_device-inference --mode retriage`
- `/model-bringup` (no key → list smallest XFAIL candidates and exit)

## Argument Parsing
Parse `$ARGUMENTS`:
- `model_key` (optional): first positional argument.
  - If omitted, see **No-key behavior** below.
- `--mode` (optional, default `bringup`):
  - `bringup` — the existing new-model FSM (VALIDATE → FIRST_RUN → …).
  - `retriage` — the XFAIL re-triage FSM (see **XFAIL Re-triage Mode** below).
    Use this when the entry already exists in the YAML with status
    `KNOWN_FAILURE_XFAIL` and you want to check whether it still reproduces.
- `--arch` (optional, default `n150`): target architecture
- `--resume` (optional flag): resume from existing state.json instead of starting fresh

## No-key behavior

If `model_key` is omitted, do **not** run the FSM. Instead:
1. Invoke the `failure_summary` skill internally to generate the digest
   (target arch from `--arch`).
2. Print the **Quick-pick: 3 smallest models** section directly to the
   terminal so the user can copy a candidate key for the next invocation.
3. Print a one-line hint:
   ```
   Run: /model-bringup <model_key> --mode retriage  to re-verify an XFAIL,
        /model-bringup <model_key>                  to bring up a new model.
   ```
4. Exit. Do not create any bringup state.

## State Location
All state lives at `.claude/bringup/<model_key_with_slashes_replaced_by_double_underscore>/state.json`.

## Startup
1. If `--resume` is set and `state.json` exists, load it and resume from the recorded stage.
2. Otherwise, if `state.json` exists, ask the user whether to resume or restart.
3. If no state exists, create a new one by invoking the `model-bringup-scaffold` skill
   (bringup mode only — retriage mode skips scaffold; see below).

## XFAIL Re-triage Mode (`--mode retriage`)

This mode is for entries that already exist in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml`
with status `KNOWN_FAILURE_XFAIL`. The goal is to determine whether the
recorded failure still reproduces against current code.

### Entry gate
Before doing anything else, verify the entry's current status in the YAML
(top-level `status` OR `arch_overrides.<arch>.status`). If it is **not**
`KNOWN_FAILURE_XFAIL`:
- Print `[bringup] --mode retriage requires a KNOWN_FAILURE_XFAIL entry; <key> is currently <status>.`
- Suggest dropping `--mode retriage` to use the standard bringup flow.
- Exit.

### Skip scaffold
The loader already exists (the entry is in the YAML, so it has a node id).
Do **not** invoke `model-bringup-scaffold`. Initialize a minimal `state.json`
at `.claude/bringup/<safe_key>/state.json` with `mode: "retriage"` and an
empty `history` array.

### Single delegated step
Invoke the `model_issue_pick` skill with `<model_key> --arch <arch>`. It will
run the pytest with `--runxfail` in the background, classify the log, and
return a verdict. The verdicts and routing:

| Verdict from model_issue_pick | Next action |
|---|---|
| `now_passing`           | Invoke `model-bringup-config-update` with `result=PASSED`. Promote `KNOWN_FAILURE_XFAIL` → `EXPECTED_PASSING`; drop `reason`. Done. |
| `now_incorrect_result`  | Invoke `model-bringup-config-update` with `result=PASSED` AND pass through the recorded PCC so config-update can add `assert_pcc: false` + a lowered `required_pcc`. Done. |
| `xfail_same`            | The recorded failure still reproduces. **Fall into the standard pipeline at DIAGNOSE** with the new run.log so the orchestrator can attempt a fix via REPAIR → VERIFY. The first iteration counts as iteration 1. |
| `xfail_changed`         | The failure is real but different from the YAML's recorded reason. **Fall into the standard pipeline at DIAGNOSE** with the new run.log, same as `xfail_same`. The first iteration counts as iteration 1. |
| `timeout`               | No YAML change. Append a `retriage` history entry with `result=timeout`. Exit. (Consistent with the rule that automated timeouts are not evidence.) |
| `runner_error`          | No YAML change. Surface the error and exit ESCALATED. |

The `xfail_same` vs `xfail_changed` distinction is informational only — it
controls the **final** YAML write at CONFIG_UPDATE time, not the routing:

- If `xfail_same` and the pipeline ends in PASSED → config-update promotes to
  `EXPECTED_PASSING` and drops `reason` (same as `now_passing` path).
- If `xfail_changed` and the pipeline ends in ESCALATED → config-update keeps
  status as `KNOWN_FAILURE_XFAIL` but rewrites `reason` to reflect the new
  failure (the recorded reason is stale).
- If `xfail_same` and the pipeline ends in ESCALATED → leave the YAML
  untouched; the existing reason is still accurate.

Stash the verdict (`xfail_same` | `xfail_changed`) in
`state.retriage_verdict` so CONFIG_UPDATE can read it when the loop ends.

### Logging
Open `.claude/bringup/<safe_key>/bringup_steps.txt` with the same header
block as standard bringup, but tag the mode:
```
================================================================================
MODEL BRINGUP LOG (mode: retriage)
================================================================================
```
Append one step section for the `model_issue_pick` invocation, then a step
section per fall-through stage if `xfail_changed` routed into DIAGNOSE.

## FSM Loop
Run the following loop (max 5 iterations total across repair cycles).
In `--mode retriage`, skip the loop entirely for `now_passing`,
`now_incorrect_result`, `timeout`, and `runner_error` verdicts. For both
`xfail_same` and `xfail_changed` verdicts, enter the loop at **DIAGNOSE**
with the run.log produced by `model_issue_pick`.

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
- **Provenance block** at the top:
  ```
  tt-xla SHA       : <short sha of tt-xla HEAD>
  tt-foundry SHA   : <short sha if submodule present, else 'not a submodule'>
  Generated        : <YYYY-MM-DD HH:MM>
  Source skill     : model-bringup (orchestrator)
  Mode             : bringup | retriage
  ```
- model_key, arch, total iterations
- Each iteration: stage, diagnosis (with `source: json_report|stdout_fallback`),
  repair attempted, links to `iter_<N>_run.log` and `iter_<N>_result.json`
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
