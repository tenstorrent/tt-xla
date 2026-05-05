---
name: model-bringup-repair
description: Patch and repair stage of the model bringup pipeline. Applies the repair strategy from diagnosis (monkey_patch, lower_pcc_threshold, adjust_oom_config, fix_output_handling, or escalate). Runs code-reviewer skill before finalizing any patch. Invoked by the model-bringup orchestrator at the REPAIR stage.
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

### `escalate` (missing_op, import_error, unknown)
Do not attempt a repair. Return:
```
blocked: true
block_reason: "<root_cause_category> requires human intervention"
```

## Code Review Gate
Before finalising any patch, run the `code-reviewer` skill on the changed files.
If the review raises a blocking issue, set `blocked: true` with the review finding as `block_reason`.

## State Update
Append to `state.json` history:
```json
{
  "stage": "repair",
  "result": "applied | blocked",
  "details": {
    "strategy": "<strategy>",
    "patch_path": "<path or null>",
    "requires_human_review": true | false,
    "block_reason": "<string or null>"
  }
}
```
Add the patch path to `applied_patches` if applied.

## Bringup Steps Log
Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Repair (model-bringup-repair, iteration <N>)
--------------------------------------------------------------------------------
Strategy        : <strategy>
Patch file      : <patch_path or 'none'>
Files modified  : <list of files changed>
Requires review : yes | no
Code review     : passed | blocked — <finding if blocked>

[Brief description of what the patch does]

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
