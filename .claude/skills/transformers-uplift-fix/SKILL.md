---
name: transformers-uplift-fix
description: Fix transformers compatibility regressions in tt-xla and tt-forge-models after the pinned version was bumped. Called by the transformers-uplift CI orchestrator with a scope (api-check, model-test-uplifts, model-perf-uplift) and a captured failure context. Edits source-level only — no monkey-patching, no shims, no git operations.
allowed-tools: Bash Read Grep Glob Write Edit
---

# Transformers Uplift Fix

You are fixing transformers compatibility issues that surfaced after uplifting
`transformers` to `TARGET_VERSION` in tt-xla.

## Instructions

1. Read the `FAILURES` file, then read the scope-specific instructions
   for the `SCOPE` env var in the **Scope** section below.
2. Check the transformers changelog between the `CURRENT_VERSION` and
   `TARGET_VERSION` env vars to help diagnose root causes:
   `https://github.com/huggingface/transformers/releases`.
3. Apply fixes at the SOURCE level.
4. When you identify a root cause, fix it everywhere it occurs in the
   repo — not just where the failure surfaced. Grep the repo for other
   files that need the same change and update them in the same pass.
5. If a failure cannot be fixed without violating a Rule, leave it alone
   and call it out clearly in your final message. The orchestrator
   surfaces unfixed failures in the eventual PR — silent half-fixes
   are worse than a clear "couldn't fix this".
6. Before exiting, write a summary of what you did to
   `.github/transformers-uplift/fix-summary.md` using the template in
   **Output format** below. The orchestrator uses this file as the
   commit message body for your edits.

## Rules

- Under `third_party/` (these are external submodules), ONLY apply fixes
  to `third_party/tt_forge_models`.
- Do NOT touch `venv/requirements-dev.txt`. The transformers pin is
  already at the target version and the orchestrator owns that file.
- Do NOT `git add`, commit, push, branch, or open PRs. Leave the working
  tree dirty with your edits; the orchestrator owns every git step.
- No monkey-patching of `transformers`. No `if version < X` shims for
  older versions. The whole repo is moving to `TARGET_VERSION` and
  staying there — backward compatibility is not a goal.
- No drive-by refactors. No "clean up" of unrelated code. Ignore
  warnings — focus only on the entries in `FAILURES`.

## Scope: api-check

The caller ran `pytest --collect-only` on the model test suite. Failures
are IMPORT-TIME errors raised when pytest tried to import a test module
(which in turn imports its loader, which in turn imports `transformers`).

Additional Instructions:

1. Find the failing import (pytest's collection error includes the file
   and line — read it).
2. Check `venv/lib/python*/site-packages/transformers/` to find if
   the symbol has been moved, removed or replaced. `grep -r 'class Renamed' venv/.../transformers/`
   is fast.
3. Update the import in the loader. Often it's a one-line change.

## Scope: model-test-uplifts

The caller ran model tests on real TT hardware via `call-test-uplift.yml`.

Additional Instructions:

1. Fixes can land in tt_forge_models loaders OR in tt-xla test
   infrastructure (`tests/torch/models/`, `tests/infra/`,
   `examples/pytorch/llama.py`). Don't assume one location — read the
   failing file path and edit there.
2. Check for removed kwargs and delete them. Add new required ones using the documented
   defaults from the transformers changelog.
3. If a test fails purely on `pcc < required_pcc` (no exception, no
   shape mismatch), don't fight it — that's genuine numerical drift.
   Lower `required_pcc` in the test_config YAML or set
   `assert_pcc: false`, and call it out for human review.
4. For changed return shapes or renamed attributes (`Cache.key_cache`
   → `keys`, `outputs.vision_model_output` access patterns, etc.),
   update the call site to the new shape. Don't paper over it with
   `if isinstance(...)` — the new shape is the only one that matters now.
5. **Cache and attention utilities churn every uplift.** Expect
   breakage in `transformers.cache_utils` and in the attention API —
   check those first when failures look unfamiliar.

## Scope: model-perf-uplift

The caller ran the perf benchmark sweep via `call-perf-uplift.yml`. There are three
types of issues we want to focus on:
1. Broken benchmark test infrastructure.
2. Performance (samples/sec) regression beyond the configured threshold.
3. Missing expected fused TTNN ops.

Additional Instructions:

-  Focus on issues related to the transformers uplift.
-  **Benchmark infrastructure broken by transformers API changes.**
   The benchmark code itself raised an exception (TypeError, AttributeError,
   ImportError, etc.) because of changes in transformers that benchmark infra relies
   on. The failing file lives under `tests/benchmark/` (e.g.
   `tests/benchmark/benchmarks/llm_benchmark.py`,
   `tests/benchmark/llm_utils/decode_utils.py`, etc.).
-  **Performance Regression** DO NOT attempt blind fixes. Only act if
   you can attribute the regression to a specific transformers diff. Otherwise list it
   under Skipped for human review and add a one-line hypothesis to the entry's bullet
   under ## Skipped in fix-summary.md
-  **Missing Fusion** read the IR dump path from the failure, grep for the
   expected op, then check transformers diffs for the op pattern that
   produced it. Fix by adapting the model wrapper in tt-xla/tt-forge-models
   to restore the pattern. Skip if the fusion regression has no clear
   source-side cause.
-  Apply the same approach as `model-test-uplifts` — similar churn patterns
   (Cache API, attention API, return-shape renames, etc.).

## Output format

Before exiting, write `.github/transformers-uplift/fix-summary.md` with
this exact structure. Keep the subject line under 70 chars; keep each
bullet to one line. The orchestrator commits your edits using this file
as the commit body.

```markdown
transformers uplift: <scope> fixes — <one-line subject>

## Fixed
- <file/path>: <what changed and why>
- <file/path>: <what changed and why>

## Skipped (left for human review)
- <test or file>: <reason — Rule violation, ambiguity, PCC drift, etc.>

## Stats
- Failures input: <N>
- Fixed: <M>
- Skipped: <K>
```

Notes:
- The first line is the commit subject — keep it short and specific
  (`"transformers uplift: model-test-uplifts fixes — Cache.key_cache → keys"`,
  not `"applied fixes"`).
- If there's nothing under **Skipped**, omit the section entirely.
- If you exit early (timeout, blocked by a Rule on every failure),
  still write the file with whatever you did. The orchestrator falls
  back to a generic commit message only when the file is missing.
