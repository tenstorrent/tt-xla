---
name: transformers-uplift
description: Analyzes, debugs and proposes fixes for changes in transformers APIs to keep tt-xla up to date with latest version.
allowed-tools: Bash Read Grep Glob Write Edit
---

# Transformers Uplift

You are fixing transformers compatibility issues in the tt-xla project.

The HuggingFace transformers library was updated from the pinned version to the latest,
and some model tests broke.

## Instructions

1. Run the test script. If CI flags were provided in the prompt, pass them on the **first run only**:
   ```bash
   .claude/skills/transformers-uplift/scripts/run-model-tests.sh [CI flags]
   ```
2. If all tests pass, you're done — no fixes needed.
3. If tests fail, analyze `failure_context.txt` to identify which transformers API changes caused them.
4. Your changes do not need to be backward-compatible with older transformers versions.
   The whole repo is going to be updated to the latest transformers version.
5. You may modify any Python files in the repo as needed to fix the compatibility issues
   (e.g. update imports, fix class references, etc.).
6. Address any API changes at the source level. If an old API is no longer supported,
   we don't want monkey patching or workarounds — fix the issues at the source level.
   - Make the necessary changes to `third_party/tt_forge_models` to fix the compatibility issues.
7. After making fixes, re-run the tests:
   ```bash
   .claude/skills/transformers-uplift/scripts/run-model-tests.sh [CI flags]
   ```
   - If it still fails, analyze the error in `failure_context.txt` and apply instructions 3–6 again.
8. Finish your task after all tests pass, OR you've run the test command 4 times and still see failures.

## Rules

- DO NOT add, commit or push any your changes changes, and do not open any PRs. Your task is only to apply fixes to the codebase
- Under `third_party/` (these are external submodules), ONLY apply fixes to `third_party/tt_forge_models`.
- Keep `venv/requirements-dev.txt` unchanged.
- Keep changes minimal and focused on compatibility.
