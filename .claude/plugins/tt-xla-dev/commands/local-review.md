---
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(git status:*), Bash(git stash:*), Bash(pre-commit:*), Bash(black:*), Bash(python3:*), Bash(grep:*), Bash(find:*), Read, Glob, Grep
description: Review staged/branch changes before creating a PR — lint, diff analysis, test gap detection, and pre-filled PR checklist
argument-hint: "[--branch <base-branch>] (default: compares staged changes; with --branch compares HEAD vs base)"
---

## tt-xla Local PR Review

You are performing a pre-PR review of changes in the tt-xla repository. Work through the five phases below in order, reporting findings as you go.

### Setup

Determine review scope:
- If the argument `--branch <base>` was given: compare `git diff <base>...HEAD`
- Otherwise: compare staged changes via `git diff --staged`

Run:
```bash
git diff --staged --stat
# or if --branch was given:
git diff <base>...HEAD --stat
```

Also get the list of changed files:
```bash
git diff --staged --name-only
# or
git diff <base>...HEAD --name-only
```

---

### Phase 1 — Lint Gate

Run pre-commit on changed files:
```bash
pre-commit run --files $(git diff --staged --name-only | tr '\n' ' ')
```

If `pre-commit` is not available, fall back to:
```bash
# Python files
black --check $(git diff --staged --name-only | grep '\.py$' | tr '\n' ' ')
python3 -m isort --check-only $(git diff --staged --name-only | grep '\.py$' | tr '\n' ' ')

# C++ files
for f in $(git diff --staged --name-only | grep -E '\.(cpp|h|hpp)$'); do
  clang-format --dry-run -Werror "$f" 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done
```

Check SPDX headers on newly added files:
```bash
git diff --staged --name-only --diff-filter=A | grep -E '\.(cpp|h|hpp|py)$' | while read f; do
  grep -q 'SPDX-License-Identifier: Apache-2.0' "$f" || echo "MISSING SPDX: $f"
done
```

Report: PASS / FAIL with specific violations.

---

### Phase 2 — Diff Analysis

Read the full diff:
```bash
git diff --staged
# or git diff <base>...HEAD
```

For each changed file, note:
- **What changed**: brief functional summary (not just "lines added")
- **Risk flags** (highlight these):
  - File touches multiple CODEOWNERS areas (will need more reviewers)
  - TODO / FIXME / HACK / print statement / debug code added
  - Large function added (>50 lines) with no corresponding test change
  - Public API change (function signature, exported symbol) with no doc update

Parse `.github/CODEOWNERS` to identify which owner groups are affected by the changed paths:
```bash
cat .github/CODEOWNERS
```

Report which CODEOWNERS areas are touched and which owners will be auto-requested.

---

### Phase 3 — Test Coverage Check

For each changed **source** file (`.cpp`, `.h`, `.py` in non-test directories), check whether a corresponding test exists:

| Source path | Expected test location |
|---|---|
| `pjrt_implementation/src/` | `tests/jax/` or `tests/torch/` |
| `python_package/tt_torch/` | `tests/torch/` |
| `python_package/tt_jax/` | `tests/jax/` |
| `integrations/vllm_plugin/` | `tests/integrations/vllm_plugin/` |

```bash
# Example: find test files that reference the changed module
git diff --staged --name-only | grep -v '^tests/' | while read src; do
  base=$(basename "$src" | sed 's/\.[^.]*$//')
  found=$(find tests/ -name "*${base}*" -o -name "test_${base}*" 2>/dev/null | head -3)
  echo "Source: $src -> Tests: ${found:-NONE FOUND}"
done
```

Report any source files with no identifiable test coverage.

---

### Phase 4 — Commit Message Check

Check the staged commits (or branch commits) for convention compliance:
```bash
git log --oneline -10
# or for branch: git log <base>..HEAD --oneline
```

For each commit title:
- Verify `[Area]` prefix is from the known list (see `.claude/commit-template.md`) OR uses bare-verb style
- Verify title ≤ 72 characters
- Verify no `feat:` / `fix:` / `chore:` prefixes
- Verify imperative form

Report any violations.

---

### Phase 5 — Pre-PR Checklist Output

Produce a filled-in checklist based on findings from Phases 1-4. Use the template from `.claude/pr-body-template.md` and pre-tick items that passed.

```
## Pre-PR Review Summary

### Lint
- [x/] pre-commit / black / clang-format: PASS or FAIL (details)
- [x/] SPDX headers: all present / MISSING on: <files>

### Diff Analysis
- Changed areas: <list>
- CODEOWNERS affected: <owners>
- Risk flags: <none / list>

### Test Coverage
- Source files with tests: <n>/<total>
- Gaps: <list or "none">

### Commit Messages
- All commits follow convention: yes / no (violations: <list>)

### Suggested PR Checklist
- [x] New/existing tests provide coverage for changes
- [ ] pre-commit run --all-files passes
- [x] SPDX headers present on new files
- [ ] CLAUDE.md updated if needed
- [x] CODEOWNERS will auto-assign: <owners>

### Verdict
READY FOR PR / NEEDS FIXES (list blocking issues)
```
