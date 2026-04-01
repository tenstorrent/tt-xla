---
allowed-tools: Bash(git diff:*), Bash(git log:*), Bash(git branch:*), Bash(git push:*), Bash(git status:*), Bash(gh pr:*), Bash(gh issue:*), Bash(gh label:*), Bash(gh api:*), Read, Grep
description: Create a tt-xla GitHub issue + PR with proper title, body template, CODEOWNERS-derived reviewers, and labels
argument-hint: "[area-prefix] (optional: e.g. 'vLLM', 'CI', 'pjrt' — auto-detected if omitted)"
---

## tt-xla PR Creation

You are creating a GitHub issue and pull request for the tt-xla repository. Work through each step carefully.

### Step 1 — Pre-flight Checks

Verify we are NOT on `main`:
```bash
git branch --show-current
```
If on `main`, stop and ask the user to switch to a feature branch.

Check for uncommitted changes:
```bash
git status --short
```
Warn if there are unstaged changes that might be missing from the PR.

Show the commits that will be in the PR:
```bash
git log main..HEAD --oneline
```
If there are no commits ahead of main, stop — nothing to PR.

---

### Step 2 — Create GitHub Issue

Before creating the PR, create a GitHub issue to document the problem and provide a ticket link.

Ask the user: "Do you have an existing GitHub issue to link? (enter number or URL, or press Enter to create one now)"

**If creating a new issue:**

Build the issue body from the commit log and diff:
```bash
git log main..HEAD --format='%s%n%b' | head -40
git diff main...HEAD --stat
```

Create the issue:
```bash
gh issue create \
  --title "<same title as the PR, without the [Area] prefix if it reads naturally>" \
  --body "$(cat <<'EOF'
## Problem
<!-- Why is this change needed? What is broken or missing? -->
<summarize from commit messages and diff>

## Proposed Solution
<!-- What approach does this PR take? -->
<summarize from diff>

## Files Changed
<git diff --stat output>
EOF
)" \
  --repo tenstorrent/tt-xla
```

Save the returned issue URL/number — it will be linked in the PR body as the Ticket.

**If the user provides an existing issue number:** use it directly (format: `https://github.com/tenstorrent/tt-xla/issues/<N>`).

---

### Step 3 — Detect Area and Generate PR Title

Read the changed paths:
```bash
git diff main...HEAD --name-only
```

If `$ARGUMENTS` provides an area prefix (e.g. "vLLM", "CI"), use it directly.
Otherwise, auto-detect using the path lookup table from `.claude/commit-template.md`:

| Path pattern | Prefix |
|---|---|
| `tests/integrations/vllm_plugin/` or `integrations/vllm_plugin/` | `[vLLM plugin]` |
| `.github/` | `[CI]` |
| `tests/infra/` or `tests/runner/` or `pytest.ini` | `[Test Infra]` |
| `pjrt_implementation/` | `[pjrt]` |
| `python_package/tt_torch/` (fusion) | `[FX fusing]` |
| `python_package/` or `CMakeLists.txt` | `[build]` |
| `scripts/` | `[tools]` |
| `tests/` (other) | `[test]` |
| Multiple areas or no clear match | no prefix, bare verb |

Read the commit log to understand the change:
```bash
git log main..HEAD --format='%s%n%b' | head -40
```

Generate a PR title following the convention from `.claude/commit-template.md`:
- Format: `[Area] Short imperative description`
- ≤ 72 characters, sentence-case, no trailing period
- No `feat:` / `fix:` prefixes

**Present the proposed title to the user and ask for confirmation before proceeding.**

---

### Step 4 — Generate PR Body

Read the PR body template from `.claude/pr-body-template.md`.

Auto-fill the sections:

**Ticket:** Use the issue URL from Step 2 (e.g. `https://github.com/tenstorrent/tt-xla/issues/4047`).

**Problem description:** Summarize *why* this change is needed, inferred from commit messages and diff.

**What's changed:** Pull from `git log main..HEAD --format='%s%n%b'` and the diff summary:
```bash
git diff main...HEAD --stat
```

**Testing:** Based on changed paths, suggest the relevant test command:
- Changes in `pjrt_implementation/` or `python_package/tt_jax/` → `pytest -v tests/jax/single_chip`
- Changes in `python_package/tt_torch/` or `tests/torch/` → `pytest -v tests/torch -m single_device`
- Changes in `tests/integrations/vllm_plugin/` → suggest vLLM integration tests
- CI / build changes → `pre-commit run --all-files`

**Checklist:** Pre-tick based on what you can verify:
- SPDX headers: `git diff main...HEAD --name-only --diff-filter=A | xargs grep -L 'SPDX-License-Identifier' 2>/dev/null`
- Tests present: check if changed source files have corresponding test files

---

### Step 5 — Determine Reviewers from CODEOWNERS

Read CODEOWNERS:
```bash
cat .github/CODEOWNERS
```

For each changed path, find the matching CODEOWNERS entry (most specific match wins).
Collect all owner handles, deduplicate, strip the `@` prefix.

Always include at least one of the global owners: `mrakitaTT`, `nvukobratTT`, `AleksKnezevic`.

Build the `--reviewer` list (max 15 reviewers for GitHub).

---

### Step 6 — Determine Labels

Map the detected area prefix to a GitHub label:

| Area | Label |
|---|---|
| `[CI]` | `ci` |
| `[vLLM plugin]` or `[vLLM]` | `vllm` |
| `[pjrt]` | `pjrt` |
| `[test]` or `[Test Infra]` | `testing` |
| `[build]` | `build` |
| `[FX fusing]` | `fx-fusing` |
| `Uplift third_party/tt-mlir` | `uplift-mlir` |
| `Uplift third_party/tt_forge_models` | `uplift-forge-models` |

Check that the label exists on the repo:
```bash
gh label list --json name | python3 -c "import sys,json; labels=[l['name'] for l in json.load(sys.stdin)]; print('\n'.join(labels))"
```

If a needed label does not exist, create it:
```bash
gh label create <label> --color "0075ca" --description "<description>"
```

---

### Step 7 — Push and Create PR

If the branch has not been pushed yet:
```bash
git push -u origin $(git branch --show-current)
```

Create the PR:
```bash
gh pr create \
  --title "<generated title>" \
  --body "$(cat <<'EOF'
<generated body>
EOF
)" \
  --reviewer <reviewer1>,<reviewer2>,... \
  --label <label> \
  --base main
```

---

### Step 8 — Post-Creation

After the PR is created:
1. Print the PR URL.
2. Run a quick CI status check (wait ~15 seconds for workflows to trigger):
   ```bash
   sleep 15 && gh pr checks <PR_NUMBER>
   ```
3. Report which CI jobs have started.

Remind the user they can run `/tt-xla-dev:ci-review <PR_NUMBER>` at any time for a full CI summary.
