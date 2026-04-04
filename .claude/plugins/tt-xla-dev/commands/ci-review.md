---
allowed-tools: Bash(gh pr:*), Bash(gh run:*), Bash(gh api:*), Bash(git branch:*), Bash(git log:*)
description: Query and summarize GitHub CI results for a tt-xla PR — shows job status, failure logs, and merge readiness
argument-hint: "[pr-number] (default: current branch's open PR)"
---

## tt-xla CI Review

You are summarizing the GitHub CI status for a tt-xla pull request.

### Step 1 — Resolve the PR

If a PR number was given as `$ARGUMENTS`, use it directly.

Otherwise, resolve from the current branch:
```bash
gh pr view --json number,title,state,headRefName,url
```

If no open PR exists for the current branch, report that and stop.

---

### Step 2 — Fetch Check Status

```bash
gh pr checks <PR_NUMBER> --json name,state,conclusion,startedAt,completedAt,detailsUrl
```

Also get the overall PR status:
```bash
gh pr view <PR_NUMBER> --json mergeable,mergeStateStatus,reviewDecision,statusCheckRollup
```

---

### Step 3 — Categorize by Known tt-xla CI Jobs

The tt-xla `pr-main.yml` workflow defines these jobs. Map each check result to its category:

| Job name pattern | Category | Blocking? |
|---|---|---|
| `pre-commit` | Lint gate | Yes |
| `inspect-changes` | Change detection | No |
| `build-ttxla` / `build-ttxla-debug` | Build | Yes |
| `test` / `basic-test` | Push test suite | Yes |
| `test_forge_models_push` | Model tests | Yes |
| `build-docs` | Documentation build | No |
| `check-all-green` | Merge sentinel | Yes |

---

### Step 4 — For Each Failing Job, Fetch Log Tail

```bash
# Get the run ID from the detailsUrl or:
gh run list --branch <branch> --json databaseId,name,status,conclusion | head -5

# Fetch failed job logs
gh run view <RUN_ID> --log-failed 2>&1 | tail -150
```

For each failure, identify:
- The specific error message / stack trace
- Which file or test caused the failure
- The likely root cause (CMake error, test assertion, lint violation, etc.)

---

### Step 5 — Output Structured Summary

```
## CI Review: PR #<N> — <title>

**Branch:** <branch>  **URL:** <url>

### Overall: PASS / FAIL / IN PROGRESS / WAITING

| Job | Status | Duration | Notes |
|-----|--------|----------|-------|
| pre-commit | PASS | 1m23s | |
| build-ttxla | FAIL | 8m12s | CMake error |
| test | IN PROGRESS | — | |
| check-all-green | FAIL | — | blocked by build-ttxla |

### Failing Jobs

#### <job-name>
**Status:** FAIL  **Run:** <url>

```
<log excerpt — last 30-50 lines of failure>
```

**Likely cause:** <brief diagnosis>
**Suggested fix:** <concrete next step>

---

### Merge Readiness

- Required checks green: yes / no
- `check-all-green`: PASS / FAIL / pending
- Review decision: APPROVED / REVIEW_REQUIRED / CHANGES_REQUESTED
- Mergeable: yes / no / CONFLICTING

**Verdict:** READY TO MERGE / BLOCKED (reasons)
```
