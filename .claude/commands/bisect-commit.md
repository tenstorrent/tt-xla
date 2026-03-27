# Bisect a test failure between two tt-xla commits

Given a regression JSON file (or explicit arguments), find the exact commit that introduced a test failure by:
1. Running the test on the bad commit to capture the error message
2. Git-bisecting between good and bad to pinpoint the first bad commit

Each test run: checks out the commit, sources venv, updates submodules, downloads and installs the wheel built by CI for that commit, then runs the test.

## Usage

```
/bisect-commit <regression_json_path>
/bisect-commit <test_id> <good_commit> <bad_commit>
```

**Arguments** (from `$ARGUMENTS`):

**Option A — JSON file (preferred):**
- `regression_json_path` — path to a regression JSON file, e.g. `investigation/regression_densenet_pytorch-169-single_device-inference.json`

The JSON file must contain:
```json
{
  "test_id": "densenet/pytorch-169-single_device-inference",
  "good_sha": "c77995f6c200de122b33bb5b408cc02dcc31dd65",
  "bad_sha":  "1e5781dc41ca608cc3d8c5f9aeeaea21e0b53706"
}
```
(`good_run_id`, `bad_run_id`, `good_run_date`, `bad_run_date` are optional extra context.)

**Option B — explicit arguments:**
- `test_id` — pytest test ID, e.g. `densenet/pytorch-169-single_device-inference`
- `good_commit` — the known-good commit SHA (full or short)
- `bad_commit` — the known-bad commit SHA (full or short)

**Examples:**
```
/bisect-commit investigation/regression_densenet_pytorch-169-single_device-inference.json
/bisect-commit densenet/pytorch-169-single_device-inference abc1234 def5678
```

---

## Instructions for Claude

Determine the repository root at the start:
```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
```

The GitHub repo is always `tenstorrent/tt-xla`. Use `GITHUB_REPO=tenstorrent/tt-xla` throughout.

Use `REPO_ROOT` throughout — never hardcode the local path.

---

### Phase 1 — Parse arguments and load inputs

Check `$ARGUMENTS`:

- If it ends in `.json` (or contains a path with `/`), treat it as **Option A**:
  - Read the file (resolve relative to `$REPO_ROOT` if not absolute)
  - Extract `test_id`, `good_sha`, `bad_sha` from the JSON
  - If `good_run_date` / `bad_run_date` are present, display them for context

- Otherwise treat as **Option B**: first token = `test_id`, second = `good_commit`, third = `bad_commit`

Resolve both SHAs to full 40-char hashes:
```bash
cd "$REPO_ROOT"
git rev-parse <good_sha>
git rev-parse <bad_sha>
```

Count commits in range (for estimating bisect steps):
```bash
git rev-list --count <good_sha>..<bad_sha>
```

Show the user:
```
test_id:       <test_id>
good:          <full_good_sha>  (<good_run_date if available>)
bad:           <full_bad_sha>   (<bad_run_date if available>)
commits in range: N
estimated bisect steps: ceil(log2(N))
```

---

### Phase 2 — Understand the CI failure before touching the chip

**Never run the test locally first without knowing what CI saw.** Running on a dirty chip produces false errors that lead bisect astray. Always establish the ground-truth failure from CI logs before any local execution.

#### 2a. Fetch the CI failure from GitHub Actions

If the JSON contains `bad_run_url` (e.g. `https://github.com/tenstorrent/tt-xla/actions/runs/<run_id>/job/<job_id>`), extract the job ID from the URL:

```bash
# Extract job_id from bad_run_url
JOB_ID=$(echo "<bad_run_url>" | grep -oP '(?<=/job/)\d+')

# Download job logs
gh api "repos/$GITHUB_REPO/actions/jobs/$JOB_ID/logs" > /tmp/ci_bad_job.log 2>&1 || true
```

If `bad_run_url` is not present but `bad_run_id` is, find the failed job for the test:
```bash
# List jobs for the run and find the one related to this test
gh api "repos/$GITHUB_REPO/actions/runs/<bad_run_id>/jobs?per_page=100" \
  --jq '.jobs[] | select(.conclusion == "failure") | {id: .id, name: .name}' | head -20
# Then download the relevant job log
gh api "repos/$GITHUB_REPO/actions/jobs/<job_id>/logs" > /tmp/ci_bad_job.log 2>&1 || true
```

Search the CI log for the test failure and extract the raw error:
```bash
grep -A 30 "<test_id>\|FAILED\|PASSED\|Error\|assert\|Timeout\|Aborted" \
  /tmp/ci_bad_job.log | head -60
```

Show the user the raw CI error output as-is. This is the reference error — do not interpret or classify it, just present it:
```
CI error (raw):
<paste of the relevant lines from the CI log>
```

This raw CI error is the ground truth. All subsequent steps (local reproduction and bisect) use this exact output as the reference.

#### 2b. Reset chip and reproduce locally

**Always reset the chip before running the test locally:**
```bash
tt-smi -r 2>/dev/null || true
sleep 3
```

Install the CI wheel for the bad commit (follow the wheel-install helper below), then:

```bash
cd "$REPO_ROOT"
git checkout <bad_sha>
source "$REPO_ROOT/venv/bin/activate"
git submodule update --init
```

Install the wheel, then run:
```bash
pytest "tests/runner/test_models.py::test_all_models_torch[<test_id>]" \
  --no-header -rN --timeout=300 \
  2>&1 | tee /tmp/bisect_bad_run.log
```

Extract the key error message:
```bash
grep -A 20 "FAILED\|AssertionError\|Error\|PCC\|assert\|Timeout\|Aborted" /tmp/bisect_bad_run.log | head -40
```

**Compare local output with the CI error:**

- If local output **matches** the CI error → confirmed reproduction. Proceed to Phase 3 using the CI error as the reference.
- If local result is **PASSED** → the failure may not reproduce in isolation. Stop and report to user.
- If local error is **different** from CI → chip state issue is likely. Reset chip again (`tt-smi -r`, sleep 3) and retry once. If still different, report both outputs to the user and ask how to proceed — do not bisect on a false error.

Report to the user:
```
CI error (raw):    <snippet from CI log>
Local error (raw): <snippet from local run>
Match:             YES / NO
```

If match is NO and the user wants to continue anyway, ask them to confirm which error to use as the bisect reference before proceeding to Phase 3.

---

### Phase 3 — Create the bisect run script

Determine the log directory — named after the test slug and short bad SHA so it's easy to find:
```
LOG_DIR=/tmp/bisect_<test_id_slug>_<short_bad_sha>
```
For example: `/tmp/bisect_densenet_pytorch-169-single_device-inference_1e5781d`

Create it:
```bash
mkdir -p "$LOG_DIR"
```

Each bisect step writes its full output to `$LOG_DIR/<short_commit>.log`. A running summary is appended to `$LOG_DIR/bisect_summary.log`.

Write `/tmp/bisect_commit_test.sh`:

```bash
#!/usr/bin/env bash
# Auto-generated by /bisect-commit skill
set -euo pipefail

REPO=__REPO_ROOT__
GITHUB_REPO=__GITHUB_REPO__
TEST_ID="__TEST_ID__"
LOG_DIR="__LOG_DIR__"
SUMMARY="$LOG_DIR/bisect_summary.log"

cd "$REPO"
COMMIT=$(git rev-parse HEAD)
SHORT=${COMMIT:0:7}
LOG="$LOG_DIR/$SHORT.log"

echo ""
echo "======================================================"
echo "=== BISECT STEP: $COMMIT ==="
echo "======================================================"

# ---- Activate venv ----
source "$REPO/venv/bin/activate"

# ---- Update submodules ----
echo "=== Updating submodules ==="
git submodule update --init 2>&1 | tail -5

# ---- Download and install CI wheel ----
echo "=== Finding CI wheel for $SHORT ==="

RUN_ID=$(gh api "repos/$GITHUB_REPO/actions/runs?head_sha=$COMMIT&event=push&per_page=1000" \
  --jq '.workflow_runs[] | select(.name == "On push") | .id' 2>/dev/null | head -1)

if [ -z "$RUN_ID" ]; then
  echo "No 'On push' CI run found for $COMMIT — skipping commit (exit 125)"
  exit 125
fi

echo "CI run ID: $RUN_ID"
WHEEL_DIR="/tmp/bisect_wheels/$SHORT"
rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"

gh run download "$RUN_ID" \
  --repo "$GITHUB_REPO" \
  --dir "$WHEEL_DIR" \
  --name "xla-whl-release-$SHORT" 2>&1

WHEEL=$(find "$WHEEL_DIR" -name "*.whl" | head -1)
if [ -z "$WHEEL" ]; then
  echo "Wheel not found in $WHEEL_DIR — artifact may have expired. Skipping commit (exit 125)"
  echo "$SHORT  SKIPPED  (no wheel)" >> "$SUMMARY"
  exit 125
fi

echo "Installing wheel: $WHEEL"
uv pip install "$WHEEL" --quiet

# ---- Reset chip before each test run ----
# This prevents a prior test leaving the chip in a bad state from causing false failures
echo "=== Resetting chip ==="
tt-smi -r 2>/dev/null || true
sleep 3

# ---- Run the test — full output saved to per-commit log ----
echo "=== Running test: $TEST_ID ==="
pytest "tests/runner/test_models.py::test_all_models_torch[$TEST_ID]" \
  --no-header -rN \
  --timeout=300 \
  2>&1 | tee "$LOG"

PYTEST_EXIT=${PIPESTATUS[0]}

# Skip if test wasn't collected (didn't exist yet at this commit)
if grep -q "no tests ran\|collected 0 items" "$LOG"; then
  echo "TEST NOT COLLECTED — skipping commit (exit 125)"
  echo "$SHORT  SKIPPED  (not collected)" >> "$SUMMARY"
  exit 125
fi

if [ $PYTEST_EXIT -eq 0 ]; then
  echo "TEST PASSED"
  echo "$SHORT  PASSED" >> "$SUMMARY"
  exit 0
else
  echo "TEST FAILED"
  echo "$SHORT  FAILED" >> "$SUMMARY"
  exit 1
fi
```

Substitute `__REPO_ROOT__` with `$REPO_ROOT`, `__GITHUB_REPO__` with `$GITHUB_REPO`, `__TEST_ID__` with the actual test_id value, and `__LOG_DIR__` with the log directory path, then:
```bash
chmod +x /tmp/bisect_commit_test.sh
```

---

### Phase 4 — Run git bisect

```bash
cd "$REPO_ROOT"
git bisect reset   # clean up any previous bisect state
git bisect start
git bisect bad <bad_sha>
git bisect good <good_sha>
git bisect run /tmp/bisect_commit_test.sh
```

Print each bisect step result for the user as it runs. Each step shows: commit SHA, result (PASSED / FAILED / skipped), and any error snippet.

When a step is **skipped** (`exit 125`), note the reason (no CI run, expired artifact, build failure, test not collected).

---

### Phase 5 — Report results

When `git bisect run` completes:

**5a. Show the first bad commit:**
```bash
git show --stat <first_bad_sha>
```

**5b. Find the associated PR:**
```bash
gh api "repos/$GITHUB_REPO/commits/<first_bad_sha>/pulls" --jq '.[].html_url'
```

**5c. Show the failure from the first-bad commit's log:**
```bash
grep -A 15 "FAILED\|AssertionError\|Error\|assert\|Timeout\|Aborted" "$LOG_DIR/<first_bad_short>.log" | head -30
```

**5d. Show the full bisect summary:**
```bash
cat "$LOG_DIR/bisect_summary.log"
```

**5e. Clean up:**
```bash
git bisect reset
```

**5f. Write findings to `investigation/bisect_<short_bad_sha>_<test_id_slug>.md`:**

```markdown
# Bisect: <test_id>

**Range:** <good_sha> → <bad_sha>
**First bad commit:** <first_bad_sha>
**PR:** <pr_url>
**Date:** <commit date>
**Logs:** `<LOG_DIR>/`

## CI error (reference)

<paste of the raw CI error from Phase 2>

## Bisect summary

<contents of $LOG_DIR/bisect_summary.log>
← annotate the first bad commit line

## Error on first bad commit

<paste of key lines from $LOG_DIR/<first_bad_short>.log>
```

---

### Wheel install helper

Use this procedure whenever you need to install the CI wheel for a given commit SHA:

```bash
COMMIT=<full_sha>
SHORT=${COMMIT:0:7}

# Find the "On push" run for this commit
RUN_ID=$(gh api "repos/$GITHUB_REPO/actions/runs?head_sha=$COMMIT&event=push&per_page=1000" \
  --jq '.workflow_runs[] | select(.name == "On push") | .id' | head -1)

echo "Run ID: $RUN_ID"

# Download the wheel artifact
WHEEL_DIR="wheels/$SHORT"
mkdir -p "$WHEEL_DIR"
gh run download "$RUN_ID" \
  --repo "$GITHUB_REPO" \
  --dir "$WHEEL_DIR" \
  --name "xla-whl-release-$SHORT"

# Install
WHEEL=$(find "$WHEEL_DIR" -name "*.whl" | head -1)
uv pip install "$WHEEL"
```

---

### Error handling and recovery

| Situation | Action |
|-----------|--------|
| `gh api` returns no "On push" run for a commit | `exit 125` — skip commit (no CI wheel available) |
| Artifact download fails / artifact expired | `exit 125` — skip commit |
| Wheel installs but test not collected | `exit 125` — commit predates the test |
| Build from source fails | `exit 125` — skip commit |
| Test hangs beyond `--timeout=300` | Pytest exits non-zero → FAILED; investigate if it keeps happening |
| Chip error / TT device not responding | Run `tt-smi -r` to reset. If `tt-smi` is missing: `pip install tt-smi`, then retry |
| All commits skipped (no artifacts in range) | Report to user — the artifact retention window may have passed. Offer to build from source instead by modifying the bisect script |
| `git bisect` lands on a merge commit | Normal — bisect will find the merge commit if that's where the regression landed |
| Detached HEAD warnings | Expected during bisect — ignore |
