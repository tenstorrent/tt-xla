# Bisect a test failure between two tt-xla commits

Given a test ID, a known-bad commit, a known-good commit, and an error message, find the exact commit that introduced a failure by:
1. Running the test on the bad commit to reproduce the error
2. Running the test on the good commit to verify expected behaviour
3. Git-bisecting between good and bad to pinpoint the first bad commit

Each test run follows a fixed sequence: restart chip → source venv → checkout commit → update submodules → download and install CI wheel → run test. Full output of every run is saved as `<commit_hash>.log` inside the test's log directory.

## Usage

```
/bisect-commit test_id="<id>" first_bad_sha="<sha>" last_good_sha="<sha>" known_error="<error>" [expected_good_outcome="<outcome>"] [log_dir="<path>"]
```

**Arguments** (from `$ARGUMENTS`):
- `test_id` — pytest test ID, e.g. `inception/pytorch-v4_OSMR-data_parallel-inference`
- `first_bad_sha` — the known-bad commit SHA (full or short)
- `last_good_sha` — the known-good commit SHA (full or short)
- `known_error` — the error message observed on the bad commit (used as reference for bisect)
- `expected_good_outcome` — *(optional)* what to expect on the good commit; defaults to `"test passes"`
- `log_dir` — *(optional)* directory to write per-commit log files; defaults to `bisection/bisect/<test_id_slug>_<short_bad_sha>/`

**Examples:**
```
/bisect-commit test_id="inception/pytorch-v4_OSMR-data_parallel-inference" first_bad_sha="afe4486f" last_good_sha="1e5781dc" known_error="AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.97." log_dir="bisection/bisect/inception_pytorch-v4_OSMR-data_parallel-inference_n150"
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

### Phase 1 — Parse arguments

Parse `$ARGUMENTS` as key=value pairs (values may be quoted with `"` or `'`). Extract:
- `test_id`
- `first_bad_sha`
- `last_good_sha`
- `known_error`
- `expected_good_outcome` (default: `"test passes"`)
- `log_dir` (default: derive below)

Resolve both SHAs to full 40-char hashes:
```bash
cd "$REPO_ROOT"
git fetch --quiet
FULL_BAD=$(git rev-parse <first_bad_sha>)
FULL_GOOD=$(git rev-parse <last_good_sha>)
SHORT_BAD=${FULL_BAD:0:8}
SHORT_GOOD=${FULL_GOOD:0:8}
```

If `log_dir` was not provided, derive it:
```bash
# Replace / with _ in test_id to form a safe slug
TEST_SLUG=$(echo "<test_id>" | tr '/' '_' | tr ' ' '_')
LOG_DIR="$REPO_ROOT/bisection/bisect/${TEST_SLUG}_${SHORT_BAD}"
```
Otherwise resolve it relative to `$REPO_ROOT` if not absolute.

Create the log directory:
```bash
mkdir -p "$LOG_DIR"
```

Count commits in range:
```bash
git rev-list --count ${FULL_GOOD}..${FULL_BAD}
```

Print:
```
test_id:               <test_id>
first_bad_sha:         <FULL_BAD>
last_good_sha:         <FULL_GOOD>
known_error:           <known_error>
expected_good_outcome: <expected_good_outcome>
log_dir:               <LOG_DIR>
commits in range:      N
estimated bisect steps: ceil(log2(N))
```

---

### Run sequence helper

Every time a test must be executed at a specific commit, follow this exact sequence. This sequence is used in Phase 2, Phase 3, and the bisect script in Phase 4.

```bash
# Step 1 — Restart chip
echo "=== Restarting chip ==="
tt-smi -r 2>/dev/null || {
  echo "tt-smi not found or failed — reinstalling"
  deactivate 2>/dev/null || true
  pip uninstall tt-smi -y 2>/dev/null || true
  source "$REPO_ROOT/venv/bin/activate"
  pip install tt-smi
  tt-smi -r
}
sleep 3

# Step 2 — Activate venv
echo "=== Activating venv ==="
source "$REPO_ROOT/venv/bin/activate"

# Step 3 — Checkout the commit
echo "=== Checking out <COMMIT> ==="
cd "$REPO_ROOT"
git checkout <COMMIT>

# Step 4 — Update submodules
echo "=== Updating submodules ==="
git submodule update --init 2>&1 | tail -5

# Step 5 — Download CI wheel
echo "=== Finding CI wheel for <SHORT> ==="
RUN_ID=$(gh api "repos/$GITHUB_REPO/actions/runs?head_sha=<COMMIT>&event=push&per_page=1000" \
  --jq '.workflow_runs[] | select(.name == "On push") | .id' 2>/dev/null | head -1)
if [ -z "$RUN_ID" ]; then
  echo "No 'On push' CI run found for <COMMIT> — cannot install wheel"
  exit 1
fi
WHEEL_DIR="/tmp/bisect_wheels/<SHORT>"
rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"
gh run download "$RUN_ID" \
  --repo "$GITHUB_REPO" \
  --dir "$WHEEL_DIR" \
  --name "xla-whl-release-<SHORT>" 2>&1
WHEEL=$(find "$WHEEL_DIR" -name "*.whl" | head -1)
if [ -z "$WHEEL" ]; then
  echo "Wheel artifact not found or expired for <SHORT>"
  exit 1
fi

# Step 6 — Install wheel
echo "=== Installing wheel: $WHEEL ==="
uv pip install "$WHEEL" --quiet

# Step 7 — Run test, log to <SHORT>.log
LOG="$LOG_DIR/<SHORT>.log"
echo "=== Running test: $TEST_ID ==="
pytest "tests/runner/test_models.py::test_all_models_torch[$TEST_ID]" \
  --no-header -rN \
  --timeout=300 \
  2>&1 | tee "$LOG"
PYTEST_EXIT=${PIPESTATUS[0]}
```

---

### Phase 2 — Reproduce error on bad commit

Run the full run sequence for `$FULL_BAD`. Log goes to `$LOG_DIR/$SHORT_BAD.log`.

After the run, extract and display the error:
```bash
grep -A 20 "FAILED\|AssertionError\|Error\|PCC\|assert\|Timeout\|Aborted\|signal" \
  "$LOG_DIR/$SHORT_BAD.log" | head -40
```

**Compare local output with `known_error`:**

- If the error **matches** (same type and root cause) → confirmed. Print `REPRODUCED: YES`. Proceed to Phase 3.
- If the test **passed** → reproduction failed. Print `REPRODUCED: NO (test passed on bad commit)`. Stop and report.
- If the error is **different** → chip may be in a bad state. Reset chip again and retry once. If still different, print both outputs, report to user, and stop — do not bisect on a false error.

Print:
```
Phase 2 — Bad commit reproduction
SHA:           <FULL_BAD>
Known error:   <known_error>
Local error:   <snippet from log>
Reproduced:    YES / NO
```

---

### Phase 3 — Verify good commit

Run the full run sequence for `$FULL_GOOD`. Log goes to `$LOG_DIR/$SHORT_GOOD.log`.

After the run, classify the result against `expected_good_outcome`:
- If `expected_good_outcome` is `"test passes"` (default):
  - Test passed → `GOOD_VERIFIED: YES`
  - Test failed → `GOOD_VERIFIED: NO` — print the error; stop and report unless user says to continue
- If `expected_good_outcome` is a specific error:
  - Error matches expected → `GOOD_VERIFIED: YES`
  - Error doesn't match → `GOOD_VERIFIED: NO` — print both expected and actual; stop and ask user how to proceed

Print:
```
Phase 3 — Good commit verification
SHA:              <FULL_GOOD>
Expected outcome: <expected_good_outcome>
Actual result:    <PASSED / FAILED with snippet>
Good verified:    YES / NO
```

If `GOOD_VERIFIED: NO` and user wants to continue anyway, note this in the report and proceed to Phase 4.

---

### Phase 4 — Run git bisect

Write `$LOG_DIR/bisect_run.sh`:

```bash
#!/usr/bin/env bash
# Auto-generated by /bisect-commit skill
set -euo pipefail

REPO="__REPO_ROOT__"
GITHUB_REPO="__GITHUB_REPO__"
TEST_ID="__TEST_ID__"
LOG_DIR="__LOG_DIR__"
SUMMARY="$LOG_DIR/bisect_summary.log"

cd "$REPO"
COMMIT=$(git rev-parse HEAD)
SHORT=${COMMIT:0:8}
LOG="$LOG_DIR/$SHORT.log"

echo "" | tee -a "$SUMMARY"
echo "=====================================================" | tee -a "$SUMMARY"
echo "=== BISECT STEP: $COMMIT ===" | tee -a "$SUMMARY"
echo "=====================================================" | tee -a "$SUMMARY"

# ---- Step 1: Restart chip ----
echo "=== Restarting chip ===" | tee -a "$SUMMARY"
tt-smi -r 2>/dev/null || {
  echo "tt-smi not found or failed — reinstalling" | tee -a "$SUMMARY"
  deactivate 2>/dev/null || true
  pip uninstall tt-smi -y 2>/dev/null || true
  source "$REPO/venv/bin/activate"
  pip install tt-smi
  tt-smi -r
}
sleep 3

# ---- Step 2: Activate venv ----
echo "=== Activating venv ===" | tee -a "$SUMMARY"
source "$REPO/venv/bin/activate"

# ---- Step 3: Already at this commit via git bisect ----
echo "=== At commit: $COMMIT ===" | tee -a "$SUMMARY"

# ---- Step 4: Update submodules ----
echo "=== Updating submodules ===" | tee -a "$SUMMARY"
git submodule update --init 2>&1 | tail -5

# ---- Step 5: Download CI wheel ----
echo "=== Finding CI wheel for $SHORT ===" | tee -a "$SUMMARY"
RUN_ID=$(gh api "repos/$GITHUB_REPO/actions/runs?head_sha=$COMMIT&event=push&per_page=1000" \
  --jq '.workflow_runs[] | select(.name == "On push") | .id' 2>/dev/null | head -1)

if [ -z "$RUN_ID" ]; then
  echo "No 'On push' CI run found for $COMMIT — skipping (exit 125)" | tee -a "$SUMMARY"
  echo "$SHORT  SKIPPED  (no CI run)" >> "$SUMMARY"
  exit 125
fi

echo "CI run ID: $RUN_ID" | tee -a "$SUMMARY"
WHEEL_DIR="/tmp/bisect_wheels/$SHORT"
rm -rf "$WHEEL_DIR"
mkdir -p "$WHEEL_DIR"

gh run download "$RUN_ID" \
  --repo "$GITHUB_REPO" \
  --dir "$WHEEL_DIR" \
  --name "xla-whl-release-$SHORT" 2>&1 | tee -a "$SUMMARY"

WHEEL=$(find "$WHEEL_DIR" -name "*.whl" | head -1)
if [ -z "$WHEEL" ]; then
  echo "Wheel not found — artifact may have expired. Skipping (exit 125)" | tee -a "$SUMMARY"
  echo "$SHORT  SKIPPED  (no wheel)" >> "$SUMMARY"
  exit 125
fi

# ---- Step 6: Install wheel ----
echo "=== Installing wheel: $WHEEL ===" | tee -a "$SUMMARY"
uv pip install "$WHEEL" --quiet

# ---- Step 7: Run test ----
echo "=== Running test: $TEST_ID ===" | tee -a "$SUMMARY"
pytest "tests/runner/test_models.py::test_all_models_torch[$TEST_ID]" \
  --no-header -rN \
  --timeout=300 \
  2>&1 | tee "$LOG"

PYTEST_EXIT=${PIPESTATUS[0]}

# Skip if test wasn't collected at this commit
if grep -q "no tests ran\|collected 0 items" "$LOG"; then
  echo "TEST NOT COLLECTED — skipping (exit 125)" | tee -a "$SUMMARY"
  echo "$SHORT  SKIPPED  (not collected)" >> "$SUMMARY"
  exit 125
fi

if [ $PYTEST_EXIT -eq 0 ]; then
  echo "TEST PASSED" | tee -a "$SUMMARY"
  echo "$SHORT  PASSED" >> "$SUMMARY"
  exit 0
else
  echo "TEST FAILED" | tee -a "$SUMMARY"
  echo "$SHORT  FAILED" >> "$SUMMARY"
  exit 1
fi
```

Substitute `__REPO_ROOT__`, `__GITHUB_REPO__`, `__TEST_ID__`, `__LOG_DIR__` with actual values, then:
```bash
chmod +x "$LOG_DIR/bisect_run.sh"
```

Run bisect:
```bash
cd "$REPO_ROOT"
git bisect reset   # clean up any previous bisect state
git bisect start
git bisect bad "$FULL_BAD"
git bisect good "$FULL_GOOD"
git bisect run "$LOG_DIR/bisect_run.sh"
```

Print each bisect step result as it runs: commit SHA, result (PASSED / FAILED / SKIPPED), and any error snippet.

When a step is **skipped** (`exit 125`), note the reason (no CI run, expired artifact, test not collected).

---

### Phase 5 — Report results

When `git bisect run` completes, git identifies the blame commit — the earliest commit where the test fails. Capture it:
```bash
BLAME_SHA=$(git rev-parse refs/bisect/bad)
BLAME_SHORT=${BLAME_SHA:0:8}
```

`BLAME_SHA` is the commit that **introduced** the regression. It is NOT the same as the input `first_bad_sha` (which was just the known-bad starting point of the range).

**5a. Show the blame commit:**
```bash
git show --stat "$BLAME_SHA"
```

**5b. Find the associated PR:**
```bash
gh api "repos/$GITHUB_REPO/commits/$BLAME_SHA/pulls" --jq '.[].html_url'
```

**5c. Show the failure log for the blame commit:**
```bash
grep -A 15 "FAILED\|AssertionError\|Error\|assert\|Timeout\|Aborted" \
  "$LOG_DIR/${BLAME_SHORT}.log" | head -30
```

**5d. Show the full bisect summary:**
```bash
cat "$LOG_DIR/bisect_summary.log"
```

**5e. Clean up bisect state:**
```bash
git bisect reset
```

**5f. Write findings to `$LOG_DIR/bisect_result.md`:**

```markdown
# Bisect Result: <test_id>

**Range:** <FULL_GOOD> (last_good_sha) → <FULL_BAD> (first_bad_sha)
**Blame commit:** <BLAME_SHA>
**PR:** <pr_url>
**Commit date:** <commit date>
**Log directory:** `<LOG_DIR>/`

## Known error (reference)

<known_error>

## Phase 2 — Bad commit reproduction

SHA: <FULL_BAD>
Reproduced: YES/NO
Error snippet:
<snippet>

## Phase 3 — Good commit verification

SHA: <FULL_GOOD>
Expected: <expected_good_outcome>
Result: PASSED/FAILED
Snippet:
<snippet>

## Bisect summary

<contents of $LOG_DIR/bisect_summary.log>
← annotate the first bad commit line

## Error on blame commit

<paste of key lines from $LOG_DIR/<BLAME_SHORT>.log>
```

Also print the result path:
```
Bisect complete.
Blame commit: <BLAME_SHA>
PR:           <pr_url>
Result written to: <LOG_DIR>/bisect_result.md
Logs in:          <LOG_DIR>/
```

---

### Error handling and recovery

| Situation | Action |
|-----------|--------|
| `tt-smi` command not found | Deactivate env, `pip uninstall tt-smi -y`, `source venv/activate`, `pip install tt-smi`, retry `tt-smi -r` |
| `gh api` returns no "On push" run for a commit | `exit 125` — skip commit (no CI wheel available) |
| Artifact download fails / artifact expired | `exit 125` — skip commit |
| Wheel installs but test not collected | `exit 125` — commit predates the test |
| Test hangs beyond `--timeout=300` | Pytest exits non-zero → FAILED |
| Chip error / TT device not responding | Run `tt-smi -r` again. If still failing, apply the full tt-smi reinstall procedure |
| All commits skipped (no artifacts in range) | Report to user — artifact retention window may have passed |
| Bad commit doesn't reproduce the error | Stop, report both errors to user — do not bisect on a false reference |
| Good commit doesn't match expected outcome | Stop, report, ask user how to proceed before bisecting |
| `git bisect` lands on a merge commit | Normal — bisect will find the merge commit if that's where the regression landed |
| Detached HEAD warnings | Expected during bisect — ignore |
| Any other unexpected error mid-bisect | Investigate root cause, fix it, then resume bisect with `git bisect run "$LOG_DIR/bisect_run.sh"` |
