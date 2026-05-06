---
name: model-bringup-run
description: Test execution stage of the model bringup pipeline. Runs pytest for a given model_key with a 5-minute hard timeout, captures output to a log file, and records result (passed/failed/timeout) in state.json. Invoked by the model-bringup orchestrator at the FIRST_RUN and VERIFY stages.
allowed-tools: Bash Read Write
---

# Model Bringup — Test Execution

You are the **test execution** stage of the model bringup pipeline.

## Invocation
`/model-bringup-run <model_key> [--arch <arch>] [--iteration <N>]`

## Responsibility
Run the pytest test for the given model_key and capture the full output to a log file.

## Timeout Policy
**Maximum allowed wall-clock time: 5 minutes (300 seconds).**
If the test process exceeds this limit, kill it immediately and treat the result as
`TIMEOUT`. Do NOT wait for it to finish.

## Steps

### 1. Discover test node IDs via collect

Parse the `model_key` to build a grep pattern for `tests/runner/test_models.py`:

- **Full structured key** (`ltx2/pytorch-Fast-single_device-inference`): match exact bracket content
  ```bash
  grep 'test_all_models_torch\[ltx2/pytorch-Fast-single_device-inference\]'
  ```
- **Family-only key** (`ltx2`): match all single-device inference variants
  ```bash
  grep 'test_all_models_torch\[ltx2/pytorch-.*-single_device-inference\]'
  ```

Run collect and filter:
```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep 'test_all_models_torch\[<pattern>\]'
```

Collect the full `tests/runner/test_models.py::test_all_models_torch[...]` lines as `test_node_ids`.

If `test_node_ids` is empty, **fail immediately** — do not proceed to run:
```
[run] FAILED — no test node found for <model_key> in tests/runner/test_models.py
  Check: third_party/tt_forge_models/<family>/pytorch/loader.py exists and imports cleanly.
  Verify: python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader"
```

### 2. Run pytest with timeout
Execute the discovered node IDs with a hard 300-second wall-clock limit:
```bash
timeout 300 python -m pytest <test_node_ids> -v --tb=short 2>&1 | tee .claude/bringup/<safe_key>/logs/iter_<N>_run.log
exit_code=${PIPESTATUS[0]}
```

- If `exit_code == 124` → the process was killed by `timeout`; record `result = "timeout"`.
- If `exit_code == 0` → `result = "passed"`.
- Any other exit code → `result = "failed"`.

Set env var `TT_XLA_ARCH=<arch>` for the subprocess.

Do not suppress any output — the full stdout+stderr must be captured.

### 3. Handle TIMEOUT result

If `exit_code == 124`, do **NOT** immediately mark as UNKNOWN. The 300s budget
is often too short for first-run weight downloads, HF cache misses, or initial
JIT/XLA compilation, and the actual pass/fail signal usually appears within a
few extra minutes. **Pause the pipeline and ask the user to run the test
manually with a longer budget**, then resume with their log.

#### 3a. Print the manual-run prompt and stop the turn

Print exactly this block, then **end the assistant turn without further tool
calls** so the user can reply:

```
[run] iter=<N>  TIMEOUT at 300s — pipeline paused, manual run required.

Please run the test yourself with a longer budget and share the log:

  TT_XLA_ARCH=<arch> timeout 1800 python -m pytest \
    '<test_node_id>' \
    -v --tb=short -s 2>&1 | tee /tmp/<safe_key>_manual.log

Then reply with one of:
  - the path to the log file (e.g. /tmp/<safe_key>_manual.log), or
  - the pasted log content, or
  - "skip" to fall through to the original TIMEOUT/UNKNOWN escalation.
```

Do not proceed past this point until the user has replied.

#### 3b. Process the user's reply

When the user replies, branch on what they provided:

- **Log path** (file exists, e.g. `/tmp/<safe_key>_manual.log`):
  Copy it to `.claude/bringup/<safe_key>/logs/iter_<N>_manual.log` and inspect
  the tail.

- **Pasted log content**:
  Write it to `.claude/bringup/<safe_key>/logs/iter_<N>_manual.log` and inspect
  the tail.

- **"skip"** (or any clear opt-out):
  Fall through to the original TIMEOUT escalation in **3c** below.

Inspect the tail of the manual log and classify:

| Tail signature | Recorded `result` | Next stage |
|---|---|---|
| `=== <N> passed in <T>s ===` | `passed` | continue to Step 4, then orchestrator → CONFIG_UPDATE(PASSED) |
| `=== <N> failed in <T>s ===` or traceback / `1 error` | `failed` | continue to Step 4 with the manual log path; orchestrator → DIAGNOSE |
| Truncated / still timed out at the larger budget / no clear verdict | (treat as still TIMEOUT) | fall through to **3c** |

When recording a passed/failed result from a manual log, use the manual log
path in `details.log` and add `details.source: "manual_run"` plus the budget
the user used (`details.manual_timeout_seconds`).

#### 3c. Fallback: original TIMEOUT/UNKNOWN escalation

Only reached if the user replied "skip" or the manual log was still
inconclusive.

- Append to `history`:
  ```json
  {
    "stage": "first_run",
    "timestamp": <now>,
    "result": "timeout",
    "details": {
      "log": "logs/iter_<N>_run.log",
      "returncode": 124,
      "duration_seconds": 300,
      "failure_reason": "Test exceeded 300s wall-clock limit and manual run did not produce a verdict. Marking UNKNOWN (bringup not attempted)."
    }
  }
  ```
- Set `state.stage = "escalated"` in state.json.
- Call `model-bringup-config-update` skill with `result=TIMEOUT`.
- **Stop the FSM — do not proceed to DIAGNOSE.**

Output:
```
[run] iter=<N>  result=TIMEOUT  duration=300s
  Marking as UNKNOWN (bringup not attempted) — exceeded 5-minute execution limit
  and manual run did not produce a verdict.
  log: .claude/bringup/<safe_key>/logs/iter_<N>_run.log
```

### 4. Record result in state.json (non-timeout)
Load `.claude/bringup/<safe_key>/state.json`, append to `history`:
```json
{
  "stage": "first_run",
  "timestamp": <now>,
  "result": "passed" | "failed",
  "details": {
    "log": "logs/iter_<N>_run.log",
    "returncode": <int>,
    "duration_seconds": <float>
  }
}
```

### 5. Write to bringup_steps.txt

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — First Run / Verify (model-bringup-run, iteration <N>)
--------------------------------------------------------------------------------
Collect : pytest -q --collect-only tests/runner/test_models.py | grep '<family>/pytorch-...'
Node IDs: <list of discovered test node IDs>
Command : TT_XLA_ARCH=<arch> timeout 300 python -m pytest <test_node_ids> -v --tb=short
Log     : .claude/bringup/<safe_key>/logs/iter_<N>_run.log
Duration: <Xs>
Exit    : <exit_code>

Test result summary:
  <last 3 lines of pytest output, e.g. "1 passed" or "1 failed">

RUN RESULT: PASSED | FAILED | TIMEOUT
[If TIMEOUT:]  Marking as UNKNOWN — exceeded 5-minute execution limit.
[If FAILED:]   Passing log to DIAGNOSE stage.
```

### 6. Output (non-timeout)
```
[run] iter=<N>  result=<PASSED|FAILED>  duration=<Xs>
  log: .claude/bringup/<safe_key>/logs/iter_<N>_run.log
```

Return the log path so the orchestrator can pass it to `model-bringup-diagnose`.
