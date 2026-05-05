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
If result is `"timeout"`:
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
      "failure_reason": "Test exceeded 300s wall-clock limit — likely downloading large model weights or OOM. Marking UNKNOWN (bringup not attempted)."
    }
  }
  ```
- Set `state.stage = "escalated"` in state.json.
- Call `model-bringup-config-update` skill with `result=TIMEOUT`.
- **Stop the FSM — do not proceed to DIAGNOSE.**

Output:
```
[run] iter=<N>  result=TIMEOUT  duration=300s
  Marking as UNKNOWN (bringup not attempted) — exceeded 5-minute execution limit.
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
