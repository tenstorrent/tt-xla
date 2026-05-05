---
name: model-bringup-config-update
description: Config update stage of the model bringup pipeline. Updates the test YAML config and bringup_status marker in the test fixture to reflect the final outcome (PASSED, TIMEOUT, or ESCALATED). Invoked by the model-bringup orchestrator as the final stage.
allowed-tools: Read Write Edit Bash Grep
---

# Model Bringup — Config Update

You are the **config update** stage of the model bringup pipeline.

## Invocation
`/model-bringup-config-update <model_key> --result <PASSED|TIMEOUT|ESCALATED> [--arch <arch>]`

## Responsibility
Update the test YAML configuration and the test file's `bringup_status` marker
to reflect the final bringup outcome.

## Steps

### 1. Locate the YAML config file
Check for:
1. `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
2. Other YAML files under `tests/runner/test_config/` matching the model's run_mode and parallelism.

### 2. Locate the bringup_status in the test fixture
Find `bringup_status=BringupStatus.<value>` inside the pytest fixture for the model variant
in `third_party/tt_forge_models/<family>/pytorch/tests/test_*.py`.

### 3. Apply the update

**If result == PASSED:**
- Set `bringup_status=BringupStatus.EXPECTED_PASSING` in the test fixture.
- Ensure the test variant is **not** marked with `pytest.mark.skip`.
- In the YAML config, add or update the model entry:
  ```yaml
  <model_key>:
    status: EXPECTED_PASSING
    supported_archs: ["<arch>"]
    assert_pcc: false  # generative model; set true if PCC is stable
  ```
- Update `state.json`: set `stage: "passed"`.
- Append history entry: `{ "stage": "config_update", "result": "passed" }`.

**If result == TIMEOUT:**
- Set `bringup_status=BringupStatus.UNKNOWN` in the test fixture.
- Add `pytest.mark.skip(reason="<timeout_reason>")` to the test variant.
- In the YAML config, add entry with `status: NOT_SUPPORTED_SKIP`.
- Update `state.json`: set `stage: "not_supported_skip"`.
- Append history entry: `{ "stage": "config_update", "result": "not_supported_skip" }`.
- Write `.claude/bringup/<safe_key>/escalation_report.md` with timeout details.

**If result == ESCALATED:**
- Set `bringup_status=BringupStatus.KNOWN_FAILURE_XFAIL` in the test fixture (or
  `BringupStatus.UNSPECIFIED` if cause is truly unknown).
- Add `@pytest.mark.xfail(reason="<last failure_reason from state.json>")` to the test.
- In the YAML config, add entry with `status: KNOWN_FAILURE_XFAIL` and `reason:`.
- Update `state.json`: set `stage: "escalated"`.
- Append history entry: `{ "stage": "config_update", "result": "escalated" }`.
- Write `.claude/bringup/<safe_key>/escalation_report.md` with:
  - model_key, arch
  - All history entries (stage, result, details)
  - Final diagnosis and repair attempts
  - Recommended next human action

### 4. Write to bringup_steps.txt

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Config Update (model-bringup-config-update)
--------------------------------------------------------------------------------
Result        : PASSED | TIMEOUT | ESCALATED

bringup_status updated:
  File : <test file path>
  From : BringupStatus.<old>
  To   : BringupStatus.<new>

YAML config updated:
  File  : tests/runner/test_config/torch/test_config_inference_single_device.yaml
  Entry :
    <yaml block>

[If ESCALATED or TIMEOUT:]
  Escalation report: .claude/bringup/<safe_key>/escalation_report.md

CONFIG UPDATE RESULT: PASSED | TIMEOUT | ESCALATED
```

Then close the file with the final summary block (see orchestrator SKILL.md for format).

### 5. Output

On PASSED:
```
[config-update] PASSED
  bringup_status → EXPECTED_PASSING
  test file:  <path>
  yaml entry: <path>
```

On TIMEOUT:
```
[config-update] UNKNOWN
  bringup_status → UNKNOWN
  reason:     exceeded 300s execution limit
  report:     .claude/bringup/<safe_key>/escalation_report.md
```

On ESCALATED:
```
[config-update] ESCALATED
  bringup_status → KNOWN_FAILURE_XFAIL
  report:     .claude/bringup/<safe_key>/escalation_report.md
```
