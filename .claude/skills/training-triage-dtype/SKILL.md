---
name: training-triage-dtype
description: Fix FAILED_FE_COMPILATION failures caused by bfloat16 dtype mismatches or unsupported ops in bfloat16. Checks whether the model works in float32 and fixes mixed-precision loader issues.
---

# Pattern 2 — dtype errors in bfloat16

**Errors:**
- `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
- `RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'`

These indicate either mixed-precision inputs or an op that PyTorch doesn't
support in bfloat16. The question is: does the model work at all in float32 on CPU?

---

## Setup

All commands run from the repo root with the venv active:

```bash
source venv/activate
```

---

## Step 1 — List failures

To see all `FAILED_FE_COMPILATION` entries across all patterns:
```bash
python tools/triage_fe_failures.py
```

To restrict to this pattern:
```bash
python tools/triage_fe_failures.py --pattern dtype
```

---

## Step 2 — Run failing models on CPU in float32

```bash
python tools/triage_fe_failures.py --pattern dtype --run
```

The script runs each loader with `dtype_override=torch.float32`.

---

## Step 3 — Interpret results and fix

**No error in float32** → the model works in float32; the issue is inconsistent
dtype in the loader. Check `load_inputs` and `load_model` for mixed
`float32`/`bfloat16` assignments. Fix: ensure all tensors use the same dtype
throughout `load_inputs`.

**Same or different error** → deeper issue unrelated to dtype. Update the YAML
`reason` field to document the new error; do not mark as `EXPECTED_PASSING`.

**`deformable_im2col` error** → this PyTorch op genuinely lacks bfloat16
support. If the model is otherwise correct, add a note in the YAML that it
needs a custom op implementation or a float32 training path.

---

## Step 4 — Update the YAML

For each fixed model, remove `status: NOT_SUPPORTED_SKIP`,
`bringup_status: FAILED_FE_COMPILATION`, and `reason: ...` and set
`status: EXPECTED_PASSING`:
```yaml
model/path/pytorch-Variant-single_device-training:
  status: EXPECTED_PASSING
  markers: [nightly]
```

For models that remain broken, update `reason` to reflect the new error
observed during the CPU run.

---

## Step 5 — Verify each fix by running pytest

Fixing the dtype mismatch often unmasks a different downstream failure
(missing inputs, grad-disabled EMA checkpoints, unpack errors, FE/MLIR
compilation errors, runtime OOM, etc.). After flipping a model to
`EXPECTED_PASSING`, run the corresponding pytest **one model at a time**
to avoid timeouts and to keep failure attribution clean. Each test
typically takes 15–60 s for the FE-compilation check.

For each fixed entry `<key>` (e.g. `yolov6/pytorch-N-single_device-training`):

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<key>]" 2>&1 | tail -40
```

Interpret the result:
- **PASSED** — leave the YAML at `status: EXPECTED_PASSING`.
- **FAILED** — the dtype fix is verified (compilation got past it), but
  a *different* downstream issue blocks the test. Revert the YAML entry
  to `status: NOT_SUPPORTED_SKIP` with a `bringup_status` and a `reason`
  that captures the new error verbatim (one line, quoted).

  `bringup_status` reflects the **pipeline stage** of the failure, not
  the Python exception class. A `RuntimeError` from torch during forward
  or autograd is still `FAILED_FE_COMPILATION` — `FAILED_RUNTIME` is
  reserved for TT-device runtime, post-compilation. Pick by stage:
  - `FAILED_FE_COMPILATION` — anything before the TT device executes:
    model load, CPU baseline forward/backward, autograd graph build,
    dtype mismatches during forward, FX/stablehlo lowering. Most torch
    `RuntimeError`s land here. (Example reasons in the YAML:
    `"RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'"`,
    `"RuntimeError: mat1 and mat2 must have the same dtype..."`.)
  - `FAILED_RUNTIME` — TT device runtime only: kernel hangs, L1/DRAM
    allocation overflow on TT cores, TT-Metal asserts. (Example:
    `"RuntimeError: Test Hangs"`, `"Statically allocated circular
    buffers ... beyond max L1 size"`.)
  - `FAILED_TTMLIR_COMPILATION` — failures inside the TT-MLIR compiler
    proper (post-stablehlo, before runtime).

  Quick disambiguator: if the traceback shows the failure inside
  `_run_on_cpu(...)` or before any TT/XLA call, it's
  `FAILED_FE_COMPILATION`, regardless of the exception type.

Example revert when the test fails with a torch autograd error:
```yaml
yolov6/pytorch-N-single_device-training:
  status: NOT_SUPPORTED_SKIP
  bringup_status: FAILED_FE_COMPILATION
  reason: "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
```

**Run-budget guidance:** when fixing many entries, run pytest
sequentially (not in parallel) — each test acquires the TT device. Use
`pytest --collect-only` first to confirm the test ID resolves to exactly
one item; then run each in turn. If a single test exceeds ~10 minutes,
kill it and mark `bringup_status: FAILED_RUNTIME`,
`reason: "Test timed out"`.

Report a summary at the end:
- Which models pass end-to-end (kept `EXPECTED_PASSING`).
- Which models had the dtype error replaced by a new error (reverted to
  `NOT_SUPPORTED_SKIP` with the captured reason).
- Which models remain unchanged because the fix could not be determined
  from the float32 CPU run.

---

## Reference files

- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- `tools/triage_fe_failures.py` — triage script
