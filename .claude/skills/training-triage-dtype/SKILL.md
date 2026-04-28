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

## Reference files

- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- `tools/triage_fe_failures.py` — triage script
