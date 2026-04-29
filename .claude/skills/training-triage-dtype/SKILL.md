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

### Quick-lookup: YAML reason → first action

| YAML `reason` substring | First action | Notes |
|---|---|---|
| `mat1 and mat2 must have the same dtype` | Mixed-precision loader bug. Run CPU `--run` in float32 (Step 2); if it succeeds, fix the inconsistent dtype in `load_inputs`/`load_model` (Step 3). | **Check the inference yaml first** — if inference is `EXPECTED_PASSING` in bfloat16, prefer minimum-impact fix. |
| `'deformable_im2col' not implemented for 'BFloat16'` | Op genuinely lacks bf16. Mark with reason; needs custom op or float32 training path (Step 3). | Not loader-fixable in the general case. |

### Cross-skill referrals

If the dtype fix unmasks a different downstream failure, route deterministically:

| New error after dtype fix | Skill |
|---|---|
| `targets should not be none ...`, `decoder_input_ids or decoder_inputs_embeds`, `Expected more than 1 value per channel ...`, `Model expects targets ...` | `training-triage-inputs` |
| `No handler for class <X> exists in unpack_forward_output`, `tt-forge-models doesn't implement unpack_forward_output ...` | `training-triage-unpack` |
| `TT_FATAL: ...`, `Out of Memory`, `DRAM Auto slice ...` | leave as `FAILED_RUNTIME`; not loader-fixable |

---

## Setup

All commands run from the repo root with the venv active:

```bash
source venv/activate
```

### Per-model dependencies (`ModuleNotFoundError` cases)

If the CPU `--run` reports `ModuleNotFoundError: No module named 'X'`,
check for a per-model requirements file before assuming the env is broken:

```bash
ls third_party/tt_forge_models/<model>/pytorch/requirements*.txt
```

Three filename conventions are in use:

- `requirements.txt` — install with `pip install -r`.
- `requirements.nodeps.txt` — install with `pip install --no-deps -r`.
- `requirements.nodeps.nobuildisolation.txt` — install with
  `pip install --no-deps --no-build-isolation -r`.

If no requirements file exists, treat the import error as an
environment/upstream issue and update the YAML reason accordingly.

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

> **Pre-flight: don't break the inference test.** The same loader serves
> the inference yaml. If the inference test passes in bfloat16 today,
> hard-coding `dtype_override=torch.float32` or stripping the bf16 branch
> from `load_inputs` will silently break it. Check first:
>
> ```bash
> grep "<model>/<variant>-single_device-inference" \
>     tests/runner/test_config/torch/test_config_inference_*.yaml
> ```
>
> If inference is `EXPECTED_PASSING`, prefer a *minimum-impact* fix:
> cast only the offending tensor to match its peer (e.g. bias to match
> weight) rather than changing the whole loader's dtype path. If
> inference is already `KNOWN_FAILURE_XFAIL` / `NOT_SUPPORTED_SKIP`,
> a broader fix is fine.

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
timeout 300 pytest -svv "tests/runner/test_models.py::test_all_models_torch[<key>]" >/tmp/pytest.log 2>&1
# then read /tmp/pytest.log with the Read tool (or grep) — do NOT use tail
```

**Always wrap pytest in `timeout 300` (5 min).** The FE-compilation check
typically completes in 15–60 s; anything past 5 min on a single test is
either a hang or a slow OOM/sliced-allocation retry that won't recover.
If `timeout` returns 124, mark `bringup_status: FAILED_RUNTIME`,
`reason: "Test timed out"` and move on.

**Never pipe pytest output through `tail`.** The tail of a tt-xla pytest
run is almost always a generic `Error Code 13` / pytest exit summary
that hides the real failure, which lives several hundred lines earlier
in the dump (the actual Python traceback, the assert text, or the
TT-MLIR error block). Dump the full output to a temp file and read it
— search for `FAILED`, `Error`, `Traceback`, or the model name to find
the originating line. The same applies to `head` and short `grep -A`
windows: only the full file gives you the right error to put in the
YAML `reason`.

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
one item; then run each in turn. Always wrap with `timeout 300` (see
above) — if it triggers, mark `bringup_status: FAILED_RUNTIME`,
`reason: "Test timed out"`.

**Skip larger siblings of OOM-ing variants.** If `Medium` of a family
hits a TT DRAM OOM, don't bother running `Large` / `Large_v3` /
`Large_v3_Turbo`: they only get bigger and will OOM the same way.
Mark them directly as `FAILED_RUNTIME` with a reason that explicitly
notes the extrapolation, e.g. `"TT_FATAL: Out of Memory ... DRAM
buffer (extrapolated from Medium variant which OOMs at 72 MB)"`.

### Skip-without-running matrix

Apply these without re-running pytest. Each row converts a deterministic
condition into a YAML mark, no further investigation needed:

| Condition | Mark | Reason text (template) |
|---|---|---|
| Sibling Large/X-Large of a variant that already OOMed on TT DRAM | `FAILED_RUNTIME` | `"TT_FATAL: Out of Memory ... DRAM buffer (extrapolated from <smaller> variant which OOMs at <X> MB)"` |
| `timeout 300` triggered (`exit code 124`) | `FAILED_RUNTIME` | `"Test timed out"` |
| YAML reason matches a known-stale dtype string AND CPU `--run` shows a different error | `FAILED_FE_COMPILATION` (or whatever the new stage is) | the verbatim CPU `--run` error |
| Loader fails to import an `<X>` module that has **no** `requirements*.txt` under `third_party/tt_forge_models/<model>/pytorch/` | `FAILED_FE_COMPILATION` | `"ModuleNotFoundError: No module named '<X>'"` |
| Loader fails to import despite a present `requirements*.txt` (install fails or post-install loader still raises) | `FAILED_FE_COMPILATION` | the verbatim install/import error |

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
