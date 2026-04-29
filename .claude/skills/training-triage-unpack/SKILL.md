---
name: training-triage-unpack
description: Fix FAILED_FE_COMPILATION failures caused by unpack_forward_output not being implemented for a model. Adds _register_attr entries to training_utils.py or overrides unpack_forward_output in loader.py.
---

# Pattern 3 — `unpack_forward_output` not implemented

**Error:** `tt-forge-models doesn't implement unpack_forward_output for this model.`

This fires in `training_utils.py:48` when the model's forward output is not a
plain `torch.Tensor` and has no registered handler. The fix is either:
- Register the output class in `training_utils.py` via `_register_attr`, or
- Override `unpack_forward_output` in the specific model loader.

**No CPU execution is needed for this pattern.** The YAML error message proves
the model already loaded and ran on the TT device — it only failed at the
output-unpacking step. The output class is determined by the model's `forward()`
return type, which is fixed and discoverable from the loader code or HF docs
without running anything.

### How you got here (cascade from `training-triage-inputs`)

This skill is also commonly triggered by `training-triage-inputs`'s
post-fix cascade — the failure shape is:

```
ValueError: No handler for class <X> exists in `unpack_forward_output`.
```

The class name `<X>` is in the parens of the YAML reason. If `training-triage-inputs`
just ran, its Step 7 aggregate already lists each entry's output class —
read that report rather than re-deriving from the loader. Empirically every
torchvision detection model (`ssdlite320`, `ssd300_vgg16`, `ssd300_resnet50`,
RetinaNet variants) and `gliner` ended up here after their inputs fix.

---

## Setup

All commands run from the repo root with the venv active:

```bash
source venv/activate
```

### Per-model dependencies (`ModuleNotFoundError` cases)

If the loader fails to import with `ModuleNotFoundError: No module named 'X'`,
check for a per-model requirements file before assuming the env is broken:

```bash
ls third_party/tt_forge_models/<model>/pytorch/requirements*.txt
```

Three filename conventions are in use:

- `requirements.txt` — install with `pip install -r`.
- `requirements.nodeps.txt` — install with `pip install --no-deps -r`
  (used when the package's transitive deps would conflict with the env).
- `requirements.nodeps.nobuildisolation.txt` — install with
  `pip install --no-deps --no-build-isolation -r` (needed for packages
  that build against the venv's existing torch, e.g. `yolox==0.3.0`).

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
python tools/triage_fe_failures.py --pattern unpack
```

---

## Step 2 — Identify the output class for each model

For each entry, read the loader file to identify the model and its output class.
For HuggingFace models the output class name is typically the return type
annotation of `forward()`, or matches the model class name pattern.

Check whether the class is already in `_HANDLER_REGISTRY`:
```bash
grep "ClassName" third_party/tt_forge_models/training_utils.py
```
If it is already registered, the YAML entry is stale — skip to Step 4.

---

## Step 3 — Apply the fix

**A — HuggingFace named output class** (class name ends in `Output`):

The class name is determined by the task/architecture. Apply the table:

| Output class name pattern | Attribute |
|---------------------------|-----------|
| `*ImageClassifier*`, `*Classification*`, `*ObjectDetection*`, `*Semantic*`, `*Segmenter*` | `logits` |
| `*CausalLM*`, `*MaskedLM*`, `*LMOutput*`, `*Seq2Seq*`, `*TokenClassifier*` | `logits` |
| `*QuestionAnswering*` | `start_logits` |
| `*ContextEncoder*`, `*QuestionEncoder*`, `*Encoder*` (DPR-style) | `pooler_output` |
| `*BaseModel*`, embedding models | `last_hidden_state` |
| `Sam*Segmentation*` | `pred_masks` |
| `*MaskedImageModeling*`, `*Reconstruction*` | `reconstruction` |
| `*Depth*` | `predicted_depth` |
| `UNet2DCondition*`, diffusion | `sample` |

Fix: add one line to `third_party/tt_forge_models/training_utils.py` (keep
the block alphabetically sorted):
```python
_register_attr("OutputClassName", "logits")
```

This works even when the model fails to load locally (e.g. `IRD_LF_CACHE`
not set, `AttributeError` during init, missing module). The class name comes
from the HF docs or the loader's import/type hints, not from running the model.

---

**B — dict output**:

Read the loader to identify the primary output key (look at the model's
`forward()` return statement or HF docs). Implement `unpack_forward_output`
in the model's `loader.py`:
```python
def unpack_forward_output(self, fwd_output):
    return fwd_output["dense_vecs"]  # replace with the actual key
```
Only use `--run` if the dict keys are genuinely unknown after reading the code:
```bash
python tools/triage_fe_failures.py --pattern unpack --run --limit 5 --offset 0 --output /tmp/unpack.json
```

---

**C — tuple, list, or non-HF custom class**:

> **Hard rule: never register `tuple`, `list`, or `dict` in
> `_HANDLER_REGISTRY`.** They are the literal builtins and would match
> every tuple/list/dict-returning forward across the codebase, almost
> always producing the wrong attribute. These output types **always**
> require a per-loader override of `unpack_forward_output` in the
> model's `loader.py` — never a global handler.

Read the loader's `forward()` or any patched forward function defined in the
loader file to understand the output structure. If the tuple has a known
structure (e.g. `(head_outputs, anchors)`), destructure it and recurse only
into the loss-relevant part:
```python
def unpack_forward_output(self, fwd_output):
    import torch
    from ...tools.utils import extract_tensors_recursive
    head_outputs, anchors = fwd_output  # adjust to actual structure
    tensors = []
    extract_tensors_recursive(head_outputs, tensors)
    if tensors:
        return torch.cat([t.flatten() for t in tensors])
    return head_outputs["cls_logits"]
```
If the structure is unknown (opaque list/tuple), use the generic form:
```python
def unpack_forward_output(self, fwd_output):
    import torch
    from ...tools.utils import extract_tensors_recursive
    tensors = []
    extract_tensors_recursive(fwd_output, tensors)
    if tensors:
        return torch.cat([t.flatten() for t in tensors])
    return fwd_output
```
Detection models that fail to load locally (e.g. `IRD_LF_CACHE`) can still
be fixed this way — the generic pattern is safe for any detection model
returning a list or tuple of tensors.

**Empirical: torchvision detection cluster.** RetinaNet, SSD300-VGG16,
SSD300-ResNet50, SSDLite320-MobileNetV3, RetinaNet_FPN_V2 all return
`(head_outputs, anchors)` from their patched forward (the patch lives in
the loader's `src/utils.py` or in the loader file itself). The reference
implementation is at `third_party/tt_forge_models/retinanet/pytorch/loader.py:331`:

```python
def unpack_forward_output(self, fwd_output):
    import torch
    from ...tools.utils import extract_tensors_recursive

    head_outputs, _anchors = fwd_output
    tensors = []
    extract_tensors_recursive(head_outputs, tensors)
    if tensors:
        return torch.cat(tensors, dim=0)
    return head_outputs
```

Copy this verbatim into the other detection loaders' `loader.py` — none of
them need anchors for the loss path, and `extract_tensors_recursive`
handles both dict-of-tensors and list-of-tensors `head_outputs`.

**Style note:** always use local imports inside `unpack_forward_output`
(not top-level), consistent with the rest of the codebase.

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

---

## Step 5 — Verify each fix by running pytest

Fixing the unpack step often unmasks a different downstream failure
(missing inputs, grad-disabled EMA checkpoints, dtype mismatches, FE/MLIR
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
- **FAILED** — the unpack fix is verified (compilation got past it), but
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
| YAML reason matches a known-stale string AND CPU `--run` (or pytest re-run) shows a different error | `FAILED_FE_COMPILATION` (or whatever the new stage is) | the verbatim observed error |
| Loader fails to import an `<X>` module that has **no** `requirements*.txt` under `third_party/tt_forge_models/<model>/pytorch/` | `FAILED_FE_COMPILATION` | `"ModuleNotFoundError: No module named '<X>'"` |
| Loader fails to import despite a present `requirements*.txt` (install fails or post-install loader still raises) | `FAILED_FE_COMPILATION` | the verbatim install/import error |

Report a summary at the end:
- Which models pass end-to-end (kept `EXPECTED_PASSING`).
- Which models had the unpack error replaced by a new error (reverted to
  `NOT_SUPPORTED_SKIP` with the captured reason).
- Which models remain unchanged because the fix could not be determined
  statically.

---

## Reference files

- `third_party/tt_forge_models/training_utils.py` — handler registry
- `third_party/tt_forge_models/base.py` — `ForgeModel.unpack_forward_output`
- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- Existing loader overrides:
  ```bash
  grep -rn "def unpack_forward_output" third_party/tt_forge_models/ --include="*.py"
  ```
