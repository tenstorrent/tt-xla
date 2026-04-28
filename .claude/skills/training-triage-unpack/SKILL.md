---
name: training-triage-unpack
description: Fix FAILED_FE_COMPILATION failures caused by unpack_forward_output not being implemented for a model. Adds _register_attr entries to training_utils.py or overrides unpack_forward_output in loader.py.
---

# Pattern 1 — `unpack_forward_output` not implemented

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

Report a summary: which models were updated to `EXPECTED_PASSING` and which
remain unchanged (fix could not be determined statically).

---

## Reference files

- `third_party/tt_forge_models/training_utils.py` — handler registry
- `third_party/tt_forge_models/base.py` — `ForgeModel.unpack_forward_output`
- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- Existing loader overrides:
  ```bash
  grep -rn "def unpack_forward_output" third_party/tt_forge_models/ --include="*.py"
  ```
