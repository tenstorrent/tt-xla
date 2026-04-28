---
name: training-triage-inputs
description: Fix FAILED_FE_COMPILATION failures caused by missing training inputs — models that need targets, decoder_input_ids, labels, or other training-only tensors that load_inputs doesn't provide.
---

# Pattern 3 — missing training inputs

**Errors:**
- `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds.`
- `Model expects targets to be passed while in training mode`
- `AssertionError: targets should not be none when in training mode`
- `AttributeError: 'NoneType' object has no attribute 'max'`
- `ValueError: Expected more than 1 value per channel when training…`

These mean `load_inputs` doesn't pass what the model needs during training.
The model is being called the same way for inference and training, but training
requires extra tensors (ground-truth labels, decoder start tokens, etc.).

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
python tools/triage_fe_failures.py --pattern inputs
```

---

## Step 2 — Identify what each model needs

For each model, read its loader and the model's forward signature:
```bash
grep -n "def load_inputs\|def forward" third_party/tt_forge_models/<model>/pytorch/loader.py
```
Then check the actual model class's `forward()` in the venv or HuggingFace
source to see what arguments are required in training mode.

---

## Step 3 — Fix by sub-pattern

**`decoder_input_ids` / `decoder_inputs_embeds`** (seq2seq models like Whisper, T5-like):
Add `decoder_input_ids` or `decoder_inputs_embeds` to the dict returned
by `load_inputs`. A common approach is to use the `decoder_start_token_id`:
```python
decoder_input_ids = torch.full((batch, 1), model.config.decoder_start_token_id)
inputs["decoder_input_ids"] = decoder_input_ids
```

**`targets` / `targets should not be none`** (detection/segmentation models):
Pass dummy ground-truth targets matching the expected format.
Look at training scripts or HuggingFace examples for the model to understand
the target format (e.g., YOLOX expects `[batch_size, n_gt, 5]` tensors).
Add to `load_inputs`:
```python
targets = torch.zeros(batch_size, 1, 5)  # adjust shape to model's expectation
return inputs, targets
```
Then update `unpack_forward_output` if the loss is returned as part of a tuple.

**`'NoneType' object has no attribute 'max'`**:
The model calls `.max()` on something that is `None` — likely a target or
label tensor that should have been passed. Inspect the model's forward source
to find where `.max()` is called and what variable is `None`. This is usually
a missing `labels` or `targets` kwarg.

**`Expected more than 1 value per channel`** (batch norm in training mode):
The input spatial dimensions are too small for batch norm in training mode.
Increase the input size in `load_inputs` so that spatial dims > 1.

---

## Step 4 — Verify the fix runs on CPU

```bash
source venv/activate && python -c "
import sys; sys.path.insert(0, '.')
from third_party.tt_forge_models.<model>.pytorch.loader import ModelLoader
import torch
loader = ModelLoader()
model = loader.load_model().train()
inputs = loader.load_inputs()
with torch.enable_grad():
    out = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
    loss = loader.unpack_forward_output(out)
    loss.backward()
print('OK:', type(out).__name__)
"
```

---

## Step 5 — Aggregate and update the YAML

After going through all missing-input models, look for common patterns (e.g.,
all detection models need the same target format). If a bulk fix applies,
apply it in a shared base class or update multiple loaders at once.

For each fixed model, remove `status: NOT_SUPPORTED_SKIP`,
`bringup_status: FAILED_FE_COMPILATION`, and `reason: ...` and set
`status: EXPECTED_PASSING`:
```yaml
model/path/pytorch-Variant-single_device-training:
  status: EXPECTED_PASSING
  markers: [nightly]
```

---

## Reference files

- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- `tools/triage_fe_failures.py` — triage script
