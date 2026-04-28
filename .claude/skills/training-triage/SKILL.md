---
name: training-triage
description: Triages FAILED_FE_COMPILATION training test failures. Covers four error patterns: unpack_forward_output not implemented, bfloat16 dtype issues, missing training inputs (targets/decoder_ids/etc.), and aggregation of missing-inputs models.
---

# Training FE Failure Triage

Triage `FAILED_FE_COMPILATION` entries in
`tests/runner/test_config/torch/test_config_training_single_device.yaml`.

Each failure falls into one of four patterns. The workflow below describes
how to diagnose and fix each one. Run patterns independently or together.

---

## Setup

All commands run from the repo root with the venv active:

```bash
source venv/activate
```

The triage script at `tools/triage_fe_failures.py` does the mechanical work.
Use it to get structured data; then apply the pattern-specific fix workflows below.

---

## Step 0 — List all failures (no execution required)

```bash
python tools/triage_fe_failures.py
```

This prints every `FAILED_FE_COMPILATION` entry grouped by pattern with loader paths.
No model weights are downloaded. Start here to understand scope.

To restrict to one pattern group:
```bash
python tools/triage_fe_failures.py --pattern unpack   # or dtype | inputs
```

---

## Pattern 1 — `unpack_forward_output` not implemented

**Error:** `tt-forge-models doesn't implement unpack_forward_output for this model.`

This fires in `training_utils.py:48` when the model's forward output is not a
plain `torch.Tensor` and has no registered handler. The fix is either:
- Register the output class in `training_utils.py` via `_register_attr`, or
- Override `unpack_forward_output` in the specific model loader.

### Workflow

Do not ask clarifying questions. Run the steps below immediately.

1. First, list all models for this pattern (no execution):
   ```bash
   python tools/triage_fe_failures.py --pattern unpack
   ```
   This gives you the full model list with loader paths.

2. Run models in batches of 5 to collect output class names. The `--run` flag
   executes each model on CPU and will time out if run on all models at once.
   Use `--offset` and `--limit` to split into batches:
   ```bash
   python tools/triage_fe_failures.py --pattern unpack --run --limit 5            # models 1-5
   python tools/triage_fe_failures.py --pattern unpack --run --limit 5 --offset 5 # models 6-10
   python tools/triage_fe_failures.py --pattern unpack --run --limit 5 --offset 10 # models 11-15
   ```
   Repeat until all models are covered. For each model the script prints
   `output_class=XYZ` and whether it's already registered in `_HANDLER_REGISTRY`.

3. For each unregistered output class, determine the correct tensor attribute
   by reading the loader and model output class source. Priority order for
   training loss:
   - `logits` — classification, LM heads (cross-entropy loss)
   - `last_hidden_state` — encoder-only/embedding models
   - `loss` — if model returns pre-computed loss (rare in this codebase)
   - `sample` — diffusion model outputs
   - `predicted_depth` — depth estimation

4. **If the same output class is shared across models** (common for HF model families):
   Add a single line to `third_party/tt_forge_models/training_utils.py`:
   ```python
   _register_attr("OutputClassName", "logits")  # or whichever attr
   ```

5. **If the model returns a plain tuple/list or a custom class** that can't be
   unambiguously mapped to one tensor (e.g., object detection models returning
   `(cls_scores, bbox_preds)`, or models where the "right" tensor depends on
   task): override `unpack_forward_output` in the model's `loader.py`:
   ```python
   def unpack_forward_output(self, fwd_output):
       # Return the tensor that makes sense for the training loss
       return fwd_output[0]  # or fwd_output.logits, etc.
   ```
   See existing overrides in:
   ```bash
   grep -rn "def unpack_forward_output" third_party/tt_forge_models/ --include="*.py" -l
   ```

6. After applying all fixes, update their status in the YAML (see Step 4 below).

---

## Pattern 2 — dtype errors in bfloat16

**Errors:**
- `RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16`
- `RuntimeError: 'deformable_im2col' not implemented for 'BFloat16'`

These indicate either mixed-precision inputs or an op that PyTorch doesn't
support in bfloat16. The question is: does the model work at all in float32 on CPU?

### Workflow

1. Run the failing models on CPU in float32:
   ```bash
   python tools/triage_fe_failures.py --pattern dtype --run
   ```
   The script runs each loader with `dtype_override=torch.float32`.

2. Interpret results:
   - **No error** → the model works in float32. The issue is bf16 training.
     Check whether `load_inputs` or `load_model` in the loader casts tensors
     inconsistently. Look for mixed `float32`/`bfloat16` assignments.
     Fix: ensure all tensors use the same dtype throughout `load_inputs`.
   - **Same or different error** → the model has a deeper issue not related to
     dtype. Document the new error in the YAML `reason` field.

3. For `deformable_im2col`: this PyTorch op genuinely lacks bf16 support.
   If the model is otherwise correct, add a note in the YAML that it needs
   a custom op implementation or a float32 training path.

---

## Pattern 3 — missing training inputs

**Errors:**
- `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds.`
- `Model expects targets to be passed while in training mode`
- `AssertionError: targets should not be none when in training mode`
- `AttributeError: 'NoneType' object has no attribute 'max'`
- `ValueError: Expected more than 1 value per channel when training…`

These mean `load_inputs` doesn't pass what the model needs during training.
The model is being called the same way for inference and training, but training
requires extra tensors (ground-truth labels, decoder start tokens, etc.).

### Workflow

1. List all affected models:
   ```bash
   python tools/triage_fe_failures.py --pattern inputs
   ```

2. For each model, read its loader and model's forward signature:
   ```bash
   grep -n "def load_inputs\|def forward" third_party/tt_forge_models/<model>/pytorch/loader.py
   ```
   Then check the actual model class's `forward()` in the venv or HuggingFace
   source to see what arguments are required in training mode.

3. Fixes by sub-pattern:

   **`decoder_input_ids` / `decoder_inputs_embeds`** (seq2seq models like Whisper, T5-like):
   Add `decoder_input_ids` or `decoder_inputs_embeds` to the dict returned
   by `load_inputs`. A common approach is to use shifted labels or a
   `decoder_start_token_id`:
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

4. After fixing `load_inputs`, verify the fix runs on CPU:
   ```bash
   python -c "
   import sys; sys.path.insert(0, '.')
   from third_party.tt_forge_models.<model>.pytorch.loader import ModelLoader
   loader = ModelLoader()
   model = loader.load_model().train()
   inputs = loader.load_inputs()
   import torch
   with torch.enable_grad():
       out = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
       loss = loader.unpack_forward_output(out)
       loss.backward()
   print('OK:', type(out).__name__)
   "
   ```

5. Aggregate: after going through all missing-input models, look for common
   patterns (e.g., all detection models need the same target format). If a bulk
   fix applies (e.g., all YOLO variants), apply it in a shared base class or
   update multiple loaders at once.

---

## Step 4 — after fixing

For each fixed model, update the YAML directly after applying the fix. Do not
run pytest — just mark the entry as expected passing.

1. Remove `status: NOT_SUPPORTED_SKIP`, `bringup_status: FAILED_FE_COMPILATION`,
   and `reason: ...` from the YAML entry and set `status: EXPECTED_PASSING`:
   ```yaml
   model/path/pytorch-Variant-single_device-training:
     status: EXPECTED_PASSING
     markers: [nightly]
   ```

2. After updating the YAML, report a summary: which models were updated to
   `EXPECTED_PASSING` and which remain unchanged (i.e., the fix could not be
   determined).

---

## Reference files

- `third_party/tt_forge_models/training_utils.py` — handler registry
- `third_party/tt_forge_models/base.py` — `ForgeModel.unpack_forward_output`
- `tools/triage_fe_failures.py` — triage script
- `tests/runner/test_config/torch/test_config_training_single_device.yaml` — test config
- Existing loader overrides:
  ```bash
  grep -rn "def unpack_forward_output" third_party/tt_forge_models/ --include="*.py"
  ```
