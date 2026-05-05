---
name: model-bringup-scaffold
description: Scaffold and validation stage of the model bringup pipeline. Validates that a model_key has a loader and is importable. If the loader is missing, creates loader.py and package __init__.py files following the pattern of existing models. Initializes state.json. Invoked by the model-bringup orchestrator at the VALIDATE stage.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — Scaffold & Validation

You are the **scaffold and validation** stage of the model bringup pipeline.

## Invocation
`/model-bringup-scaffold <model_key> [--arch <arch>]`

## Responsibility
Validate that the model_key is ready to run before any test execution begins.
If any file is missing, **create it** following the conventions of existing models.
Initialize the bringup state directory and state.json.

---

## Step 1 — Parse model_key

The model_key may be in one of two formats:

**Format A — structured key** (preferred):
```
<family>/<framework>-<variant>-<parallelism>-<run_mode>
```
Example: `ltx2/pytorch-Fast-single_device-inference`
- `family` = `ltx2`
- `framework` = `pytorch`
- `variant` = `Fast`
- `parallelism` = `single_device`
- `run_mode` = `inference`

**Format B — HuggingFace model ID** (e.g. `google/bert_uncased_L-2_H-128_A-2`):
- Detect: Part 1 does not match `<framework>-<variant>-<parallelism>-<run_mode>`
- Normalize: derive `family` from the model name (e.g. `bert_tiny`), set `framework=pytorch`, `parallelism=single_device`, `run_mode=inference`, derive `variant` from the repo name
- Synthesize the structured key and continue

---

## Step 2 — Locate or create the loader

Check for `third_party/tt_forge_models/<family>/pytorch/loader.py`.

**If loader exists:** proceed to Step 3.

**If loader is absent:** create the full model directory structure (see "Creating a new loader" below), then proceed to Step 3.

---

### Creating a new loader

#### 2a. Find a reference model
Find the most similar existing model to use as a template:
- For text/NLP models: look at `bert/`, `gpt_neo/`, `llama/`
- For image models: look at `resnet/`, `clip/`
- For video/diffusion: look at `flux/`, `ltx2/`
- Use `find third_party/tt_forge_models -name loader.py | head -5` to browse

Read the reference `loader.py` in full before writing anything.

#### 2b. Inspect the HuggingFace model
Use `python -c` to inspect the model:
```python
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("<hf_model_id>")
print(config)
model = AutoModel.from_pretrained("<hf_model_id>")
print(type(model))
# Print forward signature
import inspect
print(inspect.signature(model.forward))
```
Record: model class, input names, output structure, hidden dim, num layers.

#### 2c. Create directory structure
```
third_party/tt_forge_models/<family>/
├── __init__.py
└── pytorch/
    ├── __init__.py          ← re-exports ModelLoader, ModelVariant
    └── loader.py            ← model loader
```

No `tests/` subdirectory is needed. `tests/runner/test_models.py` discovers models
automatically via the dynamic loader from `loader.py` alone.

#### 2d. Write `loader.py`
Follow the reference model's structure exactly:
- `ModelVariant(StrEnum)` — one entry per checkpoint; name the variant after the HF repo slug (e.g. `L2_H128_A2`)
- `ModelConfig` or `LLMModelConfig` — set `pretrained_model_name`
- `ModelLoader(ForgeModel)`:
  - `_get_model_info()` — use appropriate `ModelTask` (e.g. `NLP_MASKED_LM`, `MM_VIDEO_TTT`)
  - `load_model()` — load from HuggingFace with `torch_dtype=torch.bfloat16`
  - `load_inputs()` — generate synthetic inputs matching the model's `forward()` signature
  - `unpack_forward_output()` — extract tensor from dataclass/tuple/dict output
- SPDX header at top

#### 2e. Write `pytorch/__init__.py`
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .loader import ModelLoader, ModelVariant
```

---

## Step 3 — Validate imports

Run:
```bash
python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant; print('OK')"
```

If this fails, fix the import error before proceeding.

---

## Step 4 — Validate discoverability via collect

Confirm the model is visible to `tests/runner/test_models.py`:
```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep "test_all_models_torch\[<family>/pytorch-"
```

If no lines appear, the loader failed to import during collection — check the import error in the collect output and fix it before proceeding.

---

## Step 5 — Initialize bringup state

Create `.claude/bringup/<safe_model_key>/` with subdirectories `logs/` and `patches/`.

Write `state.json`:
```json
{
  "model_key": "<model_key>",
  "arch": "<arch>",
  "stage": "validate",
  "iteration": 0,
  "history": [],
  "applied_patches": [],
  "failure_reasons": [],
  "created_at": <unix_timestamp>,
  "updated_at": <unix_timestamp>
}
```

Append history entry: `{ "stage": "validate", "result": "passed", "details": { "loader_path": "third_party/tt_forge_models/<family>/pytorch/loader.py", "loader_created": true|false } }`.

---

## Step 6 — Write to bringup_steps.txt

Append a section to `.claude/bringup/<safe_key>/bringup_steps.txt`.
If the file does not exist yet (first stage), write the header block first:
```
================================================================================
MODEL BRINGUP LOG
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
================================================================================
```

Then append:
```
--------------------------------------------------------------------------------
STEP 1 — Parse & Scaffold (model-bringup-scaffold)
--------------------------------------------------------------------------------
Input model_key : <original model_key>
Format detected : A (structured) | B (HuggingFace model ID)
Normalized to   : family=<family>  variant=<variant>  parallelism=<p>  run_mode=<r>

Loader path     : third_party/tt_forge_models/<family>/pytorch/loader.py
Loader created  : yes | no

[If created:]
  Reference model : <reference loader path>
  Model class     : <class name>
  HF model ID     : <id>
  Input signature : <key input fields>
  Files written   :
    - third_party/tt_forge_models/<family>/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/loader.py

Import validation  : python -c "from ... import ModelLoader, ModelVariant" → OK | FAILED
Collect validation : pytest --collect-only tests/runner/test_models.py | grep '<family>/pytorch-' → <N> test(s) found | NONE FOUND

SCAFFOLD RESULT: PASSED | FAILED
```

## Step 7 — Output

On success:
```
[scaffold] PASSED
  loader:         third_party/tt_forge_models/<family>/pytorch/loader.py
  collect check:  <N> test(s) visible in tests/runner/test_models.py
  loader_created: yes | no
```

Only exit with failure (and let the orchestrator escalate) if:
- The HF model ID is unreachable / does not exist
- The model inspection (`AutoModel.from_pretrained`) fails with an unrecoverable error
- The import validation still fails after creating the loader (syntax/logic error)
