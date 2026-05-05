---
name: triage-unpack-forward-output
description: Triage and auto-fix one tt-forge-models training test stuck at FAILED_FE_COMPILATION with reason "tt-forge-models doesn't implement unpack_forward_output for this model." Runs the model on CPU, inspects forward outputs, gathers model docs, writes a documented unpack_forward_output override on the loader, re-runs the training and inference tests, and updates the YAML status. Operates on exactly one test at a time.
allowed-tools: Bash Read Edit Write Grep Glob WebFetch
argument-hint: <test-name>
---

# Triage `unpack_forward_output` training failures

This skill handles **one** failure pattern only:

> `tt-forge-models doesn't implement unpack_forward_output for this model.`

It works on exactly one test (passed as `<test-name>`). If the failure reason is anything else, abort immediately.

## Argument: `<test-name>` accepted forms

Auto-detect both:

- **YAML key** — e.g. `yolov9/pytorch-T-single_device-training`. Looks like `<model_dir>/<framework>-<variant>-<parallelism>-<run_mode>`.
- **`model_info.name`** — e.g. `pytorch_YOLOv9_T_cv_object_det_github`. Composed in `third_party/tt_forge_models/config.py` as `f"{framework}_{model}_{variant}_{task}_{source}"`.

## Background (load into context, do not paraphrase to the user)

- Default `unpack_forward_output` lives in `third_party/tt_forge_models/base.py` and delegates to `third_party/tt_forge_models/training_utils.py`. The util has a registry keyed by output class name (`BaseModelOutputWithPast`, `CausalLMOutputWithPast`, `CLIPOutput`, `ImageClassifierOutput`, …).
- When the model returns a class **not in the registry** (a bare `list`/`tuple`/`dict`, or a custom dataclass), the call raises `ValueError("No handler for class … exists in unpack_forward_output. Register a handler or implement custom unpack_forward_output for the specific model.")`. The runner then surfaces the failure with `bringup_status: FAILED_FE_COMPILATION` and the canonical `reason:` shown above.
- The fix is per-model: override `unpack_forward_output` on that model's `ModelLoader` to return only the tensors that participate in the training loss.
- Existing override examples to mimic:
  - `third_party/tt_forge_models/yolov9/pytorch/loader.py:171-194` — list-of-tensors via `extract_tensors_recursive`
  - `third_party/tt_forge_models/centernet/pytorch/loader.py:190-209` — list-of-dicts
  - `third_party/tt_forge_models/clip/pytorch/loader.py:209-220` — tuple
- Test-runner glue: `tests/runner/test_models.py` parametrizes `test_all_models_torch` with `(test_entry, parallelism, run_mode)`. Pytest test IDs combine in stack order: `<test_entry_id>-<parallelism_id>-<run_mode_id>`. The `<test_entry_id>` is `<model_path>-<variant>` (see `tests/runner/utils/dynamic_loader.py:319`).

## Workflow

### Step 1 — Resolve `<test-name>` → YAML entry + loader path

1. If `<test-name>` matches `^[a-z0-9_]+/.+-(single_device|data_parallel|tensor_parallel)-(training|inference)$`, treat as a YAML key. Otherwise treat as a `model_info.name` string.
2. **YAML key path:** locate the entry in `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Loader path is `third_party/tt_forge_models/<model_dir>/pytorch/loader.py` where `<model_dir>` is the segment before the first `/` in the key.
3. **`model_info.name` path:** parse as `<framework>_<model>_<variant>_<task>_<source>`. The `framework` is always `pytorch` for this skill. Search loaders with: `grep -rln "model=.*\"<model>\"\|ModelName.*<model>" third_party/tt_forge_models/`, then for each candidate, read its `_get_model_info` to confirm the variant/task/source line up. Once the loader is identified, derive the YAML key as `<model_dir>/pytorch-<variant>-single_device-training` and confirm it exists in the YAML.
4. If neither resolves, abort with: `Could not resolve <test-name> to a YAML entry or loader. Provide either the YAML key or the model_info.name.`

### Step 2 — Gate on the failure pattern

1. Read the YAML entry. Its `reason:` must contain the substring `unpack_forward_output`. If not, abort: `This skill only handles unpack_forward_output failures. The failing reason here is: <reason>.`
2. `grep -n "def unpack_forward_output(" <loader_path>`. If the override **already exists**, skip to Step 6 — the YAML entry may simply be stale. Note this in the final report.

### Step 3 — CPU triage script

Write `/tmp/triage_<model_dir>_<variant>.py`:

```python
import sys, torch
sys.path.insert(0, "/localdev/<user>/tt-xla")
from third_party.tt_forge_models.<model_dir>.pytorch.loader import ModelLoader, ModelVariant

loader = ModelLoader(variant=ModelVariant.<VARIANT>)  # omit variant arg if loader has no ModelVariant
model = loader.load_model().eval()
inputs = loader.load_inputs()

with torch.no_grad():
    out = model(**inputs) if isinstance(inputs, dict) else (
        model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
    )

def describe(x, depth=0, name="out"):
    pad = "  " * depth
    if isinstance(x, torch.Tensor):
        print(f"{pad}{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype}")
    elif isinstance(x, dict):
        print(f"{pad}{name}: dict({len(x)} keys)")
        for k, v in x.items():
            describe(v, depth+1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {type(x).__name__}({len(x)})")
        for i, v in enumerate(x):
            describe(v, depth+1, f"[{i}]")
    else:
        attrs = [a for a in dir(x) if not a.startswith('_') and isinstance(getattr(x, a, None), (torch.Tensor, list, tuple, dict))]
        print(f"{pad}{name}: {type(x).__name__} attrs={attrs}")
        for a in attrs:
            describe(getattr(x, a), depth+1, a)

describe(out)
```

Run it: `python /tmp/triage_<model_dir>_<variant>.py &> /tmp/triage_<model_dir>_<variant>.log` (cwd must be the tt-xla repo root, with `source venv/activate` already done).

**Read the log with the Read tool. Never use `tail` or `less` — they hide errors behind generic exit codes** (this is a hard rule from feedback memory).

If the script errored out before producing structure (e.g. `load_inputs` requires a non-default kwarg), adjust the script (try `dtype_override=torch.bfloat16`, or `seq_len`/`batch_size` if `load_inputs` accepts them — see `tests/runner/utils/dynamic_loader.py:380-419` for what the test runner passes) and re-run.

### Step 4 — Gather model docs

From the loader file, extract the model's HuggingFace id (`pretrained_model_name` in `_VARIANTS`), GitHub URL or paper if commented at the top. Use `WebFetch` on the model card / repo README to identify what training loss conventionally consumes.

Defaults if docs are silent or model fits a known family:

| Family / category                | Loss-relevant output(s)                                            |
| -------------------------------- | ------------------------------------------------------------------ |
| Causal / masked / seq2seq LMs    | `logits`                                                           |
| Image classification             | `logits`                                                           |
| Object detection (YOLO, DETR…)   | concatenated detection-head tensors (per-scale class+box logits)   |
| Semantic / instance segmentation | per-pixel `logits` (drop auxiliary FPN side outputs)               |
| Depth / dense regression         | the regressed tensor (e.g. `predicted_depth`)                      |
| Encoder-only feature extractors  | `last_hidden_state` (or `pooler_output` if that's what loss reads) |

### Step 5 — Write the `unpack_forward_output` override

Add the method to the `ModelLoader` class in the loader file. Two non-negotiable rules:

1. **Return only what the loss depends on.** Never include auxiliary outputs (attention weights, hidden-state caches, FPN side branches, raw anchors, debug tensors) just because they are tensors. Including unused tensors inflates the autograd graph and changes what is exercised on TT.
2. **Docstring with these three sections.** The override must carry them — they are the audit trail for why this override exists:
   - **Forward output structure** — full type/shape signature of what the model returns. Example: `list[Tensor] of length 3, shapes [(B, 256, 80, 80), (B, 256, 40, 40), (B, 256, 20, 20)]`.
   - **What is selected and why** — which subset is returned and which loss it is the gradient source of. Example: `returning the three detection-head tensors concatenated; these are the only outputs consumed by YOLOLoss; auxiliary feature maps are dropped`.
   - **Why the override is needed at all** — one line. Example: `default registry has no handler for list[Tensor]`.

Implementation guidance:

- For nested list/tuple/dict structures: import `extract_tensors_recursive` from `third_party.tt_forge_models.tools.utils`, walk the loss-relevant subtree, and `torch.cat(tensors, dim=0)` (or stack) — see `yolov9/pytorch/loader.py:171-194`.
- For dataclass-shaped HF-style outputs not in the registry: pick the loss-relevant attribute (typically `logits`) and return it directly.
- For a plain tuple where only one element is the loss target: index it.
- Keep the implementation minimal. No extra abstractions, no inline comments beyond the required docstring.

After writing, run `pre-commit run --files <loader_path>` if pre-commit is installed; otherwise just confirm `python -c "from third_party.tt_forge_models.<model_dir>.pytorch.loader import ModelLoader"` succeeds.

### Step 6 — Verify with pytest

Run two commands. **No `less`. No `tail`. No piping into anything that hides exit codes.** Dump straight to a file and Read it back:

```bash
timeout 300 pytest tests/runner/test_models.py::test_all_models_torch[<model_dir>/pytorch-<variant>-single_device-training] -svv &> /tmp/verify_<model_dir>_<variant>_training.log
timeout 300 pytest tests/runner/test_models.py::test_all_models_torch[<model_dir>/pytorch-<variant>-single_device-inference] -svv &> /tmp/verify_<model_dir>_<variant>_inference.log
```

Read both logs.

### Step 7 — Classify the training outcome

- **Pass:** outcome = `EXPECTED_PASSING`.
- **Fail:** classify the new error and pick the matching `bringup_status`:

  | Symptom                                                              | `bringup_status`             |
  | -------------------------------------------------------------------- | ---------------------------- |
  | Crash in TTNN runtime / device assert / runtime stack trace          | `FAILED_RUNTIME`             |
  | Compiler error after the frontend (TTIR / StableHLO / TTNN compile)  | `FAILED_TTMLIR_COMPILATION`  |
  | Numerics divergence (PCC / atol failure, comparison output)          | `INCORRECT_RESULT`           |

- **If the log shows `Error code 13` or any generic exit-code line, that is NOT the real error.** `grep -nE "TT_FATAL|TT_THROW" /tmp/verify_<model_dir>_<variant>_training.log` and use the first matching line (and a few lines around it for context) as the real error. Pick the `bringup_status` from that real error, not from "Error code 13".

### Step 8 — Update the YAML

Edit `tests/runner/test_config/torch/test_config_training_single_device.yaml`. Modify only the **training** entry:

- **Pass case:**
  ```yaml
  <model_dir>/pytorch-<variant>-single_device-training:
    status: EXPECTED_PASSING
  ```
  Drop `bringup_status` and `reason` entirely.
- **Fail case:** keep the existing `status:` (`KNOWN_FAILURE_XFAIL` if it was that, otherwise `NOT_SUPPORTED_SKIP`), set `bringup_status:` to the new category, set `reason:` to a one-line excerpt of the real error (the `TT_FATAL`/`TT_THROW` line if applicable, trimmed to ~120 chars).

The inference log is **informational only**. Do not modify the inference YAML entry. Flag any inference regression to the user in the final report so they can investigate separately.

### Step 9 — Final report

Print a single concise summary to the user:

1. Resolved test → loader path + YAML key
2. Forward output structure observed on CPU (one or two lines)
3. Override added: file, method signature, what it returns
4. Pytest outcome — training pass/fail (with real error if fail), inference pass/fail
5. YAML diff (the few lines that changed)
6. Anything that needs human review (e.g. inference regression, ambiguous loss target, `extract_tensors_recursive` returned an unexpected number of tensors).

## Hard rules (do not violate)

- One test at a time. If `<test-name>` is missing or matches multiple entries, abort and ask.
- Only act when `reason:` contains `unpack_forward_output`. Anything else: abort, do not edit.
- Override must return only loss-relevant tensors and must carry the three-part docstring above.
- Never use `tail` or `less` on pytest output. Dump to a file with `&>`, then Read.
- "Error code 13" is generic — always look further for `TT_FATAL` / `TT_THROW`.
- Touch only the training YAML entry. Inference is informational.
- Do not commit or push. Leave changes staged-but-uncommitted unless the user asks otherwise.
