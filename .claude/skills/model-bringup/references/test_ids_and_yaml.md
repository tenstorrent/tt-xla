# Test IDs, YAML keys, and runner wiring

This reference matches `tests/runner/validate_test_config.py`, `tests/runner/utils/dynamic_loader.py`, and `tests/runner/conftest.py`.

## `rel_path` (loader path prefix)

Test discovery builds a **base string** from each `loader.py` path:

- `rel_path` = `os.path.relpath(os.path.dirname(loader.py), models_root)` where `models_root` is `third_party/tt_forge_models`.
- **Do not** assume the key starts with `<top_level_model>/pytorch`. Nested layouts include the full path, e.g.:
  - `llama/causal_lm/pytorch`
  - `qwen_2_5/causal_lm/pytorch`
  - `resnet50/image_classification/pytorch`

**Discovery rules:** only trees with a directory literally named `pytorch` containing `loader.py` are picked up (`TorchDynamicLoader`). Alternate folders are **not** auto-discovered.

## Canonical test IDs for `test_all_models_torch`

```
{rel_path}-{variant}-{parallelism}-{run_mode}
```

- `variant`: the **string value** of the `ModelVariant` enum member (e.g. `3.0_8B`, `3.1_8B_Instruct`). Omit the segment only if the loader has **no** variants (`_VARIANTS` empty).
- `parallelism`: `single_device` | `data_parallel` | `tensor_parallel`
- `run_mode`: **`inference`** (model bringup targets inference only — do not use `training` unless explicitly requested)

**Examples:**

```text
llama/causal_lm/pytorch-3.1_8B-single_device-inference
resnet50/image_classification/pytorch-Default-single_device-inference
```

YAML keys under `test_config:` must match the **full** string inside the pytest parametrized bracket (see `pytest_collection_modifyitems` in `tests/runner/conftest.py`).

**Important:** Always register new models with `status: EXPECTED_PASSING` initially — never `KNOWN_FAILURE_XFAIL`. The xfail marker hides actual errors in pytest output, making bringup debugging impossible. Only downgrade to `KNOWN_FAILURE_XFAIL` after observing and diagnosing a specific failure.

## PyTorch LLMs: `test_llms_torch` (decode / prefill)

If the loader implements `load_inputs_decode` and/or `load_inputs_prefill`, pytest also collects **`test_llms_torch`** with **different** IDs.

**Decode (single sequence length / batch):**

```text
{rel_path}-{variant}-llm_decode-seq_1-batch_1-{parallelism}-inference
```

**Prefill (multiple seq/batch combinations):**

```text
{rel_path}-{variant}-llm_prefill-seq_{seq}-batch_{batch}-{parallelism}-inference
```

Register matching entries in `tests/runner/test_config/torch_llm/test_config_inference_*.yaml` **in addition** to any entries in `torch/` for `test_all_models_torch`.

## Imports for CPU smoke tests

Use the **Python package path** that mirrors the directory layout:

```python
from third_party.tt_forge_models.llama.causal_lm.pytorch import ModelLoader
```

Run with repo root on `PYTHONPATH` (or `pytest` from tt-xla as usual).

## Submodule workflow

Loaders live under the **`third_party/tt_forge_models`** git submodule (upstream [tt-forge-models](https://github.com/tenstorrent/tt-forge-models)). When landing changes:

1. Commit in the **submodule** repository on the appropriate branch.
2. In **tt-xla**, commit the updated submodule pointer (and any tt-xla-only test YAML changes).

## Per-loader `requirements.txt`

`tests/runner/requirements.py` (`RequirementsManager.for_loader`) installs **`requirements.txt` placed next to `loader.py`** (same directory) before the test runs.

If the model needs extra pip packages, add `requirements.txt` beside `loader.py` so CI and local runs stay reproducible.

## TorchDynamicLoader and `dtype_override`

`TorchDynamicLoader` (`tests/runner/utils/dynamic_loader.py`) calls `load_model` / `load_inputs` with **`dtype_override=torch.bfloat16`** when those methods declare a `dtype_override` parameter. Device/reference comparison therefore typically uses **bfloat16**, not float32 — important for PCC expectations and for matching existing models.
