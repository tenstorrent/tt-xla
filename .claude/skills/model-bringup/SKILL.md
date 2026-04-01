---
name: model-bringup
description: Guides end-to-end bringup of a new model on Tenstorrent hardware via tt-xla. Use when a user wants to add, run, diagnose, or promote a model through the bringup pipeline — from initial discovery through CI integration.
allowed-tools: Bash Read Grep Glob Write Edit Agent WebSearch WebFetch
---

# Model Bringup

Systematic workflow for bringing up models on Tenstorrent hardware through the tt-xla PJRT backend. Covers the full lifecycle: discovery, first run, failure diagnosis, fix iteration, test configuration, and CI promotion.

## When to use this skill

- User wants to bring up a new model on TT hardware
- User wants to diagnose why a model is failing (compilation, runtime, accuracy)
- User wants to promote a model from experimental to nightly CI
- User wants to add test configuration for a model
- User wants to understand the current bringup status of a model

## Prerequisites

- TT-XLA environment activated (`source venv/activate`, NOT `source venv/bin/activate`)
- Build is ready (`cmake -G Ninja -B build && cmake --build build`)
- `third_party/tt_forge_models` submodule is initialized (`git submodule update --init --recursive third_party/tt_forge_models`)

## Step 1: Understand the request

Ask the user (if not already clear):
1. **Model name** — e.g., "llama 3.1 8B", "resnet50", "whisper large"
2. **Framework** — PyTorch (default) or JAX
3. **Goal** — first bringup, diagnose failure, promote to CI, or add new variant
4. **Parallelism** — single_device (default), data_parallel, or tensor_parallel
5. **Run mode** — inference (default) or training

## Step 2: Check current state and size the model

### 2a. Check if model exists in tt-forge-models

Search for the model loader:
```bash
# PyTorch models
find third_party/tt_forge_models -path "*/<model_name>*/pytorch/loader.py" 2>/dev/null

# JAX models
find third_party/tt_forge_models -path "*/<model_name>*/jax/loader.py" 2>/dev/null
```

If the loader does NOT exist, the model must first be added to the `tt-forge-models` repository. See `references/adding_model_to_forge_models.md` for guidance. Inform the user this is a prerequisite.

### 2b. Check if model is already discovered

```bash
source venv/activate && pytest -q --collect-only -k "<model_name>" tests/runner/test_models.py 2>&1 | head -30
```

### 2c. Check current test configuration

Search across all YAML config files:
```bash
grep -r "<model_path>" tests/runner/test_config/ --include="*.yaml"
```

Also check placeholders:
```bash
grep -i "<model_name>" tests/runner/test_config/torch/test_config_placeholders.yaml
```

### 2d. Check bringup status summary

If the model has existing config, note:
- `status` — EXPECTED_PASSING, KNOWN_FAILURE_XFAIL, NOT_SUPPORTED_SKIP, UNSPECIFIED
- `bringup_status` — PASSED, INCORRECT_RESULT, FAILED_FE_COMPILATION, FAILED_TTMLIR_COMPILATION, FAILED_RUNTIME, NOT_STARTED
- `reason` — linked GitHub issues

### 2e. Size the model and determine parallelism strategy

**This is critical before running anything.** Calculate model size and compare to device memory:

```python
model = loader.load_model()
total_params = sum(p.numel() for p in model.parameters())
size_bf16_gb = (total_params * 2) / (1024**3)
print(f"Parameters: {total_params:,}, Size (bf16): {size_bf16_gb:.1f} GB")
```

Device memory budgets:
- **Wormhole (WH)** — n150, p150, n300: **12 GB/chip** (~8.4 GB usable)
- **Blackhole (BH)**: **32 GB/chip** (~22 GB usable)

**Decision**:
- `model_size < 0.7 * device_dram` → **single_device** is fine
- `model_size >= 0.7 * device_dram` → needs **tensor_parallel** or **bfp8 weight conversion**
- If borderline, try `enable_weight_bfp8_conversion: true` in YAML before adding TP

If tensor parallel is needed, the loader must implement `get_mesh_config()` and `load_shard_spec()`. See `references/model_sizing_and_sharding.md` for the full sharding guide.

**Key principle: minimize CCLs.** Only shard what's needed to fit in memory. If you have spare capacity after sharding, increase batch size instead of sharding further — batch parallelism is free, CCLs are not.

## Step 3: Run the model test

### 3a. Identify the test ID

Test IDs follow the format: `<relative_model_path>-<variant>-<parallelism>-<run_mode>`

Examples:
- `resnet/pytorch-ResNet50_HuggingFace-single_device-inference`
- `llama/causal_lm/pytorch-3.1_8B-tensor_parallel-inference`

### 3b. For LLMs: start with reduced layers using the benchmark test infra

**For LLM/transformer models, always start bringup with a reduced number of layers (e.g., 1-2) before running the full model.** This dramatically reduces compilation and runtime latency, letting you iterate on failures much faster. A single-layer run exercises the same ops and graph structure as the full model — most compilation and runtime failures reproduce identically.

The **benchmark test infra** (`tests/benchmark/`) supports `--num-layers` as a pytest option. This is the preferred way to do reduced-layer bringup — use the existing test infrastructure, not standalone scripts.

#### Step 1: Check if the loader supports `num_layers`

```bash
grep -n "num_layers" third_party/tt_forge_models/<model_path>/pytorch/loader.py
```

If it does NOT support `num_layers`, add it to the loader before proceeding. See `references/llm_layer_reduction.md` for the established pattern.

#### Step 2: Check if a benchmark test exists for this model

```bash
grep -rn "<model_name>\|<ModelLoader>" tests/benchmark/test_llms.py tests/benchmark/test_encoders.py
```

If a benchmark test exists, run it with `--num-layers`:
```bash
source venv/activate
pytest -svv tests/benchmark/test_llms.py::test_<model_name> --num-layers 1 2>&1 | tee /tmp/model_bringup_1layer.log
```

If no benchmark test exists yet, add one to the appropriate file (`test_llms.py` for LLMs, `test_encoders.py` for encoder models). Follow the existing test patterns — they use `create_model_loader(ModelLoader, num_layers=num_layers)` from `tests/benchmark/utils.py` which handles the `num_layers` fixture injection.

#### Step 3: Iterate with increasing layers

1. `--num-layers 1` — fix any compilation or runtime errors (fastest iteration)
2. `--num-layers 2` — verify multi-layer stacking (catches shape issues between layers)
3. Full model run via the runner infra (no `--num-layers`) — final accuracy validation (PCC)

Only PCC/accuracy validation requires the full model. Compilation and runtime issues almost always reproduce with 1-2 layers.

**Note**: The `--num-layers` option is in the benchmark conftest (`tests/benchmark/conftest.py`), NOT in the runner conftest. The runner infra (`tests/runner/test_models.py`) always runs the full model — use it for the final validation step.

### 3c. Run the full model test via the runner infra

Once reduced-layer tests pass, run the full model through the standard test runner:

```bash
source venv/activate
pytest -svv tests/runner/test_models.py::test_all_models[<test_id>] 2>&1 | tee /tmp/model_bringup.log
```

For debug-level output (needed for deeper diagnosis):
```bash
source venv/activate
TTXLA_LOGGER_LEVEL=DEBUG TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG XLA_HLO_DEBUG=1 \
  pytest -svv tests/runner/test_models.py::test_all_models[<test_id>] 2>&1 | tee /tmp/model_bringup_debug.log
```

For graph break analysis, also add `TORCH_LOGS="+dynamo"`.

### 3d. Check the result

If the test passes, proceed to Step 5 (Configure test YAML).
If it fails, proceed to Step 4 (Diagnose failure).

## Step 4: Diagnose failure

Classify the failure into one of the bringup stages. See `references/failure_diagnosis.md` for detailed guidance.

### Failure classification

| Stage | Indicators | BringupStatus |
|-------|-----------|---------------|
| **Frontend compilation** | Errors during torch.compile/dynamo tracing, StableHLO lowering | `FAILED_FE_COMPILATION` |
| **TT-MLIR compilation** | Errors in ttir/ttnn lowering, tt-mlir compiler crashes | `FAILED_TTMLIR_COMPILATION` |
| **Runtime execution** | L1 OOM, circular buffer errors, device errors, hangs | `FAILED_RUNTIME` |
| **Accuracy** | PCC mismatch, ATOL mismatch, NaN outputs | `INCORRECT_RESULT` |

### Diagnosis workflow

1. **Read the error** — Start from the bottom of the log and work upward
2. **Search for known issues** — `grep` the error message across GitHub issues:
   ```bash
   gh search issues "<key_error_phrase>" --repo tenstorrent/tt-xla --limit 5
   gh search issues "<key_error_phrase>" --repo tenstorrent/tt-mlir --limit 5
   ```
3. **Check failing reasons database** — Look at `tests/infra/utilities/failing_reasons/` for classified patterns
4. **For graph breaks** — Use the `/graph-break-analysis` skill instead
5. **For accuracy issues** — Check if it's a dtype/precision issue, compare against CPU baseline

### Common fixes

- **Unsupported op** → Check if it's a known gap in tt-mlir, file issue if new
- **L1 OOM** → Try smaller batch size, check if model fits on target arch
- **Graph breaks** → Use `/graph-break-analysis` for systematic analysis
- **PCC mismatch** → Lower `required_pcc` threshold if within acceptable range, or investigate precision loss
- **Import errors** → Check per-model `requirements.txt`

## Step 5: Configure test YAML

### 5a. Determine the correct YAML file

Config files are in `tests/runner/test_config/` organized by framework and mode:

| Framework | Directory |
|-----------|-----------|
| PyTorch (standard) | `tests/runner/test_config/torch/` |
| PyTorch (LLM) | `tests/runner/test_config/torch_llm/` |
| JAX | `tests/runner/test_config/jax/` |

File naming pattern: `test_config_<run_mode>_<parallelism>.yaml`

Examples:
- `torch/test_config_inference_single_device.yaml`
- `torch/test_config_inference_tensor_parallel.yaml`
- `torch/test_config_training_single_device.yaml`

### 5b. Add the YAML entry

Add under the `test_config:` key. See `references/yaml_config_reference.md` for all fields and examples.

**Passing model:**
```yaml
  model_path/pytorch-variant-parallelism-mode:
    status: EXPECTED_PASSING
    required_pcc: 0.99  # adjust based on actual measured PCC
```

**Failing model (known failure):**
```yaml
  model_path/pytorch-variant-parallelism-mode:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: FAILED_TTMLIR_COMPILATION
    reason: "Error description - https://github.com/tenstorrent/tt-xla/issues/NNNN"
```

**Architecture-specific:**
```yaml
  model_path/pytorch-variant-parallelism-mode:
    status: EXPECTED_PASSING
    arch_overrides:
      n150:
        status: NOT_SUPPORTED_SKIP
        reason: "Too large for single chip"
```

### 5c. Remove placeholder if applicable

If the model appears in `tests/runner/test_config/torch/test_config_placeholders.yaml`, remove it.

### 5d. Validate configuration

```bash
source venv/activate && pytest -q --validate-test-config tests/runner/test_models.py
```

Fix any validation errors before proceeding.

## Step 6: Verify end-to-end

1. Re-run the test with the new configuration:
   ```bash
   pytest -svv tests/runner/test_models.py::test_all_models[<test_id>] 2>&1 | tee /tmp/model_verify.log
   ```
2. Confirm the test status matches the YAML configuration (passes if EXPECTED_PASSING, xfails if KNOWN_FAILURE_XFAIL, etc.)
3. Run config validation one more time.

## Step 7: Summary report

After completing bringup, provide the user with:

1. **Model**: name, variant, framework
2. **Test ID**: full parametrized test ID
3. **Status**: current bringup_status and test status
4. **PCC/ATOL**: measured accuracy metrics (if applicable)
5. **Issues**: any GitHub issues filed or linked
6. **Config file**: which YAML file was modified and what was added
7. **Next steps**: what remains to land this (PR, submodule uplift, CI promotion)

## Reference Documentation

- `references/yaml_config_reference.md` — Complete YAML field reference with examples
- `references/failure_diagnosis.md` — Detailed failure classification and diagnosis patterns
- `references/adding_model_to_forge_models.md` — Guide for adding a model to tt-forge-models
- `references/llm_layer_reduction.md` — `num_layers` pattern for fast LLM bringup iteration
- `references/model_sizing_and_sharding.md` — Device memory budgets, model sizing, and sharding strategy (minimize CCLs)
