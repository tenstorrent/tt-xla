# YAML Test Config Reference

Complete reference for model test configuration entries in `tests/runner/test_config/**/*.yaml`.

## File organization

```
tests/runner/test_config/
├── torch/
│   ├── test_config_inference_single_device.yaml
│   ├── test_config_inference_data_parallel.yaml
│   ├── test_config_inference_tensor_parallel.yaml
│   ├── test_config_training_single_device.yaml
│   ├── test_config_training_tensor_parallel.yaml
│   └── test_config_placeholders.yaml
├── torch_llm/
│   ├── test_config_inference_single_device.yaml
│   └── test_config_inference_tensor_parallel.yaml
└── jax/
    ├── test_config_inference_single_device.yaml
    ├── test_config_inference_data_parallel.yaml
    ├── test_config_inference_tensor_parallel.yaml
    └── test_config_training_single_device.yaml
```

## Test ID format

```
<relative_model_path>-<variant>-<parallelism>-<run_mode>
```

- `relative_model_path`: Path relative to `third_party/tt_forge_models/`, with `/pytorch` or `/jax` suffix replaced by just the framework prefix in the ID
- `variant`: From `ModelLoader.query_available_variants()`
- `parallelism`: `single_device`, `data_parallel`, or `tensor_parallel`
- `run_mode`: `inference` or `training`

Examples:
- `resnet/pytorch-ResNet50_HuggingFace-single_device-inference`
- `llama/causal_lm/pytorch-3.1_8B-tensor_parallel-inference`
- `bert/question_answering/pytorch-bert_base_uncased-data_parallel-inference`

## All allowed fields

### Status and metadata

| Field | Type | Description |
|-------|------|-------------|
| `status` | enum | **Required.** `EXPECTED_PASSING`, `KNOWN_FAILURE_XFAIL`, `NOT_SUPPORTED_SKIP`, `UNSPECIFIED`, `EXCLUDE_MODEL` |
| `bringup_status` | enum | Health for dashboard. `PASSED`, `INCORRECT_RESULT`, `FAILED_FE_COMPILATION`, `FAILED_TTMLIR_COMPILATION`, `FAILED_RUNTIME`, `NOT_STARTED`, `UNKNOWN` |
| `reason` | string | Human-readable context, ideally with GitHub issue link |
| `markers` | list[string] | Pytest markers: `push`, `nightly`, `extended`, etc. |
| `supported_archs` | list[string] | Architectures: `n150`, `p150`, `n300`, `n300-llmbox`, `galaxy-wh-6u` |

### Comparator controls

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `required_pcc` | float | 0.99 | Pearson correlation coefficient threshold |
| `assert_pcc` | bool | true | Set `false` to disable PCC check (temporary measure) |
| `required_atol` | float | null | Absolute tolerance threshold |
| `assert_atol` | bool | false | Enable absolute tolerance checking |
| `assert_allclose` | bool | false | Use numpy `allclose` instead of PCC |
| `allclose_rtol` | float | null | Relative tolerance for allclose |
| `allclose_atol` | float | null | Absolute tolerance for allclose |

### Test behavior

| Field | Type | Description |
|-------|------|-------------|
| `batch_size` | int | Override batch size (mainly for LLM tests) |
| `execution_pass` | string | `FORWARD` or `BACKWARD` (for training tests) |
| `enable_weight_bfp8_conversion` | bool | Enable bfp8 weight conversion |
| `inject_custom_moe` | bool | Use custom MoE for sparse models |
| `filechecks` | list[string] | FileCheck pattern files to verify IR |

### Architecture overrides

```yaml
arch_overrides:
  <arch_name>:
    # Any of the above fields can be overridden per-arch
    status: ...
    required_pcc: ...
    reason: ...
```

## Examples

### Passing model (minimal)

```yaml
  resnet/pytorch-ResNet50_HuggingFace-single_device-inference:
    status: EXPECTED_PASSING
```

### Passing with tuned PCC

```yaml
  hardnet/pytorch-hardnet68-single_device-inference:
    required_pcc: 0.97
    status: EXPECTED_PASSING
```

### Known compile failure

```yaml
  clip/pytorch-openai/clip-vit-base-patch32-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: FAILED_TTMLIR_COMPILATION
    reason: "Unsupported op in tt-mlir - https://github.com/tenstorrent/tt-xla/issues/1234"
```

### PCC regression (still passing, tracked)

```yaml
  wide_resnet/pytorch-wide_resnet101_2-single_device-inference:
    status: EXPECTED_PASSING
    required_pcc: 0.96
    bringup_status: INCORRECT_RESULT
    reason: "PCC regression after consteval changes - https://github.com/tenstorrent/tt-xla/issues/1242"
```

### Severe PCC issue (PCC check disabled)

```yaml
  gpt_neo/causal_lm/pytorch-gpt_neo_2_7B-single_device-inference:
    status: EXPECTED_PASSING
    assert_pcc: false
    bringup_status: INCORRECT_RESULT
    reason: "PCC=-1.0 - https://github.com/tenstorrent/tt-xla/issues/5678"
```

### Architecture-specific overrides

```yaml
  qwen_3/embedding/pytorch-embedding_8b-single_device-inference:
    status: EXPECTED_PASSING
    arch_overrides:
      n150:
        status: NOT_SUPPORTED_SKIP
        reason: "Too large for single chip"
        bringup_status: FAILED_RUNTIME
      p150:
        required_pcc: 0.97
```

### Tensor parallel with arch restriction

```yaml
  llama/causal_lm/pytorch-3.1_8B-tensor_parallel-inference:
    supported_archs: [n300-llmbox]
    status: EXPECTED_PASSING
```

### Model with push marker (runs on every PR)

```yaml
  resnet/pytorch-ResNet50_HuggingFace-single_device-inference:
    status: EXPECTED_PASSING
    markers: [push]
```

### Placeholder model (not yet implemented)

In `test_config_placeholders.yaml`:
```yaml
PLACEHOLDER_MODELS:
  meta-llama/Llama-3.2-90B-Vision-Instruct:
    bringup_status: NOT_STARTED
```
