---
name: tt-model-bringup
description: Build a ForgeModel-compliant loader for tt-forge-models and integrate it into the tt-xla test runner. Covers the variant system, config enums, loader scaffolding, TT-compatible model extraction, test registration, and device validation lifecycle.
argument-hint: <model-name-or-huggingface-id>
---

# Model Bringup — ForgeModel Loader & Test Integration

This skill automates the creation of a standardized model loader in `tt-forge-models` and
wires it into the tt-xla test runner for single-device and multi-chip validation.

It is organized in two phases:
- **Phase A** — Build the ForgeModel loader (CPU-only, no hardware needed)
- **Phase B** — Register in the test suite and drive through the validation lifecycle

## External Resources

- [tt-forge-models README](https://github.com/tenstorrent/tt-forge-models/blob/main/README.md)
- [HuggingFace Model Hub](https://huggingface.co/models)

## Prerequisites

1. `third_party/tt_forge_models/` is checked out (git submodule of tt-xla)
2. Python env has `torch`, `transformers` (for HF models)
3. Know the model's HuggingFace ID or have the custom source code + weights

---

# Phase A — Build the ForgeModel Loader

## A1. Gather Model Facts

Before writing any code, collect this information:

**Architecture facts:**
- Input signature: what tensors, what shapes, what dtypes?
- Output signature: logits? bounding boxes? embeddings? multiple heads?
- Does the model need a tokenizer, image preprocessor, or custom data pipeline?
- Pretrained checkpoint: HuggingFace ID, local `.pth`, or TorchVision built-in?

**Classification (used by `_get_model_info`):**

| Field | How to decide |
|-------|---------------|
| `ModelSource` | Where does the model come from? `HUGGING_FACE`, `TORCHVISION`, `TIMM`, `CUSTOM`, `TORCH_HUB`, `GITHUB`, `OSMR` |
| `ModelTask` | What does the model do? `NLP_CAUSAL_LM`, `NLP_QA`, `CV_IMAGE_CLS`, `CV_OBJECT_DET`, `MM_CAUSAL_LM`, etc. (see `config.py` for all 40+ enums) |
| `ModelGroup` | Why are we adding it? `RED` = customer ask, `PRIORITY` = strategic coverage, `GENERALITY` = breadth, `VULCAN` = auto-generated |
| `Framework` | Implementation framework: `TORCH`, `JAX`, `ONNX`, `PADDLE` |

**Variant planning:**
- How many sizes/versions? (e.g., `base`/`large`, `7B`/`70B`)
- Which variant should be the default (usually the smallest one that's still representative)?

## A2. Choose Directory Layout

Two patterns exist in `third_party/tt_forge_models/`:

**Pattern A** — one task per model (most models):
```
<model_name>/
├── __init__.py          # re-exports from pytorch/
└── pytorch/
    ├── __init__.py      # exports ModelLoader, ModelVariant
    ├── loader.py        # the ForgeModel subclass
    └── src/             # optional custom nn.Module code
```

**Pattern B** — multiple tasks per model (e.g., BERT does QA + masked LM + classification):
```
<model_name>/
├── __init__.py
└── <task_name>/         # causal_lm, question_answering, image_cls, ...
    ├── __init__.py
    └── pytorch/
        ├── __init__.py
        ├── loader.py
        └── src/
```

Import depth changes between patterns — Pattern A uses `from ...base` (3 dots), Pattern B uses `from ....base` (4 dots).

## A3. Scaffold the ModelLoader

Every loader must satisfy the `ForgeModel` contract from `base.py`. The required pieces are:

1. **`ModelVariant(StrEnum)`** — enumerate every size/version as an enum member
2. **`_VARIANTS` dict** — map each variant to a `ModelConfig` (or `LLMModelConfig` for text models)
3. **`DEFAULT_VARIANT`** — class-level attribute pointing to one variant enum member
4. **`_get_model_info(cls, variant)`** — classmethod returning a `ModelInfo` frozen dataclass
5. **`load_model(self, *, dtype_override=None, **kwargs)`** — instantiate and return the `nn.Module` in eval mode
6. **`load_inputs(self, dtype_override=None)`** — return sample inputs matching the model's forward signature

See `references/loader_templates.md` for ready-to-use templates covering:
- HuggingFace Causal LM (Template 1)
- HuggingFace task-specific models like QA/classification (Template 2)
- Custom PyTorch vision models with local weights (Template 3)
- TorchVision / TIMM models (Template 4)

### Key conventions to follow

**`load_model` must use keyword-only `dtype_override`:**
```python
def load_model(self, *, dtype_override=None, **kwargs):
    ...
    model.eval()
    return model
```

**Always call `model.eval()` before returning.**

**`dtype_override` semantics:**
- `None` → model uses its native dtype (typically float32)
- `torch.bfloat16` → for HF models, pass `torch_dtype=torch.bfloat16` to `from_pretrained`; for custom models, call `model.to(torch.bfloat16)`

**Cache expensive resources lazily** (tokenizers, configs):
```python
def __init__(self, variant=None):
    super().__init__(variant)
    self.tokenizer = None   # loaded on first use

def _load_tokenizer(self, dtype_override=None):
    if self.tokenizer is not None:
        return self.tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(
        self._variant_config.pretrained_model_name
    )
    return self.tokenizer
```

### Optional methods worth implementing

| Method | When to implement |
|--------|-------------------|
| `decode_output()` | Text models that produce token IDs; detection models with NMS |
| `unpack_forward_output()` | Training support — extract a differentiable tensor from complex output objects |
| `load_config()` | When `get_mesh_config` or other code needs `model.config` before `load_model` |
| `get_mesh_config()` | Multi-chip tensor parallelism (see `tt-multi-chip` skill) |
| `load_shard_spec()` | Weight sharding for tensor parallelism (see `tt-multi-chip` skill) |

## A4. Wire Up `__init__.py` Exports

```python
# <model_name>/pytorch/__init__.py
from .loader import ModelLoader, ModelVariant

# <model_name>/__init__.py
from .pytorch import ModelLoader
```

Pattern B adds one more level:
```python
# <model_name>/<task>/__init__.py
from .pytorch import ModelLoader
```

## A5. Sanity-Check on CPU

Verify the loader works before involving any hardware:

```python
import torch
from third_party.tt_forge_models.<model_name>.pytorch import ModelLoader

loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Forward pass with default dtype
with torch.no_grad():
    out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
print(f"Output type: {type(out)}, Model info: {loader.get_model_info()}")

# Forward pass with bfloat16 (mirrors TT hardware dtype)
model_bf = loader.load_model(dtype_override=torch.bfloat16)
inputs_bf = loader.load_inputs(dtype_override=torch.bfloat16)
with torch.no_grad():
    out_bf = model_bf(**inputs_bf) if isinstance(inputs_bf, dict) else model_bf(inputs_bf)
print("bfloat16 forward pass succeeded")
```

If this fails, fix the loader before proceeding — the test runner will hit the same errors.

---

# Phase B — Test Registration & Device Validation

## B1. Ensure TT Compatibility

Before registering the model in the test suite, audit the `nn.Module` for ops that won't
trace through XLA / StableHLO / tt-mlir. The model code in `src/model.py` (if custom) must:

- Use **fully static tensor shapes** — no data-dependent dimensions
- Avoid **in-place scatter** (`tensor[:, idx] = val`) — use functional alternatives
- Avoid **data-dependent control flow** (`if tensor.item() > x`) — restructure as `torch.where`
- Replace unsupported pooling/conv variants if the compiler doesn't handle them
- Keep preprocessing (tokenization, image normalization, voxelization) and postprocessing
  (NMS, decoding) **outside** the traced module — those run on CPU

## B2. Register in Test Config

The test runner discovers models from YAML config files. Add an entry to the appropriate file:

| Model type | Config file |
|-----------|-------------|
| Standard single-device | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| Tensor-parallel | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |
| Data-parallel | `tests/runner/test_config/torch/test_config_inference_data_parallel.yaml` |
| LLM single-device | `tests/runner/test_config/torch_llm/test_config_inference_single_device.yaml` |
| LLM tensor-parallel | `tests/runner/test_config/torch_llm/test_config_inference_tensor_parallel.yaml` |

Entry format:
```yaml
test_config:
  <model_name>/pytorch-<VariantValue>-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "Initial bringup — awaiting first PCC pass"
```

The test ID format is `<loader_path>-<variant_value>-<parallelism>-<run_mode>`.

### YAML fields reference

| Field | Purpose |
|-------|---------|
| `status` | `EXPECTED_PASSING`, `KNOWN_FAILURE_XFAIL`, or `SKIP` |
| `bringup_status` | Optional: `IN_PROGRESS`, `FAILED_TTMLIR_COMPILATION`, etc. |
| `reason` | Human-readable explanation (required when status is not `EXPECTED_PASSING`) |
| `required_pcc` | Override the default 0.99 PCC threshold (e.g., `0.97`) |
| `assert_pcc` | Set to `false` to skip PCC check entirely |
| `arch_overrides` | Per-architecture overrides (`n150`, `p150`, etc.) |
| `supported_archs` | Restrict to specific architectures |

## B3. Run the Test

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<model_name>/pytorch-<VariantValue>-single_device-inference]"
```

The test runner:
1. Instantiates `ModelLoader(variant=variant)`
2. Calls `load_model(dtype_override=torch.bfloat16)` and `load_inputs(dtype_override=torch.bfloat16)`
3. Runs the model on CPU to get reference output
4. Compiles and runs through XLA on the TT device
5. Computes PCC (Pearson Correlation Coefficient) between CPU and TT outputs
6. Passes if PCC >= `required_pcc` (default 0.99)

## B4. Troubleshooting Device Failures

When the TT run fails, the error message usually points to one of these categories:

**Compilation failures** — the model graph can't be lowered to tt-mlir:
- Check for unsupported ops, dynamic shapes, or very large tensors
- Inspect XLA IR dumps with `--dump-irs` to isolate which op fails
- Try reducing `num_layers` to isolate whether it's a specific layer

**Memory failures** — the model doesn't fit in device SRAM or DRAM:
- L1 overflow: reduce channel counts, batch size, or spatial dims
- DRAM overflow: the model may need multi-chip sharding (see `tt-multi-chip` skill)

**Numerical mismatches** — PCC below threshold:
- Ensure ground truth was computed in bfloat16 (not float32)
- Check whether any op replacements (e.g., MaxPool → Conv) changed semantics
- Some architectures have inherently lower PCC on certain hardware — use `required_pcc` to relax
- Per-arch differences can be handled with `arch_overrides` in the YAML

## B5. Promote to Passing

Once PCC meets the threshold consistently:

```yaml
  <model_name>/pytorch-<VariantValue>-single_device-inference:
    status: EXPECTED_PASSING
```

Drop the `bringup_status` and `reason` fields. If PCC is slightly below 0.99 but stable,
document the relaxed threshold:

```yaml
  <model_name>/pytorch-<VariantValue>-single_device-inference:
    required_pcc: 0.97   # <link to tracking issue>
    status: EXPECTED_PASSING
```

---

## ForgeModel API Summary

```python
class ForgeModel(ABC):
    _VARIANTS: Dict[StrEnum, ModelConfig] = {}
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):            # validates + caches _variant_config
    def load_model(self, **kwargs): ...          # abstract — return nn.Module in eval()
    def load_inputs(self, **kwargs): ...         # abstract — return sample inputs
    def _get_model_info(cls, variant) -> ModelInfo: ...  # abstract classmethod

    # Optional
    def decode_output(cls, **kwargs): ...
    def get_mesh_config(self, num_devices): ...  # → (mesh_shape, mesh_names)
    def load_shard_spec(self, model): ...        # → {tensor: shard_tuple} or None
    def load_config(self, **kwargs): ...
    def unpack_forward_output(self, fwd_output): ...
```

Config dataclasses:
```python
ModelConfig(pretrained_model_name: str)
LLMModelConfig(pretrained_model_name, max_length, attention_mechanism, sliding_window)
ModelInfo(model, variant, group, task, source, framework)  # frozen
```

## File Locations

| Item | Path |
|------|------|
| ForgeModel base class | `third_party/tt_forge_models/base.py` |
| Config enums & dataclasses | `third_party/tt_forge_models/config.py` |
| Model loader (Pattern A) | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| Model source (custom) | `third_party/tt_forge_models/<model>/pytorch/src/model.py` |
| Single-device test config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| Tensor-parallel test config | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |
| LLM test configs | `tests/runner/test_config/torch_llm/test_config_inference_*.yaml` |
| Test entry point | `tests/runner/test_models.py` |

## Pre-Submit Checklist

- [ ] `ModelLoader` extends `ForgeModel` with correct `_VARIANTS`, `DEFAULT_VARIANT`, `ModelVariant`
- [ ] `_get_model_info` returns `ModelInfo` with appropriate `ModelGroup`, `ModelTask`, `ModelSource`, `Framework`
- [ ] `load_model` uses `*, dtype_override=None, **kwargs` signature and calls `model.eval()`
- [ ] `load_inputs` returns tensors matching the model's actual forward signature
- [ ] `__init__.py` files re-export `ModelLoader` and `ModelVariant`
- [ ] SPDX copyright header on all new `.py` files
- [ ] CPU forward pass succeeds in both float32 and bfloat16
- [ ] YAML test config entry added with initial `KNOWN_FAILURE_XFAIL` status
- [ ] TT device PCC meets threshold (0.99 default, or documented `required_pcc`)
- [ ] Status promoted to `EXPECTED_PASSING` after stable validation
