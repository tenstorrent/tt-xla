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

1. `third_party/tt_forge_models/` is checked out (**git submodule** of tt-xla; loader changes commit in that repo, then bump the submodule pointer in tt-xla)
2. For PyTorch models: Python env has `torch`, `transformers`
3. For JAX models: Python env has `jax`, `flax`, `easydel`, `transformers`
4. Know the model's HuggingFace ID or have the custom source code + weights
5. If the original model has extra dependencies (e.g., `mmdet`, `mmcv`, `det3d`), install them in a **separate venv** — do not pollute the tt-xla venv. Use the separate venv only for extracting/understanding the model; the rewritten `model.py` must depend only on standard PyTorch ops.
6. For models whose weights are fetched via `get_file()` (custom/non-HuggingFace weights), set the IRD cache to avoid download timeouts:
   ```bash
   export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
   ```

## Code Generation Rules

When generating any file (loader, sanity script, CPU check, test file):

- **Do NOT add comments at the top of the file** describing how to run it, summarizing
  what the file does, or listing shell commands. The only acceptable header is the
  SPDX copyright block.
- Keep the code clean — comments should only explain non-obvious logic, not narrate
  what the code does or how to invoke it.

**See also:** `references/test_ids_and_yaml.md` — canonical **test IDs**, **`rel_path`**, **`test_llms_torch`**, submodule workflow, **`requirements.txt`**, TorchDynamicLoader **bfloat16** behavior, and JAX **EasyDel + multichip tester** path.

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
| `ModelSource` | Where does the model come from? `HUGGING_FACE`, `EASYDEL`, `TORCHVISION`, `TIMM`, `CUSTOM`, `TORCH_HUB`, `GITHUB`, `OSMR` |
| `ModelTask` | What does the model do? `NLP_CAUSAL_LM`, `NLP_QA`, `CV_IMAGE_CLS`, `CV_OBJECT_DET`, `MM_CAUSAL_LM`, etc. (see `config.py` for all 40+ enums) |
| `ModelGroup` | Why are we adding it? `RED` = customer ask, `PRIORITY` = strategic coverage, `GENERALITY` = breadth, `VULCAN` = auto-generated |
| `Framework` | Implementation framework: `TORCH`, `JAX`, `ONNX`, `PADDLE`. Use `JAX` for EasyDeL-based models |

**Variant planning:**
- How many sizes/versions? (e.g., `base`/`large`, `7B`/`70B`)
- Which variant should be the default (usually the smallest one that's still representative)?

## A2. Choose Directory Layout

The layout depends on the framework (`pytorch/` vs `jax/`) and whether the model supports
a single task or multiple tasks.

**Pattern A** — one task, single framework (simplest):
```
<model_name>/
├── __init__.py          # re-exports from pytorch/ or jax/
└── pytorch/             # or jax/
    ├── __init__.py      # exports ModelLoader, ModelVariant
    ├── loader.py        # the ForgeModel subclass
    └── src/             # optional custom module code
```

**Pattern B** — multiple tasks per model (e.g., BERT does QA + masked LM + classification):
```
<model_name>/
├── __init__.py
└── <task_name>/         # causal_lm, question_answering, image_cls, ...
    ├── __init__.py
    └── pytorch/         # or jax/
        ├── __init__.py
        ├── loader.py
        └── src/
```

**Pattern C** — same model with both PyTorch and JAX loaders:
```
<model_name>/
├── __init__.py
└── <task_name>/
    ├── __init__.py
    ├── pytorch/
    │   ├── __init__.py
    │   └── loader.py
    └── jax/
        ├── __init__.py
        └── loader.py
```

Import depth changes between patterns — Pattern A uses `from ...base` (3 dots), Pattern B/C uses `from ....base` (4 dots).

For JAX models, the `jax/` subdirectory replaces `pytorch/`. Many models (Llama, Qwen, Phi,
Falcon, Mistral, Mamba, GPT-2, GPT-J, Whisper) have both PyTorch and JAX loaders side by side.

**Test discovery:** the runner only discovers `pytorch/loader.py` and `jax/loader.py` (directory name must be exactly `pytorch` or `jax`). Experimental trees such as `bounty_jax/` are **not** auto-discovered — use `jax/` for loaders that must appear in `test_all_models_*`.

**Per-loader dependencies:** optional `requirements.txt` in the **same directory** as `loader.py` is installed before tests (`tests/runner/requirements.py`). Use it when the model needs extra pip packages.

## A3. Scaffold the ModelLoader

Every loader must satisfy the `ForgeModel` contract from `base.py`. The required pieces are:

1. **`ModelVariant(StrEnum)`** — enumerate every size/version as an enum member
2. **`_VARIANTS` dict** — map each variant to a `ModelConfig` (or `LLMModelConfig` for text models)
3. **`DEFAULT_VARIANT`** — class-level attribute pointing to one variant enum member
4. **`_get_model_info(cls, variant)`** — classmethod returning a `ModelInfo` frozen dataclass
5. **`load_model`** — instantiate and return the model (project convention: `def load_model(self, *, dtype_override=None, **kwargs)`; `ForgeModel` in `base.py` declares `load_model(self, **kwargs)` — match team style and support `dtype_override` when the test harness should drive dtype)
6. **`load_inputs`** — return sample inputs matching the model's forward signature (same `dtype_override` / `mesh` conventions as sibling loaders)

See `references/loader_templates.md` for ready-to-use templates covering:
- HuggingFace Causal LM — PyTorch (Template 1)
- HuggingFace task-specific models like QA/classification — PyTorch (Template 2)
- Custom PyTorch vision models with local weights (Template 3)
- TorchVision / TIMM models (Template 4)
- EasyDeL JAX Causal LM (Template 5)

### Key conventions — PyTorch loaders

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

### Key conventions — JAX / EasyDeL loaders

JAX loaders follow the same `ForgeModel` contract but differ in framework-specific details:

**Model loading uses EasyDeL instead of HuggingFace Transformers:**
```python
from easydel import AutoEasyDeLModelForCausalLM

def load_model(self, *, dtype_override=None, **kwargs):
    model_kwargs = {}
    if dtype_override is not None:
        model_kwargs["dtype"] = dtype_override
    model_kwargs |= kwargs
    partition_rules = ((r".*", PartitionSpec()),)
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        self._model_name, partition_rules=partition_rules, **model_kwargs
    )
    return model
```

**`_get_model_info` must specify `ModelSource.EASYDEL` and `Framework.JAX`:**
```python
return ModelInfo(
    model="MyModel",
    variant=variant,
    group=ModelGroup.GENERALITY,
    task=ModelTask.NLP_CAUSAL_LM,
    source=ModelSource.EASYDEL,
    framework=Framework.JAX,
)
```

**Input tensors use `jax.numpy` instead of `torch`:**
```python
import jax.numpy as jnp

def load_inputs(self, dtype_override=None, mesh=None):
    inputs = self._tokenizer(self.sample_text, return_tensors="jax")
    input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
    return {"input_ids": input_ids}
```

**Batch size must be divisible by device count when `mesh` is provided:**
```python
if mesh is not None:
    num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
    batch_size = 8
    if batch_size % num_devices != 0:
        batch_size = num_devices * (batch_size // num_devices + 1)
```

**Config objects come from `easydel.modules.<arch>`:**
```python
from easydel.modules.llama import LlamaConfig       # not transformers.LlamaConfig
from easydel.modules.qwen3 import Qwen3Config
from easydel.modules.falcon import FalconConfig
```

**For dtype casting, use the JAX-specific utility:**
```python
from ....tools.jax_utils import cast_hf_model_to_type
if dtype_override is not None:
    model = cast_hf_model_to_type(model, dtype_override)
```

### Common conventions for both frameworks

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

### Optional methods

| Method | Framework | When to implement |
|--------|-----------|-------------------|
| `decode_output()` | Both | Text models producing token IDs; detection models with NMS |
| `unpack_forward_output()` | Both | Training support — extract a differentiable tensor from complex output objects |
| `load_config()` | Both | When multi-chip or other code needs `model.config` before `load_model` |
| `get_mesh_config()` | PyTorch | Multi-chip tensor parallelism for PyTorch (see `tt-multi-chip` skill) |
| `load_shard_spec()` | PyTorch | Weight sharding for PyTorch tensor parallelism (see `tt-multi-chip` skill) |
| `get_input_activations_partition_spec()` | JAX | Input partitioning for JAX multi-chip (see `tt-multi-chip` skill) |
| `load_parameters_partition_spec()` | JAX | Parameter sharding for JAX multi-chip (see `tt-multi-chip` skill) |

## A4. Wire Up `__init__.py` Exports

For PyTorch loaders:
```python
# <model_name>/pytorch/__init__.py
from .loader import ModelLoader, ModelVariant

# <model_name>/__init__.py
from .pytorch import ModelLoader
```

For JAX loaders:
```python
# <model_name>/jax/__init__.py
from .loader import ModelLoader, ModelVariant

# <model_name>/__init__.py  (or <model_name>/<task>/__init__.py)
from .jax import ModelLoader
```

Pattern B/C adds one more level:
```python
# <model_name>/<task>/__init__.py
from .pytorch import ModelLoader   # or .jax
```

## A5. Sanity-Check on CPU

All CPU sanity scripts and intermediate test files must be saved under the model's
test directory within the repo:

```
tests/torch/models/<model_name>/
```

**Do NOT use `/tmp` or any location outside the repo.** This keeps artefacts version-controlled
and co-located with the model. After the full bringup is complete (model is `EXPECTED_PASSING`),
clean up any intermediate sanity scripts that are no longer needed — only keep files that
serve as permanent regression tests.

### PyTorch models

Save the following as `tests/torch/models/<model_name>/test_<model_name>_cpu_sanity.py`:

```python
import torch
from third_party.tt_forge_models.<path.to.pytorch.package> import ModelLoader

loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

with torch.no_grad():
    out = model(**inputs) if isinstance(inputs, dict) else model(inputs)
print(f"Output type: {type(out)}, Model info: {loader.get_model_info()}")

model_bf = loader.load_model(dtype_override=torch.bfloat16)
inputs_bf = loader.load_inputs(dtype_override=torch.bfloat16)
with torch.no_grad():
    out_bf = model_bf(**inputs_bf) if isinstance(inputs_bf, dict) else model_bf(inputs_bf)
print("bfloat16 forward pass succeeded")
```

### JAX / EasyDeL models

Save the following as `tests/torch/models/<model_name>/test_<model_name>_cpu_sanity.py`:

```python
import jax
from third_party.tt_forge_models.<path.to.jax.package> import ModelLoader

loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

out = model(**inputs)
print(f"Output type: {type(out)}, Model info: {loader.get_model_info()}")

import jax.numpy as jnp
model_bf = loader.load_model(dtype_override=jnp.bfloat16)
inputs_bf = loader.load_inputs(dtype_override=jnp.bfloat16)
out_bf = model_bf(**inputs_bf)
print("bfloat16 forward pass succeeded")
```

If this fails, fix the loader before proceeding — the test runner will hit the same errors.

### Cleanup after bringup

Once the model is fully brought up and `EXPECTED_PASSING`, remove intermediate sanity
scripts from `tests/torch/models/<model_name>/` that are no longer needed. Keep only
files that serve as permanent tests (e.g., op-specific regression tests).

---

# Phase B — Test Registration & Device Validation

## B1. Ensure TT Compatibility

Before registering the model in the test suite, audit the `nn.Module` for ops that won't
trace through XLA / StableHLO / tt-mlir. Split the model at the boundary of what TT can run:
- **CPU side:** preprocessing (tokenization, image normalization, voxelization), dynamic ops, NMS / postprocessing
- **TT side:** the neural network backbone / neck / head with static shapes

The model code in `src/model.py` (if custom) must:

- Use **fully static tensor shapes** — no data-dependent dimensions
- Avoid **in-place scatter** (`tensor[:, idx] = val`) — use functional alternatives
- Avoid **data-dependent control flow** (`if tensor.item() > x`) — restructure as `torch.where`
- Replace unsupported pooling/conv variants (see known limitations table below)
- Keep preprocessing and postprocessing **outside** the traced module — those run on CPU

### Known tt-mlir Limitations

These ops are known to fail or produce incorrect results at the compiler/runtime level.
Proactively replace them during the model rewrite step:

| Op / Pattern | Issue | Workaround |
|---|---|---|
| `MaxPool3d` | No bfloat16 support in tt-mlir | Replace with stride-2 `Conv3d` |
| `ConvTranspose3d` (lhs_dilation) | Incorrect result in tt-mlir | `F.interpolate` + `Conv3d(1×1×1)` |
| `Conv2d` with large dilation (e.g., dilation=18, 512→512ch) | DRAM Auto slice fatal | Needs compiler fix; avoid large dilation or reduce spatial dims |
| `Conv3d` with large out_channels (e.g., 1280) | L1 static circular buffer overflow | Needs compiler fix |
| Dynamic token / spatial shapes | Shardy requires fully static shapes | Fix all shapes to be static constants |
| Models exceeding single-chip DRAM | Out of memory | Requires multi-chip sharding (see `tt-multi-chip` skill) |
| `F.grid_sample` with large index tensors | TTNN tiling pads trailing dims → massive memory blowup | May need decomposition or compiler fix |

## B2. Register in Test Config

The test runner discovers models from YAML config files. Add an entry to the appropriate
**inference** config file. Model bringup targets **inference only** — do not add training
configs unless explicitly requested.

**PyTorch models (inference):**

| Model type | Config file |
|-----------|-------------|
| Standard single-device | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| Tensor-parallel | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |
| Data-parallel | `tests/runner/test_config/torch/test_config_inference_data_parallel.yaml` |
| LLM single-device | `tests/runner/test_config/torch_llm/test_config_inference_single_device.yaml` |
| LLM tensor-parallel | `tests/runner/test_config/torch_llm/test_config_inference_tensor_parallel.yaml` |

**JAX models (inference):**

| Model type | Config file |
|-----------|-------------|
| Single-device | `tests/runner/test_config/jax/test_config_inference_single_device.yaml` |
| Tensor-parallel | `tests/runner/test_config/jax/test_config_inference_tensor_parallel.yaml` |
| Data-parallel | `tests/runner/test_config/jax/test_config_inference_data_parallel.yaml` |

### Test ID and YAML key (critical)

Keys must use the full **`rel_path`** from `third_party/tt_forge_models` to the **parent of `loader.py`**, not a shortened `<model_name>/pytorch` alias. The `run_mode` segment must be **`inference`**.

**Canonical pattern** for `test_all_models_torch` / `test_all_models_jax` (always `inference`):

```text
{rel_path}-{VariantValue}-{parallelism}-inference
```

### Initial status: always start with `EXPECTED_PASSING`

⚠ **Do NOT register new models as `KNOWN_FAILURE_XFAIL`.**

When a test is marked `KNOWN_FAILURE_XFAIL`, pytest treats failures as expected (xfail) and
**hides the actual error output**. This makes it impossible to see what went wrong during
bringup. Instead:

1. **Register with `EXPECTED_PASSING`** so the real error is fully visible in pytest output
2. **Run the test** — if it fails, you see the exact error (OOM, PCC drop, compilation failure, etc.)
3. **After diagnosing the failure**, update the status to `KNOWN_FAILURE_XFAIL` with a specific reason
4. When the issue is fixed and PCC meets threshold, the status stays `EXPECTED_PASSING`

Example — initial registration:

```yaml
test_config:
  llama/causal_lm/pytorch-3.1_8B-single_device-inference:
    status: EXPECTED_PASSING
```

```yaml
test_config:
  llama/causal_lm/jax-3B_v2-single_device-inference:
    status: EXPECTED_PASSING
```

Example — after observing a specific failure, update with a diagnosis:

```yaml
test_config:
  llama/causal_lm/pytorch-3.1_8B-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "OOM in attention layer — grid_sample index tensor blowup (tracking issue #1234)"
```

**PyTorch LLMs (decode/prefill):** if you implement `load_inputs_decode` / `load_inputs_prefill`, also add entries under `tests/runner/test_config/torch_llm/` using the **long** IDs (`llm_decode`, `llm_prefill`, `seq_*`, `batch_*`). See `references/test_ids_and_yaml.md` and `tests/runner/validate_test_config.py`.

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

Use the **exact** parametrized id (full `rel_path` and variant value). Examples:

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[llama/causal_lm/pytorch-3.1_8B-single_device-inference]"
```

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_jax[llama/causal_lm/jax-3B_v2-single_device-inference]"
```

**IRD cache for `get_file()` models:** If the model loader uses `get_file()` to download
weights (custom/non-HuggingFace weights), set the IRD cache before running:
```bash
export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
```

**PyTorch note:** `TorchDynamicLoader` passes **`dtype_override=torch.bfloat16`** to `load_model` / `load_inputs` when those methods accept `dtype_override`, so the default device comparison path often uses bfloat16 — align PCC expectations accordingly.

**JAX EasyDel note:** even **single-device** EasyDel models are run through **`DynamicJaxMultiChipModelTester`** with a 1-device mesh (`tests/runner/test_models.py`). Partition helpers must still be valid (typically replicated specs).

The test runner:
1. Instantiates `ModelLoader(variant=variant)`
2. Calls `load_model(dtype_override=...)` and `load_inputs(dtype_override=...)` (when supported)
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
- Max absolute error consistent with bfloat16 rounding is ~5e-4; larger deltas indicate a real problem

### Common Runtime Error Reference

| Error Message | Cause | Fix |
|---|---|---|
| `Bad StatusOr access: INTERNAL: Error code: 13` | Compilation/runtime error in tt-mlir | Check tt-mlir logs; simplify or replace the failing op |
| `Input type (float) and bias type (BFloat16)` | Input dtype mismatch on TT device | Cast all inputs to bfloat16 before sending to TT |
| `Statically allocated circular buffers grow to X B` | L1 SRAM overflow | Reduce channel counts, batch size, or spatial dimensions |
| `Not enough space to allocate X B DRAM buffer across N banks` | DRAM overflow | Model too large for single chip; use multi-chip or reduce model size |
| `DRAM Auto slice could not find valid slice configuration` | Large dilated conv can't be sliced | Avoid large dilation values or reduce spatial dims |
| `Shardy propagation only supports ranked tensors with a static shape` | Dynamic shape detected | Fix all input/intermediate shapes to be fully static |
| `TT_FATAL ... ToDeviceOp` | Tensor too large after TTNN tiling | Check for padding blowup on small trailing dimensions (see B1 limitations) |
| PCC < 0.99 but no crash | Numerical divergence | Check op replacements that changed semantics; verify bfloat16 ground truth |

## B5. Confirm Passing or Downgrade to XFAIL

Since the initial registration uses `EXPECTED_PASSING` (see B2), the workflow after running
the test is:

**If the test passes** — nothing to change; the model is already `EXPECTED_PASSING`. If PCC
is slightly below 0.99 but stable, document the relaxed threshold:

```yaml
  llama/causal_lm/pytorch-3.1_8B-single_device-inference:
    required_pcc: 0.97   # <link to tracking issue>
    status: EXPECTED_PASSING
```

**If the test fails** — you have seen the real error (not hidden by xfail). Now downgrade
to `KNOWN_FAILURE_XFAIL` with a **specific reason** describing the observed failure:

```yaml
  llama/causal_lm/pytorch-3.1_8B-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "OOM — attention gather index tensor 18.7 GB after TTNN tiling (issue #1234)"
```

When the underlying issue is fixed and the test passes again, restore to `EXPECTED_PASSING`
and drop the `bringup_status` and `reason` fields.

---

## ForgeModel API Summary

```python
class ForgeModel(ABC):
    _VARIANTS: Dict[StrEnum, ModelConfig] = {}
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):            # validates + caches _variant_config
    def load_model(self, **kwargs): ...          # base.py signature; loaders often use *, dtype_override=None
    def load_inputs(self, **kwargs): ...         # base.py signature
    def _get_model_info(cls, variant) -> ModelInfo: ...  # abstract classmethod

    # Optional — shared
    def decode_output(cls, **kwargs): ...
    def load_config(self, **kwargs): ...
    def unpack_forward_output(self, fwd_output): ...

    # Optional — PyTorch multi-chip (loaders may extend load_shard_spec — see tt-multi-chip skill)
    def get_mesh_config(self, num_devices): ...  # → (mesh_shape, mesh_names)
    def load_shard_spec(self, model): ...        # → {tensor: shard_tuple} or None

    # Optional — JAX (used for EasyDeL single- and multi-device via DynamicJaxMultiChipModelTester)
    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"): ...
    def load_parameters_partition_spec(self, model, parallelism, axis_name, ...): ...
```

Config dataclasses:
```python
ModelConfig(pretrained_model_name: str)
LLMModelConfig(pretrained_model_name, max_length, attention_mechanism, sliding_window)
ModelInfo(model, variant, group, task, source, framework)  # frozen
```

EasyDeL entry points (JAX):
```python
from easydel import AutoEasyDeLModelForCausalLM          # causal LMs
from easydel import AutoEasyDeLModelForSpeechSeq2Seq     # audio/speech models
from easydel.modules.<arch> import <Arch>Config           # per-model config + partition rules
from infra.utilities import make_easydel_parameters_partition_specs  # partition spec builder
```

## File Locations

| Item | Path |
|------|------|
| ForgeModel base class | `third_party/tt_forge_models/base.py` |
| Config enums & dataclasses | `third_party/tt_forge_models/config.py` |
| PyTorch loader (Pattern A) | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| JAX loader (Pattern A) | `third_party/tt_forge_models/<model>/jax/loader.py` |
| PyTorch loader (Pattern B) | `third_party/tt_forge_models/<model>/<task>/pytorch/loader.py` |
| JAX loader (Pattern B) | `third_party/tt_forge_models/<model>/<task>/jax/loader.py` |
| JAX utilities | `third_party/tt_forge_models/tools/jax_utils.py` |
| PyTorch single-device config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| PyTorch tensor-parallel config | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |
| PyTorch LLM configs | `tests/runner/test_config/torch_llm/test_config_inference_*.yaml` |
| JAX single-device config | `tests/runner/test_config/jax/test_config_inference_single_device.yaml` |
| JAX tensor-parallel config | `tests/runner/test_config/jax/test_config_inference_tensor_parallel.yaml` |
| JAX data-parallel config | `tests/runner/test_config/jax/test_config_inference_data_parallel.yaml` |
| JAX training config | `tests/runner/test_config/jax/test_config_training_single_device.yaml` |
| Test entry point | `tests/runner/test_models.py` |
| Test ID / YAML reference | `references/test_ids_and_yaml.md` |

## Pre-Submit Checklist

**All models:**
- [ ] `ModelLoader` extends `ForgeModel` with correct `_VARIANTS`, `DEFAULT_VARIANT`, `ModelVariant`
- [ ] `_get_model_info` returns `ModelInfo` with appropriate `ModelGroup`, `ModelTask`, `ModelSource`, `Framework`
- [ ] `load_model` / `load_inputs` follow project conventions (`dtype_override` where expected for the harness)
- [ ] `load_inputs` returns tensors matching the model's actual forward signature
- [ ] `__init__.py` files re-export `ModelLoader` and `ModelVariant`
- [ ] SPDX copyright header on all new `.py` files
- [ ] CPU forward pass succeeds in both default dtype and bfloat16
- [ ] YAML keys use full **`rel_path`** (see `references/test_ids_and_yaml.md`), initial status is `EXPECTED_PASSING` (never `KNOWN_FAILURE_XFAIL` — xfail hides errors)
- [ ] `requirements.txt` next to `loader.py` if non-standard pip deps are required
- [ ] Submodule: loader commits in `third_party/tt_forge_models`, tt-xla commit updates the submodule pointer

**PyTorch-specific:**
- [ ] `load_model` calls `model.eval()` before returning
- [ ] `ModelSource` is `HUGGING_FACE`, `TORCHVISION`, `TIMM`, or `CUSTOM`; `Framework` is `TORCH`
- [ ] Multi-chip: `get_mesh_config()` and `load_shard_spec()` implemented if needed
- [ ] LLM decode/prefill: `load_inputs_decode` / `load_inputs_prefill` and **`torch_llm/` YAML** entries if those tests are in scope

**JAX-specific:**
- [ ] Model loaded via `AutoEasyDeLModelForCausalLM.from_pretrained` (or appropriate AutoModel)
- [ ] `ModelSource` is `EASYDEL`; `Framework` is `JAX`
- [ ] `load_inputs` returns JAX arrays (`jnp`), uses `return_tensors="jax"` for tokenizer
- [ ] `load_inputs` handles `mesh` parameter for batch size divisibility
- [ ] Multi-chip: `get_input_activations_partition_spec()` and `load_parameters_partition_spec()` implemented if needed (including coherent behavior for EasyDel **single-device** / 1-mesh path)

**Final validation:**
- [ ] TT device PCC meets threshold (0.99 default, or documented `required_pcc`)
- [ ] Status promoted to `EXPECTED_PASSING` after stable validation
