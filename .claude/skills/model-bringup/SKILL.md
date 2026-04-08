---
name: model-bringup
description: Step-by-step guide for bringing up a new PyTorch model in the tt-xla tt-forge model test suite. Covers the variant system, config enums, loader scaffolding, TT-compatible model extraction, test registration, and device validation lifecycle. Use when the user wants to add, debug, or move a model to EXPECTED_PASSING status.
argument-hint: <model-name-or-huggingface-id>
---

# Model Bringup for tt-xla (Single Device)

The golden rule: **establish CPU ground truth first, then bring up on TT and verify against it.**

Never bring up a model on TT without first knowing what the correct output looks like on CPU.

This skill automates the creation of a standardized model loader in `tt-forge-models` and
wires it into the tt-xla test runner for single-device validation.

It is organized in two phases:
- **Phase A** — Build the ForgeModel loader (CPU-only, no hardware needed)
- **Phase B** — Register in the test suite and drive through the validation lifecycle

## External Resources

- [tt-forge-models README](https://github.com/tenstorrent/tt-forge-models/blob/main/README.md)
- [HuggingFace Model Hub](https://huggingface.co/models)

## Prerequisites

1. `third_party/tt_forge_models/` is checked out (**git submodule** of tt-xla; loader changes commit in that repo, then bump the submodule pointer in tt-xla)
2. Python env has `torch`, `transformers`
3. Know the model's HuggingFace ID or have the custom source code + weights
4. If the original model has extra dependencies (e.g., `mmdet`, `mmcv`, `det3d`), install them in a **separate venv** — do not pollute the tt-xla venv. Use the separate venv only for extracting/understanding the model; the rewritten `model.py` must depend only on standard PyTorch ops.
5. For models whose weights are fetched via `get_file()` (custom/non-HuggingFace weights), set the IRD cache to avoid download timeouts:
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

**See also:** `references/test_ids_and_yaml.md` — canonical **test IDs**, **`rel_path`**, **`test_llms_torch`**, submodule workflow, **`requirements.txt`**, TorchDynamicLoader **bfloat16** behavior.

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
| `Framework` | Implementation framework: `TORCH` |

**Variant planning:**
- How many sizes/versions? (e.g., `base`/`large`, `7B`/`70B`)
- Which variant should be the default (usually the smallest one that's still representative)?

## A2. Choose Directory Layout

**Pattern A** — one task, single framework (simplest):
```
<model_name>/
├── __init__.py          # re-exports from pytorch/
└── pytorch/
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
    └── pytorch/
        ├── __init__.py
        ├── loader.py
        └── src/
```

Import depth changes between patterns — Pattern A uses `from ...base` (3 dots), Pattern B uses `from ....base` (4 dots).

**Test discovery:** the runner only discovers `pytorch/loader.py` (directory name must be exactly `pytorch`). Experimental trees are **not** auto-discovered.

**Per-loader dependencies:** optional `requirements.txt` in the **same directory** as `loader.py` is installed before tests (`tests/runner/requirements.py`). Use it when the model needs extra pip packages.

## A3. Scaffold the ModelLoader

Every loader must satisfy the `ForgeModel` contract from `base.py`. The required pieces are:

1. **`ModelVariant(StrEnum)`** — enumerate every size/version as an enum member
2. **`_VARIANTS` dict** — map each variant to a `ModelConfig` (or `LLMModelConfig` for text models)
3. **`DEFAULT_VARIANT`** — class-level attribute pointing to one variant enum member
4. **`_get_model_info(cls, variant)`** — classmethod returning a `ModelInfo` frozen dataclass
5. **`load_model`** — instantiate and return the model (project convention: `def load_model(self, *, dtype_override=None, **kwargs)`; `ForgeModel` in `base.py` declares `load_model(self, **kwargs)` — match team style and support `dtype_override` when the test harness should drive dtype)
6. **`load_inputs`** — return sample inputs matching the model's forward signature (same `dtype_override` conventions as sibling loaders)

See `references/loader_templates.md` for ready-to-use templates covering:
- HuggingFace Causal LM (Template 1)
- HuggingFace task-specific models like QA/classification (Template 2)
- Custom PyTorch Vision Model with local weights (Template 3)
- TorchVision / TIMM models (Template 4)

### Key conventions

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

### Optional methods

| Method | When to implement |
|--------|-------------------|
| `decode_output()` | Text models producing token IDs; detection models with NMS |
| `unpack_forward_output()` | Training support — extract a differentiable tensor from complex output objects |
| `load_config()` | When multi-chip or other code needs `model.config` before `load_model` |

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

All CPU sanity scripts and intermediate test files must be saved under the model's
test directory within the repo:

```
tests/torch/models/<model_name>/
```

**Do NOT use `/tmp` or any location outside the repo.** This keeps artefacts version-controlled
and co-located with the model. After the full bringup is complete (model is `EXPECTED_PASSING`),
clean up any intermediate sanity scripts that are no longer needed.

Cast model and inputs to **bfloat16** before saving — TT hardware always runs in bfloat16, so the ground truth must match that dtype to avoid inflated errors.

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

Also save ground truth outputs for later comparison:
```python
torch.save(inputs_bf, "tests/torch/models/<model_name>/<model_name>_inputs_bf16.pt")
torch.save(out_bf, "tests/torch/models/<model_name>/<model_name>_cpu_output_bf16.pt")
```

If this fails, fix the loader before proceeding — the test runner will hit the same errors.

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

### Known tt-mlir Limitations (as of 2026-03)

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

The test runner discovers models from YAML config files. Add an entry to the
**inference** config file. Model bringup targets **inference only**.

**Config file:** `tests/runner/test_config/torch/test_config_inference_single_device.yaml`

### Test ID and YAML key (critical)

Keys must use the full **`rel_path`** from `third_party/tt_forge_models` to the **parent of `loader.py`**, not a shortened `<model_name>/pytorch` alias. The `run_mode` segment must be **`inference`**.

**Canonical pattern:**

```text
{rel_path}-{VariantValue}-single_device-inference
```

**Examples:**
```text
llama/causal_lm/pytorch-3.1_8B-single_device-inference
resnet50/image_classification/pytorch-Default-single_device-inference
```

### Initial status: always start with `EXPECTED_PASSING`

**Do NOT register new models as `KNOWN_FAILURE_XFAIL`.**

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

Example — after observing a specific failure, downgrade with a diagnosis:

```yaml
test_config:
  llama/causal_lm/pytorch-3.1_8B-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "OOM in attention layer — grid_sample index tensor blowup (tracking issue #1234)"
```

**PyTorch LLMs (decode/prefill):** if you implement `load_inputs_decode` / `load_inputs_prefill`, also add entries under `tests/runner/test_config/torch_llm/` using the long IDs. See `references/test_ids_and_yaml.md`.

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

Use the **exact** parametrized id (full `rel_path` and variant value):

```bash
source venv/activate
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<rel_path>-<VariantValue>-single_device-inference]"
```

**IRD cache for `get_file()` models:** If the model loader uses `get_file()` to download
weights (custom/non-HuggingFace weights), set the IRD cache before running:
```bash
export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
```

**PyTorch note:** `TorchDynamicLoader` passes **`dtype_override=torch.bfloat16`** to `load_model` / `load_inputs` when those methods accept `dtype_override`, so the default device comparison path often uses bfloat16 — align PCC expectations accordingly.

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
    def load_model(self, **kwargs): ...          # base.py signature; loaders use *, dtype_override=None
    def load_inputs(self, **kwargs): ...         # base.py signature
    def _get_model_info(cls, variant) -> ModelInfo: ...  # abstract classmethod

    # Optional
    def decode_output(cls, **kwargs): ...
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
| Loader (Pattern A) | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| Loader (Pattern B) | `third_party/tt_forge_models/<model>/<task>/pytorch/loader.py` |
| Single-device config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| LLM configs | `tests/runner/test_config/torch_llm/test_config_inference_*.yaml` |
| Test entry point | `tests/runner/test_models.py` |
| Test ID / YAML reference | `references/test_ids_and_yaml.md` |
| Loader templates reference | `references/loader_templates.md` |

## Pre-Submit Checklist

- [ ] `ModelLoader` extends `ForgeModel` with correct `_VARIANTS`, `DEFAULT_VARIANT`, `ModelVariant`
- [ ] `_get_model_info` returns `ModelInfo` with appropriate `ModelGroup`, `ModelTask`, `ModelSource`, `Framework.TORCH`
- [ ] `load_model` / `load_inputs` follow project conventions (`dtype_override` where expected for the harness)
- [ ] `load_model` calls `model.eval()` before returning
- [ ] `load_inputs` returns tensors matching the model's actual forward signature
- [ ] `__init__.py` files re-export `ModelLoader` and `ModelVariant`
- [ ] SPDX copyright header on all new `.py` files
- [ ] CPU forward pass succeeds in both default dtype and bfloat16
- [ ] YAML keys use full **`rel_path`** (see `references/test_ids_and_yaml.md`), initial status is `EXPECTED_PASSING` (never `KNOWN_FAILURE_XFAIL`)
- [ ] `requirements.txt` next to `loader.py` if non-standard pip deps are required
- [ ] Submodule: loader commits in `third_party/tt_forge_models`, tt-xla commit updates the submodule pointer
- [ ] TT device PCC meets threshold (0.99 default, or documented `required_pcc`)
- [ ] Status promoted to `EXPECTED_PASSING` after stable validation
