---
name: model-bringup
description: Step-by-step guide for bringing up a new PyTorch or JAX model in the tt-xla / tt-forge model test suite. Use when the user wants to add, debug, or move a model to EXPECTED_PASSING status.
---

# Model Bringup for tt-xla

The golden rule: **establish CPU ground truth first, then bring up on TT and verify against it.**

Never touch TT without first knowing exactly what the correct output looks like on CPU.
Never stop at first successful run — inspect output quality and iterate until outputs are physically meaningful.

---

## Architecture overview (read this first)

Model bringup spans three layers:

```
1. Toyota-fresh/  (prototype / reference implementation)
   ↓ once working, wrap it
2. third_party/tt_forge_models/<model>/pytorch/loader.py  (production interface)
   ↓ loader auto-discovered by test runner
3. tests/runner/test_config/torch/*.yaml  (CI gate)
   + scripts/verify_model_cpu_vs_tt.py  (manual verification)
```

**For simple models** (standard HuggingFace, TorchVision, TIMM): skip Toyota-fresh, go straight to tt-forge-models loader.

**For complex models** (upstream repo with CUDA deps, custom C++ ops, non-standard weight naming): prototype in `Toyota-fresh/<model>/`, then wrap in loader.

**For JAX models**: no Toyota-fresh layer. JAX tests live in `tests/jax/single_chip/models/<model>/`. See [Part B](#part-b-jax-model-bringup).

---

## Part A: PyTorch Model Bringup

### Step A1 — Understand the model from source

Clone or locate the upstream repo. Install its deps in a **separate venv** — never pollute the tt-xla venv with model-specific packages.

Read the model source carefully:

- What are the inputs and their shapes? (batch, channels, spatial dims, sequence length)
- What preprocessing is required? (voxelization, tokenization, normalization, camera projection)
- What is the output? (logits, heatmaps, bounding boxes, embeddings)
- Does it need pretrained weights, or do random weights suffice for shape checking?
- What CUDA/native extensions does it use? (numba, spconv, iou3d_nms, DCN, flash attention)

At end of this step you must know:
- Exact input tensor shapes and dtypes
- Exact output structure (tensor / dict / list / tuple of tensors, shapes)
- What a correct output looks like numerically (value ranges, distributions)

### Step A2 — Remove CUDA/GPU dependencies

For models with CUDA deps, you need to make them CPU-runnable without touching the real inference path.

**Strategy:** Wrap every CUDA import in `try/except` and provide a no-op fallback. Do not modify forward pass logic — only the import mechanism.

**Common patterns:**

```python
# numba: used in geometry ops, NMS helpers
try:
    import numba
    from numba import jit, njit
except ImportError:
    def jit(*a, **kw): return lambda f: f
    def njit(*a, **kw): return lambda f: f

# spconv: sparse 3D convolution (PointPillars does NOT use it in forward)
try:
    import spconv
except ImportError:
    spconv = None

# CUDA NMS / iou3d ops
try:
    from iou3d_nms_cuda import nms_gpu
except ImportError:
    def nms_gpu(boxes, thresh): raise RuntimeError("CUDA NMS not available on CPU")
```

If the model already has a real upstream CPU path (common for PointPillars, many BERT variants), verify the fallback works end-to-end.

**Patch files:** When you have a proven set of patches for a model, save them as `Toyota-fresh/<model>/model/patches/cpu-inference.patch`. Apply with `git apply` from the upstream repo root. This makes patches reproducible and shareable.

Apply a patch:
```bash
cd third_party/<upstream_repo>
git apply ../../Toyota-fresh/<model>/model/patches/cpu-inference.patch
# Mark as applied to avoid re-applying:
touch .cpu_inference_patched
```

Write the patch generation:
```bash
cd third_party/<upstream_repo>
git diff > ../../Toyota-fresh/<model>/model/patches/cpu-inference.patch
```

### Step A3 — Run on CPU and establish ground truth

Run the model on CPU from end to end with real or realistic inputs.

```python
import torch

model.eval()
model = model.to(torch.bfloat16)
inputs = [x.to(torch.bfloat16) if x.is_floating_point() else x for x in inputs]

with torch.no_grad():
    cpu_output = model(*inputs)
```

**Why bfloat16?** TT hardware always runs in bfloat16. The CPU ground truth must be in the same dtype to avoid inflated error scores when comparing.

Inspect outputs critically:
- Are output shapes what you expected?
- Are value ranges physically plausible? (heatmaps: mostly negative pre-sigmoid; regression heads: ~O(1); class logits: typically -5 to +5)
- For 3D detection: after decoding, are box dimensions reasonable? (car ~4.5m long, pedestrian ~0.6m wide)
- For segmentation: do predicted classes make sense for the input?
- For classification: does softmax of logits sum to 1?

Save inputs and outputs for later comparison:
```python
torch.save(inputs, "tests/torch/graphs/<model_name>_inputs.pt")
torch.save(cpu_output, "tests/torch/graphs/<model_name>_cpu_output_bf16.pt")
```

**Do not proceed until you understand what correct output looks like.**

### Step A4 — Load pretrained weights (if needed)

Many complex models require a pretrained checkpoint. If not already part of the repo:

1. Find the official checkpoint (model card, paper, MMDetection3D, HuggingFace Hub).
2. Download it and cache it in `~/.cache/<model_name>/`.
3. Load and inspect it:
   ```python
   ckpt = torch.load(path, map_location="cpu", weights_only=False)
   state_dict = ckpt.get("state_dict", ckpt)
   print([k for k in state_dict.keys()][:20])
   ```
4. Compare checkpoint keys against model's `model.state_dict().keys()` — expect naming differences.

**Weight remapping** is almost always required when using an upstream checkpoint with a det3d/mmdet3d/detectron2 model. The renaming follows predictable patterns:

| Symptom | Cause | Fix |
|---|---|---|
| All weights zero after load | Keys didn't match — silent miss | Print `missing_keys` and `unexpected_keys` from `load_state_dict(strict=False)` |
| Index shifted by 1 | Upstream model has ZeroPad2d/Identity at idx 0 shifting Sequential indices | `new_idx = old_idx + 1` |
| `conv.weight` vs `0.weight` | Named sub-modules (`conv`) vs Sequential indices (`0`) | `k.replace("conv.", "0.").replace("bn.", "1.")` |
| `heatmap` vs `hm` | Different head naming conventions | String replacement |
| `tasks` vs `task_heads` | Framework naming convention difference | String replacement |

Write a `_remap_keys(state_dict)` function that does all substitutions, then verify:
```python
remapped = _remap_keys(state_dict)
missing, unexpected = model.load_state_dict(remapped, strict=False)
assert len(missing) == 0, f"Missing: {missing}"
# unexpected keys are OK (e.g., unused heads)
```

After loading: run a forward pass and verify outputs are non-trivial (not all zeros, ranges make sense). A random-weight model and a pretrained model will produce very different output distributions.

### Step A5 — Identify the TT-runnable portion

Go through the model and split preprocessing / postprocessing from the neural network core.

**CPU side (never sent to TT):**
- Dynamic-shape operations (variable-length NMS output, variable number of detected objects)
- Operations depending on Python control flow that changes with input values
- C++ CUDA extensions
- Operations that take numpy arrays or non-tensor inputs
- Point cloud voxelization, BEV scatter operations

**TT side (neural network core):**
- Convolutions, linear layers, normalization
- Attention (standard, not flash attention)
- Element-wise ops, reductions with static shapes
- Everything that can be expressed as StableHLO

**Test whether your split is correct:**
```python
# The TT portion must accept static shapes and return static shapes
bev = torch.randn(1, 64, 512, 512, dtype=torch.bfloat16)  # fixed shape
with torch.no_grad():
    output = tt_portion(bev)
print([t.shape for t in flatten_outputs(output)])  # must be fixed
```

**Common boundaries:**
- 3D detection: voxelize + scatter on CPU → RPN+head on TT → decode+NMS on CPU
- LLM: tokenizer on CPU → transformer on TT → decode on CPU
- Segmentation: preprocess on CPU → UNet/ViT on TT → postprocess on CPU

### Step A6 — Create the Toyota-fresh reference implementation (complex models only)

For models with upstream CUDA deps or non-trivial weight loading, build a standalone working reference first.

Structure:
```
Toyota-fresh/<model>/
  model/
    utils.py          ← submodule init, weight loading, remapping, pre/postprocessing
    patches/
      cpu-inference.patch
  test/
    test_<model>.py   ← pytest tests (CPU-only and CPU vs TT)
  third_party/
    <upstream_repo>/  ← git submodule (linked from repo root third_party/)
```

The test file should have at minimum:
1. A fast test on synthetic inputs (no dataset needed): verifies shapes and value ranges
2. A full-pipeline test on real data: verifies end-to-end correctness

```python
# Example synthetic test
def test_model_rpn_head():
    model = load_model_with_weights()
    bev = torch.randn(1, 64, 512, 512, dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model(bev)
    # Check shapes
    assert outputs[0]["hm"].shape == (1, 1, 128, 128)
    # Check value ranges
    hm = outputs[0]["hm"].float()
    assert hm.min() < 0 and hm.max() < 0  # pre-sigmoid, should be mostly negative

def test_model_full_pipeline():
    model, nusc = load_model_with_weights(), load_nuscenes()
    sample = nusc.sample[0]
    inputs = preprocess_sample(nusc, sample)
    outputs = model(inputs["bev"])
    detections = decode_and_postprocess(outputs)
    assert len(detections) > 0
    # Visualize
    plot_bev_detections(inputs["points"], detections, save_path="bev_output.png")
```

### Step A7 — Create the tt-forge-models loader

Model lives in `third_party/tt_forge_models/<model_name>/pytorch/`:

```
<model_name>/
  __init__.py
  pytorch/
    __init__.py
    loader.py        ← ModelLoader class
    requirements.txt ← pip deps specific to this model (optional)
    src/
      model.py       ← nn.Module (only if the model doesn't come from an external package)
```

**`loader.py` template:**

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""<Model> loader — <upstream_url>."""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import ModelConfig, ModelInfo, ModelGroup, ModelTask, ModelSource, Framework, StrEnum
from ...base import ForgeModel


@dataclass
class <Model>Config(ModelConfig):
    # Add model-specific config fields here
    pass


class ModelVariant(StrEnum):
    BASE = "Base"
    LARGE = "Large"


class ModelLoader(ForgeModel):

    _VARIANTS = {
        ModelVariant.BASE: <Model>Config(pretrained_model_name="<hf_id_or_name>"),
        ModelVariant.LARGE: <Model>Config(pretrained_model_name="<hf_id_or_name_large>"),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant=None) -> ModelInfo:
        return ModelInfo(
            model="<ModelName>",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.RED,           # RED = high priority; GENERALITY = general
            task=ModelTask.CV_OBJECT_DET,   # Pick from ModelTask enum
            source=ModelSource.GITHUB,      # Pick from ModelSource enum
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # For Toyota-fresh models:
        from toyota_fresh_utils import load_model_with_weights
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_model_with_weights(dtype=dtype)

        # For HuggingFace models:
        # model = AutoModel.from_pretrained(self._variant_config.pretrained_model_name)
        # if dtype_override is not None:
        #     model = model.to(dtype_override)
        # model.eval()
        # return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        # Return a tuple — even for a single input
        return (torch.randn(batch_size, 64, 512, 512, dtype=dtype),)
```

**Rules for `load_model()`:**
- Always return model in `eval()` mode
- Default to `torch.bfloat16` when no `dtype_override` provided
- Only the TT-runnable portion of the model — no preprocessing, no postprocessing

**Rules for `load_inputs()`:**
- All shapes must be fully static (no dynamic dims)
- Return a tuple, even for single inputs: `return (tensor,)`
- Default to `torch.bfloat16` for float inputs

**Per-model dependencies:** If the model needs packages not in the tt-xla venv, add a `requirements.txt` next to `loader.py`. The test runner auto-installs it before running tests.

```
# requirements.txt
nuscenes-devkit>=1.0.0
```

### Step A8 — Add to test config YAML

Edit `tests/runner/test_config/torch/test_config_inference_single_device.yaml`.

**The test ID format is:**
```
<model_path>/<pytorch_dir>-<VariantValue>-single_device-inference
```
where `<model_path>` is the path from `tt_forge_models/` root to the `pytorch/` directory's parent.

Example:
```yaml
  centerpoint/pytorch-CenterPoint_Pillar-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "Initial bringup — weights loading OK, TT compilation in progress"
    markers: [large]
```

**Start with `KNOWN_FAILURE_XFAIL`.** Only promote to `EXPECTED_PASSING` after PCC ≥ 0.99.

Available status fields:
```yaml
status: EXPECTED_PASSING | KNOWN_FAILURE_XFAIL | NOT_SUPPORTED_SKIP | EXCLUDE_MODEL
bringup_status: IN_PROGRESS | FAILED_FE_COMPILATION | FAILED_TTMLIR_COMPILATION | FAILED_RUNTIME | INCORRECT_RESULT
reason: "Human explanation and/or GitHub issue link"
required_pcc: 0.99   # override default 0.99 threshold
assert_pcc: false    # disable PCC check entirely (use sparingly, for LLMs)
markers: [push]      # or [nightly], [large], [extended]

# Per-architecture overrides:
arch_overrides:
  n150:
    status: EXPECTED_PASSING
  p150:
    status: KNOWN_FAILURE_XFAIL
    reason: "PCC 0.96 on p150 — https://github.com/tenstorrent/tt-xla/issues/XXXX"
```

**Validate your YAML edit:**
```bash
python tests/runner/validate_test_config.py
```

### Step A9 — Verify the test is discoverable

```bash
source venv/activate
pytest --collect-only -q tests/runner/test_models.py | grep <model_name>
```

If the test doesn't appear:
- Check your loader.py has no import errors: `python -c "import sys; sys.path.insert(0,'third_party'); from tt_forge_models.<model>.pytorch.loader import ModelLoader; print(ModelLoader())"`
- Check the directory structure has all `__init__.py` files
- Verify `ModelVariant` values match what you put in the YAML test ID

### Step A10 — Add to verify script

Add your model to `MODEL_REGISTRY` in `scripts/verify_model_cpu_vs_tt.py`:

```python
"<model_name>": [
    ModelSpec(
        "<model_name>/VariantValue",
        "tt_forge_models.<model_name>.pytorch.loader",
        "VariantValue",   # Must match ModelVariant enum value, not name
    ),
],
```

**The `variant` string must be the enum `.value` (e.g., `"CenterPoint_Pillar"`), not the enum name (e.g., `"CENTERPOINT_PILLAR"`).**

### Step A11 — Run on TT and verify

```bash
source venv/activate

# Run via test runner (standard CI path):
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<model_name>/pytorch-<VariantValue>-single_device-inference]"

# Run via verify script (gives more numeric detail):
python scripts/verify_model_cpu_vs_tt.py --models <model_name> --verbose
```

The verify script runs:
1. CPU inference in bfloat16 (golden reference)
2. TT device inference via `torch.compile(model, backend="tt", fullgraph=True)`
3. PCC comparison on all output tensors

**Acceptance criteria:**
- PCC ≥ 0.99 on all output tensors
- No shape mismatches
- No runtime errors

### Step A12 — Debug failures (see debugging loops below)

### Step A13 — Promote to EXPECTED_PASSING

Once PCC ≥ 0.99:

```yaml
  <model_name>/pytorch-<VariantValue>-single_device-inference:
    status: EXPECTED_PASSING
```

Remove `bringup_status` and `reason` if set. Keep `required_pcc` if you lowered it (with a comment explaining why).

---

## Part B: JAX Model Bringup

JAX tests live in `tests/jax/single_chip/models/<model_name>/` and use the `JaxModelTester` infrastructure. There is no runner YAML for JAX tests.

### Step B1 — Understand the model and JAX-specific setup

JAX models use one of three model types:
- `flax.nnx.Module` — new Flax NNX API (preferred for new models)
- `flax.linen.Module` — older Flax Linen API (common in HuggingFace Flax models)
- `FlaxPreTrainedModel` — HuggingFace Flax models (`FlaxBertModel`, `FlaxResNetModel`, etc.)

Key JAX-specific constraints:
- All shapes must be static at compile time — JAX traces through Python control flow
- PRNG keys must be passed explicitly (`rngs = nnx.Rngs(0)`)
- Models are functional: weights (params/state) are separate from the model definition
- No Python side effects inside `@jax.jit` — all I/O happens outside

### Step B2 — Run on CPU first

```python
import jax
import jax.numpy as jnp
from flax import nnx

rngs = nnx.Rngs(0)
model = MyModel(rngs)
model.eval()

x = jnp.ones((1, 3, 224, 224))
with jax.disable_jit():   # Run eagerly for debugging
    output = model(x)
print(output.shape, output.dtype)
```

For `linen.Module`:
```python
import flax.linen as nn
model = MyLinenModel()
variables = model.init(jax.random.PRNGKey(0), x)
output = model.apply(variables, x)
```

For HuggingFace Flax (`FlaxPreTrainedModel`):
```python
from transformers import FlaxResNetModel
model = FlaxResNetModel.from_pretrained("microsoft/resnet-50")
output = model(pixel_values=jnp.ones((1, 3, 224, 224)))
logits = output.logits
```

### Step B3 — Load pretrained weights (if converting from PyTorch checkpoint)

Use `torch_statedict_to_pytree` from `tests/jax/single_chip/models/model_utils.py`:

```python
import torch
from tests.jax.single_chip.models.model_utils import torch_statedict_to_pytree

state_dict = torch.load("checkpoint.pth", map_location="cpu")["state_dict"]

# patterns = list of (regex_pattern, replacement) for key renaming
# banned_subkeys = keys to exclude (e.g., "num_batches_tracked")
params = torch_statedict_to_pytree(
    state_dict,
    patterns=[
        (r"layer(\d+)\.", r"layers_\1."),
        (r"\.weight$", ".kernel"),
    ],
    banned_subkeys=["num_batches_tracked"],
    dtype=jnp.bfloat16,
)
```

Note: The utility auto-transposes kernels (2D: `(in, out)→(out, in)`, 4D conv: `(C_out, C_in, H, W)→(H, W, C_in, C_out)`). This handles the PyTorch vs JAX kernel layout difference.

### Step B4 — Create JAX loader in tt-forge-models

JAX loaders live in `third_party/tt_forge_models/<model_name>/image_classification/jax/loader.py` (or appropriate task directory).

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import nnx

from ....base import ForgeModel
from ....config import ModelConfig, ModelInfo, ModelGroup, ModelTask, ModelSource, Framework, StrEnum


class ModelVariant(StrEnum):
    BASE = "Base"


class ModelLoader(ForgeModel):
    _VARIANTS = {ModelVariant.BASE: ModelConfig(pretrained_model_name="<name>")}
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant=None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant=None):
        return ModelInfo(
            model="<ModelName>",
            variant=variant or cls.DEFAULT_VARIANT,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None, **kwargs):
        # NNX:
        rngs = nnx.Rngs(0)
        model = MyNNXModel(rngs)
        model.eval()
        return model

        # Or HuggingFace Flax:
        # from transformers import FlaxResNetModel
        # return FlaxResNetModel.from_pretrained(self._variant_config.pretrained_model_name)

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or jnp.bfloat16
        return {"pixel_values": jnp.ones((1, 3, 224, 224), dtype=dtype)}
```

### Step B5 — Create the JAX test file

Tests live in `tests/jax/single_chip/models/<model_name>/test_<model_name>.py`.

**NNX model pattern:**
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import JaxModelTester, RunMode
from third_party.tt_forge_models.<model>.jax import ModelLoader, ModelVariant


class <Model>Tester(JaxModelTester):
    def __init__(self, variant=None, **kwargs):
        self._loader = ModelLoader(variant)
        super().__init__(**kwargs)

    def _get_model(self):
        return self._loader.load_model()

    def _get_input_activations(self):
        return self._loader.load_inputs()


@pytest.fixture
def inference_tester():
    return <Model>Tester()


@pytest.mark.push
def test_<model>_inference(inference_tester):
    inference_tester.test()
```

**HuggingFace Flax / linen pattern** — requires `_wrapper_model` override to extract `.logits`:
```python
class ResNetTester(JaxModelTester):
    def _wrapper_model(self, f):
        def model(args, kwargs):
            return f(*args, **kwargs)[0].logits
        return model
```

**Run the test:**
```bash
source venv/activate
pytest -svv tests/jax/single_chip/models/<model_name>/test_<model_name>.py
```

### Step B6 — Common JAX bringup pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `ConcretizationTypeError` | Python control flow on traced value | Rewrite with `jnp.where`, `jax.lax.cond`, or static args |
| `Tracer` cannot be converted to concrete value | Shape depends on input value | Make shape a constant or static arg |
| `INVALID_ARGUMENT: Shapes must be equal` | Mismatch in input parameter vs model | Check `_get_input_parameters()` and `_get_input_activations()` match model expectation |
| All-zero outputs after weight load | PyTorch→JAX kernel layout not transposed | Use `torch_statedict_to_pytree` which handles transposition |
| Nan outputs | `bfloat16` overflow in LayerNorm or SoftMax | Check for intermediate large values; try float32 first |
| `Variable not in scope` (linen) | Model called outside `model.apply()` | Use `model.apply(variables, inputs)` pattern |
| Dtype mismatch compilation error | Mixed float32/bfloat16 params | Cast everything to same dtype before compiling |

---

## Part C: 3D / Structured-Output Models

### Dataset discovery

Common dataset locations (check in order):
```
/proj_sw/user_dev/ctr-lelanchelian/tt-xla/Toyota-fresh/bevfusion/tests/.nuscenes_mini/
~/.cache/nuscenes/
/data/nuscenes/
```

If not available, download nuScenes mini (~4 GB) from https://www.nuscenes.org/data/v1.0-mini.tgz.

Load a sample:
```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="/path/to/nuscenes", verbose=False)
sample = nusc.sample[0]

# LiDAR
lidar_token = sample["data"]["LIDAR_TOP"]
lidar_path = nusc.get_sample_data_path(lidar_token)
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)  # x,y,z,intensity,ring
```

### BEV geometry reference

For PointPillars / CenterPoint (NuScenes):
```python
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)  # x_min, y_min, z_min, x_max, y_max, z_max
VOXEL_SIZE = (0.2, 0.2, 8.0)                         # voxel size in meters
GRID_X = int((51.2 - (-51.2)) / 0.2)   # 512
GRID_Y = int((51.2 - (-51.2)) / 0.2)   # 512
OUT_SIZE_FACTOR = 4                       # RPN 512 → 128
```

Decode center heatmap peaks to 3D boxes:
```python
cx = (col + reg_x) * OUT_SIZE_FACTOR * VOXEL_SIZE[0] + PC_RANGE[0]
cy = (row + reg_y) * OUT_SIZE_FACTOR * VOXEL_SIZE[1] + PC_RANGE[1]
cz = height
yaw = math.atan2(rot_sin, rot_cos)
dims = (l, w, h)  # apply exp() to raw dim predictions
```

### Postprocessing and visualization

Always decode and visualize outputs before calling a model "done":

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Plot LiDAR points (BEV projection)
ax.scatter(points[:, 0], points[:, 1], s=0.1, c='gray', alpha=0.3)
# Plot detected boxes
for box in detections:
    rect = patches.Rectangle(
        (box.x - box.l/2, box.y - box.w/2),
        box.l, box.w,
        linewidth=1, edgecolor=class_color[box.cls], facecolor='none'
    )
    ax.add_patch(rect)
ax.set_xlim(-51.2, 51.2)
ax.set_ylim(-51.2, 51.2)
ax.set_aspect('equal')
plt.savefig("bev_detections.png", dpi=150)
```

With real nuScenes data and pretrained weights: expect cars to appear along roads, pedestrians on sidewalks, tight bounding boxes around real objects. If boxes are scattered randomly, the decode math or coordinate system is wrong.

### How to decide: model math error vs comparison infra error

**Visualization test:** If visualization shows sensible detections but PCC is low → comparison infra issue (output pytree structure mismatch, ordering difference in list outputs).

**Stat test:** If intermediate tensor stats (mean, std) are wildly off → model math error (wrong weight loading, wrong preprocessing).

**Ablation:** Remove postprocessing and compare raw heatmaps directly. If raw heatmaps match, the error is in postprocessing. If raw heatmaps diverge, it's in the backbone.

---

## Debugging loops

### PCC < 0.99

1. Is the output structure correct? Print shapes of all CPU and TT tensors.
2. Run `--verbose` in verify script to see which specific output tensor fails.
3. Check if the failing tensor is a regression head vs classification head — regression often degrades faster.
4. Did you replace any ops (MaxPool3d → Conv3d, ConvTranspose3d → interpolate)? Replacements change numerical values. Check the replacement is semantically equivalent.
5. Is `fullgraph=True` capturing the whole model? If there are graph breaks, partial compilation can silently drop layers.

### Compilation error

```
Bad StatusOr access: INTERNAL: Error code: 13
```
→ Check tt-mlir logs: `TTXLA_LOGGER_LEVEL=DEBUG python ...`
→ Simplify the op that fails: isolate the subgraph in a graph test under `tests/torch/graphs/`

```
Statically allocated circular buffers grow to X B
```
→ L1 OOM: reduce channels or batch size. Or wait for compiler fix.

```
DRAM Auto slice could not find valid slice configuration
```
→ Large dilated Conv2d that can't be sliced. Known issue for certain dilation+spatial combinations. File issue and mark `NOT_SUPPORTED_SKIP`.

```
Shardy propagation only supports ranked tensors with a static shape
```
→ Dynamic shape. All input dims must be concrete integers.

```
Input type (float) and bias type (BFloat16)
```
→ Input dtype mismatch. Ensure all inputs are cast to bfloat16 before TT device.

### Wrong value ranges after weight load

1. Check `missing_keys` and `unexpected_keys` from `load_state_dict(strict=False)`.
2. Print norm of each layer's weights after loading: any all-zero layer means loading failed for that layer.
3. Compare the pretrained model output distribution vs a random-weight model: pretrained should produce confident predictions (high-activation peaks), random weights produce noisy near-zero outputs.
4. If outputs look like random noise despite claimed weight loading, the remapping is wrong — print 5 matching key pairs to verify.

### Wrong output shapes

1. Add `print(output.shape)` at every major module boundary.
2. Check all `__init__.py` list sizes match: `layer_nums`, `ds_num_filters`, `us_num_filters` in RPN must agree with head `in_channels`.
3. For 3D detection: RPN output is `(B, 384, 128, 128)` for CenterPoint PointPillars on nuScenes.

### Graph breaks (more graphs than expected)

Graph breaks are silent compilation failures — the model runs, but suboptimally or with missing layers.

```python
import torch._dynamo as dynamo
dynamo.explain(model)(inputs)  # shows where breaks occur
```

Common causes: print statements, Python-side conditionals on tensor values, `.item()` calls, unsupported ops. Remove them or wrap in `torch.compiler.disable()` to push them to CPU.

### TT comparison false negatives

The verify script compares all tensors returned by the model. If the model returns a list of dicts (3D detection common), the flattening order matters. Verify with:
```python
cpu_flat = flatten_outputs(cpu_out)
tt_flat = flatten_outputs(tt_out)
for i, (c, t) in enumerate(zip(cpu_flat, tt_flat)):
    print(i, c.shape, t.shape)
```

If shapes match but values don't, consider whether the outputs include derived tensors (e.g., NMS filtered boxes with non-deterministic ordering). Move NMS to CPU-side postprocessing if it causes ordering non-determinism.

---

## Known tt-mlir limitations (as of 2026-03)

| Op | Issue | Workaround |
|---|---|---|
| `MaxPool3d` | No bfloat16 support | Replace with stride-2 `Conv3d` |
| `ConvTranspose3d` (lhs_dilation) | Incorrect result in tt-mlir | `F.interpolate` + `Conv3d(1×1×1)` |
| `Conv2d` dilation≥8, large spatial dims | DRAM slice failure | Needs compiler fix; mark NOT_SUPPORTED_SKIP |
| `Conv3d` out_channels≥1280, large spatial | L1 static CB overflow | Needs compiler fix; reduce model size |
| Dynamic token shapes | Shardy requires static shapes | Not supported for autoregressive LLMs |
| Flash attention / SDPA with causal mask | Not always compiled | Fall back to manual attention computation |
| Models > ~8GB | OOM on single chip | Multi-chip or NOT_SUPPORTED_SKIP |

---

## Quick reference

| What | Where |
|---|---|
| Model source | `third_party/tt_forge_models/<model>/pytorch/src/model.py` |
| Model loader (PyTorch) | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| Model loader (JAX) | `third_party/tt_forge_models/<model>/<task>/jax/loader.py` |
| Test config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| CPU vs TT verify script | `scripts/verify_model_cpu_vs_tt.py` |
| JAX tests | `tests/jax/single_chip/models/<model>/test_<model>.py` |
| Saved ground truth tensors | `tests/torch/graphs/<model_name>_*.pt` |
| Toyota-fresh reference | `Toyota-fresh/<model>/` |
| Run PyTorch model test | `pytest tests/runner/test_models.py -k <model_name>` |
| Run JAX model test | `pytest tests/jax/single_chip/models/<model>/` |
| Validate test config | `python tests/runner/validate_test_config.py` |
| Discover all tests | `pytest --collect-only -q tests/runner/test_models.py` |
