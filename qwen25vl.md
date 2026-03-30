# Qwen 2.5 VL — Conv3d Static L1 Overflow

## Model
`qwen_2_5_vl/pytorch-3B_Instruct`

## Error
```
TT_THROW: Statically allocated circular buffers on core range
[(x=0,y=0) - (x=7,y=7)] grow to 1738528 B which is beyond max L1 size
of 1499136 B
```

Source: `tt_metal/impl/program/program.cpp:1043`

## Failing Op

**`ttnn::experimental::conv3d` (Conv3dDeviceOperation)** — the patch embedding convolution inside `Qwen2_5_VisionPatchEmbed`.

```python
nn.Conv3d(
    in_channels=3,
    out_channels=1280,
    kernel_size=(2, 14, 14),
    stride=(2, 14, 14),
    bias=False,
)
# input shape:  (1024, 3, 2, 14, 14)
# output shape: (1024, 1280, 1, 1, 1)
```

## Data Flow to the Failing Op

```
Input image: 448×448
        │
        ▼
Qwen2_5_VisionPatchEmbed
  Tiles image into 3D patches:
    spatial patches:  32×32 = 1024  (448/14 = 32 per axis)
    temporal groups:  1             (single frame)
  Reshaped input to Conv3d: (N=1024, 3, 2, 14, 14)
        │
        ▼
  nn.Conv3d(in=3, out=1280, kernel=(2,14,14), stride=(2,14,14))
  output: (1024, 1280, 1, 1, 1)   ← static CB allocation overflows L1
        │
        ▼
  Flatten → patch embeddings fed to ViT transformer layers
```

## Root Cause

The `Conv3d` op with `out_channels=1280` and `kernel=(2,14,14)` requires statically allocated circular buffers that grow to **1,738,528 B** — 239,392 B over the device maximum of **1,499,136 B**. This is a static allocation failure at program compile time (not a runtime memory pressure issue), so it fails in isolation regardless of model context.

The large N=1024 (number of patches) combined with 1280 output channels drives the circular buffer size beyond L1 limits.

## Sanity Test

**File:** `tests/torch/graphs/test_qwen_2_5_vl_patch_embed_conv3d.py`
**Saved input tensor:** `tests/torch/graphs/qwen_patch_embed_pixel_values.pt`

## Steps to Replicate

### 1. Activate the environment
```bash
source venv/activate
```

### 2. Run the full model test (original failure)
```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[qwen_2_5_vl/pytorch-3B_Instruct-single_device-inference]"
```

### 3. Run the isolated sanity test (minimal reproducer)
```bash
pytest -svv tests/torch/graphs/test_qwen_2_5_vl_patch_embed_conv3d.py
```

Expected output:
```
TT_THROW: Statically allocated circular buffers on core range
[(x=0,y=0) - (x=7,y=7)] grow to 1738528 B which is beyond max L1 size of 1499136 B
FAILED tests/torch/graphs/test_qwen_2_5_vl_patch_embed_conv3d.py::test_qwen_2_5_vl_patch_embed_conv3d
```

### 4. Reproduce programmatically
```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

model = torch.nn.Conv3d(
    in_channels=3,
    out_channels=1280,
    kernel_size=(2, 14, 14),
    stride=(2, 14, 14),
    bias=False,
).to(device)

x = torch.rand(1024, 3, 2, 14, 14).to(device)
out = model(x)  # <-- triggers static CB overflow fatal
xm.mark_step()
```
