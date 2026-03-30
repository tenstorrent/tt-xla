# BEVDepth LSS — DRAM Auto Slice Failure

## Model
`bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da`

## Error
```
TT_FATAL: DRAM Auto slice could not find valid slice configuration.
Tried up to 2 slices for width-slicing on output dimension 44.
Available L1: 1329888 bytes. Operation requires more memory than available
even with maximum slicing.
```

Source: `ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266`

## Failing Op

**`ttnn::conv2d` (Conv2dOp)** — the ASPP dilated convolution with `dilation=18` inside `DepthNet`.

```python
nn.Conv2d(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    padding=18,
    dilation=18,
    bias=False,
)
# input shape:  (6, 512, 16, 44)
# output shape: (6, 512, 16, 44)
```

## Data Flow to the Failing Op

```
6 camera images: (6, 3, 256, 704)
        │
        ▼
ResNet50 backbone (img_backbone)
  stem: Conv2d(3→64, k=7, stride=2) + MaxPool(stride=2)  →  (6, 64,   64, 176)
  layer1 (stride=1)                                       →  (6, 256,  64, 176)
  layer2 (stride=2)                                       →  (6, 512,  32,  88)
  layer3 (stride=2)                                       →  (6, 1024, 16,  44)
  layer4 (stride=2)                                       →  (6, 2048,  8,  22)
        │
        ▼
SECONDFPN neck (img_neck) — resamples all scales to 16×44, concatenates
  deblock[0]: Conv2d(256→128,  k=4, stride=4)           →  (6, 128, 16, 44)
  deblock[1]: Conv2d(512→128,  k=2, stride=2)           →  (6, 128, 16, 44)
  deblock[2]: ConvTranspose2d(1024→128, k=1, stride=1)  →  (6, 128, 16, 44)
  deblock[3]: ConvTranspose2d(2048→128, k=2, stride=2)  →  (6, 128, 16, 44)
  cat(dim=1)                                            →  (6, 512, 16, 44)
        │
        ▼
DepthNet (in_channels=512, mid_channels=512)
  reduce_conv: Conv2d(512→512, k=3, p=1)    →  (6, 512, 16, 44)   ✅ passes
  3× BasicBlock(512, 512)                   →  (6, 512, 16, 44)   ✅ passes
  ASPP(512, 512):
    aspp1: Conv2d(512→512, k=1,  dilation=1)   ✅ passes
    aspp2: Conv2d(512→512, k=3,  dilation=6)   ✅ passes
    aspp3: Conv2d(512→512, k=3,  dilation=12)  ⚠️  wrong PCC (0.31)
    aspp4: Conv2d(512→512, k=3,  dilation=18)  ❌ DRAM slice FATAL  ← root cause
```

## Root Cause

The `dilation=18` conv at 512→512 channels and output spatial `16×44` requires L1 circular buffers that exceed the available 1,329,888 bytes. The TTNN DRAM auto-slicer attempts to split the output along the width dimension (44 → 22 per slice) but even 2 slices are insufficient. The slicer gives up and raises a fatal.

Note: `dilation=12` also produces incorrect results (PCC=0.31), indicating the dilated conv support at this scale/channel count is broken beyond just the fatal case.

## Sanity Test

**File:** `tests/torch/graphs/test_bevdepth_resnet50_conv2d.py`
**Saved input tensor:** `tests/torch/graphs/bevdepth_depthnet_input.pt`

## Steps to Replicate

### 1. Activate the environment
```bash
source venv/activate
```

### 2. Run the full model test (original failure)
```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[bevdepth/pytorch-bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da-single_device-inference]"
```

### 3. Run the isolated sanity test (minimal reproducer)
```bash
pytest -svv tests/torch/graphs/test_bevdepth_resnet50_conv2d.py
```

Expected output:
```
TT_FATAL: DRAM Auto slice could not find valid slice configuration.
Tried up to 2 slices for width-slicing on output dimension 44.
Available L1: 1329888 bytes. Operation requires more memory than available
even with maximum slicing.
FAILED tests/torch/graphs/test_bevdepth_resnet50_conv2d.py::test_bevdepth_depthnet_aspp_dilation18
```

### 4. Reproduce programmatically (no pytest)
```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()

model = torch.nn.Conv2d(
    in_channels=512,
    out_channels=512,
    kernel_size=3,
    padding=18,
    dilation=18,
    bias=False,
).to(device)

x = torch.rand(6, 512, 16, 44).to(device)
out = model(x)  # <-- triggers DRAM slice FATAL
xm.mark_step()
```
