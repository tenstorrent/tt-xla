# Bug: depthwise causal conv1d (`ttnn.conv2d`, groups == channels, 1×W image) fails on Wormhole

## Summary

A **depthwise causal 1D convolution** — `groups == in_channels == out_channels`, a degenerate
`1 × W` image and a tiny `1 × K` kernel — is what Gated Delta Net / Qwen3-Next-style models use
for their token-mixing conv. Lowered through StableHLO → tt-mlir → TTNN it becomes a single
`ttnn.conv2d` with `groups == channels`, and that op **fails on Wormhole**, in two different ways
depending on channel count:

- **Large channels (C = 10240, the Qwen3.6-27B GDN `conv_dim`)** — `TT_FATAL` at execution:
  the conv2d DRAM auto-slicer cannot find an L1-fitting slice configuration, *even on an otherwise
  empty device*. Compile is fast (~0.8 s); the op simply cannot run.
- **Small channels (C = 24)** — the conv2d dispatches but **hangs on device readback**: the
  completion queue never signals and the host spin-waits indefinitely (>20 min, no progress).

Both are `ttnn.conv2d` failures for this shape class. We worked around it in the model by expressing
the depthwise causal conv as `K` shifted slice-multiply-adds (pure elementwise, no `ttnn.conv2d`),
which compiles and runs in seconds — but the underlying `conv2d` support gap should be fixed.

## Environment

- tt-mlir `53762683b`, tt-metal `1efd27cafdb`
- Wormhole b0, single device used for the repro (so memory pressure is *not* from other tensors)
- Reached via the tt-xla PJRT plugin (`torch-xla`, `backend="tt"`), but the failure is in the
  compiled TTNN program at runtime, reproducible from the StableHLO below.

## Minimal reproducer

```python
# conv_dev.py  —  python conv_dev.py <C> <T>
import sys, torch, torch.nn.functional as F
import torch_xla, torch_xla.core.xla_model as xm
C, K, T = int(sys.argv[1]), 4, int(sys.argv[2])
dev = xm.xla_device()
x = torch.randn(C, K - 1 + T).to(dev)     # [C, L]  channel-major
w = torch.randn(C, K).to(dev)             # [C, K]  depthwise kernel
y = F.conv1d(x.unsqueeze(0), w.unsqueeze(1), groups=C).squeeze(0)   # depthwise conv1d
torch_xla.sync()
print(y.sum().cpu())
```

- `python conv_dev.py 10240 32` → `TT_FATAL` (DRAM auto-slice), below.
- `python conv_dev.py 24 10`   → hangs on readback after `sync()`.

## What the graph looks like

### StableHLO (what XLA emits for `F.conv1d(..., groups=C)`)

1D conv is emitted as a rank-4 `convolution` with the singleton spatial dim on **height**:

```
%convolution.7 = f32[1,10240,1,32] convolution(
    f32[1,10240,1,35] %input,            # N=1, C=10240, H=1, W=35
    f32[10240,1,1,4]  %weight),          # O=10240, I/group=1, kH=1, kW=4
  window={size=1x4}, dim_labels=bf01_oi01->bf01,
  feature_group_count=10240
```

### TTNN IR (after tt-mlir lowering) — single grouped conv2d

```mlir
%5 = "ttnn.permute"(%2) {permutation = [0,2,3,1]}   // NCHW -> NHWC : 1x1x35x10240
%8 = "ttnn.to_layout"(%5) {layout = row_major}       loc("convolution.7_workaround")
%9 = "ttnn.conv2d"(%8, %7, %0) <{
        groups = 10240, in_channels = 10240, out_channels = 10240,
        input_height = 1, input_width = 35, kernel_size = [1, 4],
        stride = [1,1], padding = [0,0,0,0], dilation = [1,1],
        conv2d_config = <enable_kernel_stride_folding = false, config_tensors_in_dram = true>,
        compute_config = <math_fidelity = hifi4, fp32_dest_acc_en = true>}>
   : (tensor<1x1x35x10240xf32>, tensor<10240x1x1x4xf32>) -> tensor<1x1x32x10240xf32>
%10 = "ttnn.permute"(%9) {permutation = [0,3,1,2]}  // back to NCHW
```

Note tt-mlir already tags a partial **`convolution.7_workaround`** (the row-major relayout) on this
conv — i.e. there is an existing special-case that is insufficient for this shape.

## Failure mode A — C = 10240: DRAM auto-slice cannot fit L1 (TT_FATAL)

```
TT_FATAL: DRAM Auto slice could not find valid slice configuration. Tried up to 1 slices for
width-slicing on output dimension 32. Available L1: 1329760 bytes. Operation requires more memory
than available even with maximum slicing.
@ ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266: found_valid_config
backtrace:
  ttnn::operations::op_slicing::determine_slice_config(...)
  ttnn::operations::conv::conv2d::conv2d_DRAM(...)
  ttnn::conv2d(...)
  tt::runtime::ttnn::operations::conv::run(ConcatOp/Conv2dOp, ProgramContext&)
```

Compile finished in ~0.8 s; the device is otherwise empty. The `conv2d_DRAM` path's auto-slicer
(`op_slicing.cpp:266`, `determine_slice_config`) concludes the op needs more L1 than the 1.33 MB
available even at maximum width-slicing, and aborts. So the depthwise conv2d for this shape is
simply not runnable, independent of surrounding memory pressure.

(In the full Qwen3.6-27B run the same conv surfaces slightly earlier as an OOM on a ~21 MB DRAM
buffer in a `ConcatOp → tilize_with_val_padding` inside the conv2d_DRAM path — same root op.)

## Failure mode B — C = 24: hang on device readback

`sync()` returns (~0.8 s compile), then the program executes and **never completes the readback**.
A `py-spy --native` / `gdb` sample of the stuck process shows the host spin-polling the device
completion queue forever:

```
MainThread:  ... TransferFromDevice -> BlockUntilReady           (blocked)
hot thread:  FDMeshCommandQueue::read_completion_queue
           -> copy_buffer_data_to_user_space (mmio_device_id=3, channel=1)
           -> SystemMemoryManager::completion_queue_wait_front
           -> loop_and_wait_with_timeout ...                      (100% CPU spin)
```

i.e. the depthwise conv2d program is dispatched but its result readback never signals completion.

## Root cause (hypothesis) and source pointers

The general `ttnn.conv2d` is being asked to run a **depthwise conv (`groups == in == out == C`) with a
`1 × W` image and `1 × K` kernel**. tt-metal *has* a dedicated efficient depthwise-1D path, but it is
gated behind `is_1d_conv(kernel_height == 1 && image_height == 1)`, and the StableHLO→TTIR 1D→2D
reshape puts the kernel/image on dimensions that don't reliably hit that gate, so the op falls into
the generic `conv2d_DRAM` slicer, which can't fit/complete for this shape.

- tt-metal slicer that aborts: `ttnn/cpp/ttnn/operations/sliding_window/op_slicing/op_slicing.cpp:266`
  (`found_valid_config`, `determine_slice_config`); entered from `conv2d_DRAM`.
- tt-metal depthwise gate: `ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:538` (`is_1d_conv`),
  `:540` (`is_1d_depthwise_conv`).
- tt-metal weight prep / dense-grouped vs depthwise dispatch:
  `ttnn/cpp/ttnn/operations/conv/conv2d/prepare_conv2d_weights.cpp:1571-1590`; dedicated depthwise
  path `:921` (`convert_conv_weight_tensor_to_depthwise_layout`) + kernels
  `conv2d/device/kernels/compute_depthwise_conv1d.cpp`, `reader_depthwise_conv1d.cpp`.
- tt-mlir 1D→2D conv lowering: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
  `Legalize1DConvolutionPattern` / `convert1DConvTo2D` (~line 1651) — controls which spatial axis
  the singleton dim lands on, and already emits the `*_workaround` relayout.

## Suggested fixes

1. **tt-mlir**: in `Legalize1DConvolutionPattern::convert1DConvTo2D`, orient the inserted unit spatial
   dim so the depthwise 1D conv satisfies tt-metal's `is_1d_conv` (kernel_height == 1 && image_height
   == 1), routing it to the dedicated depthwise-1D kernel instead of the generic `conv2d_DRAM`.
2. **tt-metal**: make `conv2d_DRAM` / `op_slicing` handle a depthwise (`groups == channels`) `1 × W`
   conv without requiring all channels resident in L1; and make `is_1d_conv` orientation-agnostic.
3. Either of the above should also remove the small-C readback hang (it's the same op dispatched
   into a configuration the device can't complete).

## Current workaround (in tt-xla GDN)

`integrations/vllm_plugin/vllm_tt/layers/gdn/conv1d.py` implements the depthwise causal conv as
`K` shifted slice-multiply-adds (`y[:, t] = Σ_j w[:, j] * padded[:, t+j]`), which lowers to plain
elementwise mul/add (no `ttnn.conv2d`) and compiles+runs in seconds with identical numerics.
```
