<!--
TODO(odjuricic) sections to be added:
* async exec with 2 CQs
* int quantization?
* d2m
* emitPy
* Section on perf tooling?
* Section on performance profiling and spotting bottlenecks
  * Tracy support PLEASE
* AMP?
* multichip / multihost?
-->

# Improving Model Performance

This guide covers best practices and techniques for optimizing the performance of PyTorch models running on single chip Tenstorrent hardware using the tt-xla frontend of the forge compiler.

## Overview

1. **[Optimization Levels](#1-optimization-levels)** - Compiler optimization levels (0, 1, 2) to balance compile and runtime performance
2. **[Device Warmup](#2-device-warmup)** - Eliminate first-run overhead by performing warmup iterations
3. **[Data Formats](#3-data-formats)** - Use bfloat16 and bfloat8_b for faster computation and reduced memory usage
4. **[Runtime Trace](#4-runtime-trace)** - Reduce host-device communication overhead by recording and replaying command sequences
5. **[Batch Size Tuning](#5-batch-size-tuning)** - Find the optimal batch size to maximize throughput for your model

For a complete working example, see the code below from `examples/pytorch/mnist_performant.py`, which demonstrates all these optimizations together.
### Mnist peformant example:
```python
{{#include ../../../examples/pytorch/mnist_performant.py}}
```

Let's break down each performance optimization in detail.

---

## 1. Optimization Levels

The `optimization_level` compiler option controls multiple optimization passes from tt-mlir in a coordinated way. tt-xla offers three levels (0, 1, 2).

To set the optimization level, use:
```python
torch_xla.set_custom_compile_options({
    "optimization_level": "1",
})
```

### Optimization Levels Breakdown

#### Level 0 (Default)
- All MLIR optimizer passes disabled
  - All tensors in DRAM
- **Use for:** Iterating fast, safest option
- **Compilation time:** Fastest
- **Runtime performance:** Slowest

#### Level 1 (Recommended)
- Basic optimizations enabled
  - Const-eval of Conv2D weights preprocessing and fusion patterns
  - All tensors in DRAM
- **Use for:** General model compilation, good balance
- **Compilation time:** Moderate
- **Runtime performance:** Good

#### Level 2
- Advanced optimizations enabled, all level 1 plus:
  - Maximize number of tensors to put in SRAM instead of DRAM
- **Use for:** Maximum performance
- **Compilation time:** Slower (one-time cost)
- **Runtime performance:** Best

---

## 2. Device Warmup

Run at least 3 dummy iterations before measuring performance:
```python
# Warmup iterations.
with torch.no_grad():
    for _ in range(3):
        output = model(input)
```

### Why Warmup is Necessary

The first iteration is extremely slow due to it running:
* Model compilation and optimization
* Op kernel compilation
* Transferring of model weights to device
* Const-eval of model weight and constants
* Caching of op kernels on device

The second iteration is needed for:
* Capturing runtime trace to reduce op dispatch overhead (Section 4)

All of the above is a one time fixed cost and all subsequent iterations of the model will be orders of magnitude faster.

---

## 3. Data Formats

TT Hardware supports multiple lower precision data formats ([docs](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/tensor.html#data-type)). For use trough tt-xla try the following:
* bfloat16
* bfloat8_b

### bfloat16

To use bfloat16, convert your model in pytorch **before** compiling:
```python
# Convert model weights and operations to bfloat16.
model = model.to(dtype=torch.bfloat16)
```

Ensure your input tensors match the model's data type:

```python
inputs = inputs.to(torch.bfloat16)
```

bfloat16 (Brain Floating Point 16-bit) provides:
- **Faster computation** compared to fp32
- **Reduced memory usage** (50% of fp32)
- **Better utilization** on TT hardware
- **Minimal to no accuracy loss** for most workloads

### bfloat8_b

Enable bfp8 conversion using compile options. The model **MUST** be cast to bfloat16 before compilation.
```python
torch_xla.set_custom_compile_options({
    "enable_bfp8_conversion": "true",  # Enable bfloat8_b
})
```

bfloat8_b (Block Float 8-bit) provides even faster computation and more memory reduction.

#### Notes

- **Possibility of accuracy loss** for some workloads
- **Verify output:** Check that accuracy is acceptable for your use case
- **Automatic conversion:** Model is automatically converted during compilation (for bfp8)
- **Not always beneficial:** Profile your specific model to verify improvement

---

## 4. Runtime Trace

### What is Runtime Trace?

[Runtime tracing](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) is a performance optimization that eliminates some of the host to device communication by recording the commands for dispatching operations and replaying these as a single command when executing a trace.

### How to Enable

**Step 1:** Set environment variable before importing torch_xla:

```python
import os
os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = "10000000"  # ~10MB
```

**Step 2:** Enable trace in compiler options:

```python
torch_xla.set_custom_compile_options({
    "enable_trace": "true",
})
```

### Requirements

- `TT_RUNTIME_TRACE_REGION_SIZE` should be set (recommended: `"10000000"` or 10MB)
  - The trace region size determines how much memory is allocated in DRAM for storing the trace. Adjust based on your model.
  - If you see trace-related errors, try increasing this value.
- Program cache must be enabled with `TT_RUNTIME_ENABLE_PROGRAM_CACHE` must be set to `"1"` (This is set by default)

---

## 5. Batch Size Tuning

Batch size impacts:
- **Throughput** (samples/second) - larger batches typically (not always) increase throughput
- **Latency** (time per sample) - larger batches increase per-sample latency
- **Memory usage** - larger batches require more device memory

### Tuning Process

1. **Typical values to start with** (e.g., 1, 2, 4, 8, 16, 32)
2. **Measure throughput** for each batch size
3. **Increase batch size** until:
   - Throughput plateaus or starts decreasing
     - Sometimes smaller batches can use SRAM much more effectively, leading to an overall greater throughput than using bigger batches
   - Memory is exhausted (OOM error)
