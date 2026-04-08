# Graph Break Analysis: Mochi VAE Decoder on TT-XLA

## Table of Contents

1. [Problem Summary](#1-problem-summary)
2. [Diagnosis: Identifying the Graph Break](#2-diagnosis-identifying-the-graph-break)
3. [Root Cause: `super().unflatten()` in Dynamo](#3-root-cause-superunflatten-in-dynamo)
4. [Impact: Graph Fragmentation and DRAM OOM](#4-impact-graph-fragmentation-and-dram-oom)
5. [Fix: Monkey-Patching MochiChunkedGroupNorm3D](#5-fix-monkey-patching-mochichunkedgroupnorm3d)
6. [Results](#6-results)
7. [Lessons Learned](#7-lessons-learned)

---

## 1. Problem Summary

When running the Mochi VAE decoder with Megatron-style channel tensor-parallel
sharding (`decoder_sharded.py`), the model OOM'd on DRAM during the pixel shuffle
operation. The permute output tensor `[1,512,8,3,60,2,106,2]` was allocated in
tile layout as `memref<5898240x1×tile<32x32, bf16>>` = **~12 GB**, exceeding the
available ~732 MB per bank.

The root cause was not the pixel shuffle itself, but **42 graph breaks** from
`MochiChunkedGroupNorm3D` that shattered the decoder into 8 separate subgraphs.
This forced the pixel shuffle into an isolated 3-op subgraph where all
intermediate tensors became graph outputs and had to be materialized.

---

## 2. Diagnosis: Identifying the Graph Break

### Step 1: Enable graph break logging

```bash
TORCH_LOGS="graph_breaks" python decoder_sharded.py 2>&1 | tee decoder.log
```

### Step 2: Count graph breaks and unique graphs

```bash
# Graph break count
grep -c "Graph Break Reason" decoder.log
# → 42

# Unique SyncTensorsGraph IDs
grep -o "SyncTensorsGraph\.[0-9]*" decoder.log | sort -u
# → SyncTensorsGraph.14, .108, .4, .5, .6, .20, .77, .83  (8 graphs)
```

### Step 3: Identify the break reason

Every single graph break had the same reason:

```
Graph Break Reason: Attempted to call a super() attribute that is not a function or method
  Explanation: Dynamo does not know how to trace the call `super().unflatten()`
               because `super().unflatten` is not a function or method attribute.
  Hint: Ensure the attribute accessed via `super()` is a standard method or function.
```

Source: `torch/_tensor.py:1418` → called from
`MochiChunkedGroupNorm3D.forward()` at `autoencoder_kl_mochi.py:64`.

---

## 3. Root Cause: `super().unflatten()` in Dynamo

### The call chain

```
MochiChunkedGroupNorm3D.forward() line 64:
    output = output.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)
                    ↓
torch._tensor.Tensor.unflatten() line 1418:
    return super().unflatten(dim, sizes)
                    ↓
torch._C._TensorBase.unflatten  (C-level method_descriptor)
                    ↓
Dynamo: SuperVariable.call_method() → GRAPH BREAK
```

### Why Dynamo can't trace it

`Tensor.unflatten()` is a Python-level method that delegates to
`super().unflatten()`, resolving to `torch._C._TensorBase.unflatten` (a C-level
`method_descriptor`). Dynamo's `SuperVariable.call_method()` checks if this
descriptor is in `get_tensor_method()`, but it's **not** in that set because:

- `getattr(torch.Tensor, 'unflatten')` returns the **Python-level function**
  (from `_tensor.py`), not the C-level descriptor
- Since the Python function isn't a `MethodDescriptorType`, it never gets added
  to the `get_tensor_method()` set
- When `super()` resolves to the C-level descriptor, the check fails
- Execution falls through to `unimplemented_v2`, triggering the graph break

This is a known limitation in torch 2.7.0's Dynamo tracing
(see https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0006.html).

### Why 42 graph breaks

The Mochi decoder has **19 `MochiResnetBlock3D`** instances, each containing
**2 `MochiChunkedGroupNorm3D`** modules (norm1, norm2) = **38 unflatten calls**.
The extra 4 come from Dynamo compilation frame restarts after graph breaks.

### Note: the chunked norm pattern was also suboptimal

The original `MochiChunkedGroupNorm3D.forward()` also uses:
```python
output = torch.cat([self.norm_layer(chunk) for chunk in x.split(self.chunk_size, dim=0)], dim=0)
```

This list comprehension with `x.split()` requires Dynamo to unroll the loop at
trace time, creating specialization overhead. With `batch_size=1` and `T<=48`,
`B*T` is always small enough for single-pass GroupNorm, making chunking
unnecessary.

---

## 4. Impact: Graph Fragmentation and DRAM OOM

### Before fix: 8 subgraphs

| Graph | Operation | Source |
|-------|-----------|--------|
| **14** | `conv_in` (1x1x1 conv + bias) | Decoder.forward line 618 |
| **108** | ResBlock first half: norm1 + SiLU + causal conv | ResnetBlock3D lines 115-116 |
| **4** | `unflatten` reshape after chunked GroupNorm | ChunkedGroupNorm3D line 64 |
| **5** | `permute` (dim reorder after GroupNorm / after linear proj) | ChunkedGroupNorm3D line 64 |
| **77** | ResBlock conv1 path: SiLU + causal 3D conv + bias | ResnetBlock3D line 117 |
| **83** | ResBlock conv2 path: norm2 + SiLU + conv + residual | ResnetBlock3D lines 119-123 |
| **20** | Linear projection in UpBlock (768→6144) | UpBlock3D line 392 |
| **6** | **Pixel shuffle: reshape + permute + reshape** | UpBlock3D lines 401-403 |

### The pixel shuffle subgraph (SyncTensorsGraph.6)

Because graph breaks isolated the pixel shuffle into its own tiny subgraph, **all
3 intermediate tensors became graph outputs**:

```mlir
func.func @main(%arg0: tensor<1x6144x8x60x106xbf16>)
    -> (tensor<1x512x3x2x2x8x60x106xbf16>,      // %0 = reshape to 8D  (622 MB)
        tensor<1x512x8x3x60x2x106x2xbf16>,       // %1 = permute       (11.6 GB in tile!)
        tensor<1x512x24x120x212xbf16>) {           // %2 = reshape to 5D (622 MB)
  %0 = stablehlo.reshape %arg0
  %1 = stablehlo.transpose %0
  %2 = stablehlo.reshape %1
  return %0, %1, %2       // ← ALL THREE returned, only %2 needed
}
```

The permute output `%1` in tile layout = `memref<5898240x1×tile<32x32, bf16>>`
= **12,079,595,520 bytes** → OOM.

### Why all intermediates become outputs

When a Dynamo graph break isolates a subgraph, the XLA bridge's
`partition_fx_graph_for_cpu_fallback` (in `dynamo_bridge.py`) uses
`CapabilityBasedPartitioner` to create fused partitions. The
`fuse_as_graphmodule` function marks any node with a user **outside** its
partition as an output. When partition boundaries fall around the pixel shuffle,
all 3 intermediate tensors get external users and become partition outputs.

**In isolation, the pixel shuffle correctly produces 1 output.** The 3-output
problem is entirely caused by the surrounding graph fragmentation.

---

## 5. Fix: Monkey-Patching MochiChunkedGroupNorm3D

### The patch (`_patch_chunked_groupnorm` in `decoder_sharded.py`)

Two changes to eliminate the graph break:

**1. Replace chunked norm with direct call:**
```python
# Before (graph break from list comprehension + split):
output = torch.cat([self.norm_layer(chunk) for chunk in x.split(self.chunk_size, dim=0)], dim=0)

# After (single pass, no split/cat overhead):
output = self.norm_layer(x)
```

**2. Replace `unflatten` with `reshape`:**
```python
# Before (graph break from super().unflatten()):
output = output.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3, 4)

# After (reshape avoids the super() path entirely):
output = output.reshape(batch_size, -1, *output.shape[1:])
output = output.permute(0, 2, 1, 3, 4)
```

### Alternatives considered and tested

| Alternative | Works with Dynamo? | Notes |
|-------------|-------------------|-------|
| `x.unflatten(dim, sizes)` (bound method) | **Yes** | Dynamo intercepts at `LOAD_ATTR`, doesn't inline |
| `torch.unflatten(x, dim, sizes)` (functional) | **Yes** | Functional form, recognized by Dynamo |
| `x.reshape(B, -1, *x.shape[1:])` | **Yes** | Semantically equivalent, no super() call |
| `x.view(B, -1, *x.shape[1:])` | **Yes** | Same, but requires contiguous input |
| `torch.Tensor.unflatten(x, dim, sizes)` | **No** | Dynamo inlines body, hits super() |
| `torch._C._TensorBase.unflatten(x, ...)` | **No** | Dynamo can't trace C builtin directly |

We chose `reshape` because it handles non-contiguous tensors and is the most
robust option.

### Also applied: pixel shuffle `.contiguous()` removal

The original diffusers pixel shuffle does
`permute(0,1,5,2,6,3,7,4).contiguous()`. The `.contiguous()` forces
materialization of the permute intermediate. The `_pixel_shuffle_remove_contiguous`
patch removes it and uses `.reshape()` instead of `.view()` to handle
non-contiguous tensors without forcing a sync barrier.

---

## 6. Results

### Before patches

```
Graph breaks:            42
Unique SyncTensorsGraphs: 8  (14, 108, 4, 5, 6, 20, 77, 83)
Pixel shuffle graph:     3 outputs (reshape, permute, reshape) → 12 GB OOM
Result:                  DRAM OOM on permute output allocation
```

### After patches

```
Graph breaks:            0
Unique SyncTensorsGraphs: 1  (SyncTensorsGraph.7051)
Pixel shuffle:           Fused into the single unified graph
Result:                  No pixel shuffle isolation, no spurious intermediate outputs
```

The entire decoder now compiles into a **single graph** with no graph breaks.

---

## 7. Lessons Learned

### Debugging graph breaks on TT-XLA

1. **Always start with `TORCH_LOGS="graph_breaks"`** — it gives the exact reason,
   file, and line number for every graph break. This is the single most useful
   diagnostic.

2. **Count `SyncTensorsGraph` IDs** in the log to see how many subgraphs exist.
   Ideally there should be 1 for a simple forward pass.

3. **Graph breaks cause cascading problems** — a seemingly harmless break in
   GroupNorm created 8 subgraphs and forced 12 GB of unnecessary intermediate
   materialization.

### Common graph break patterns to watch for

| Pattern | Why it breaks | Fix |
|---------|--------------|-----|
| `super().unflatten()` | Dynamo can't trace `SuperVariable` for C method descriptors | Use `tensor.reshape()` instead |
| `torch.cat([f(x) for x in tensor.split(...)])` | List comprehension over data-dependent split | Call `f(tensor)` directly if possible |
| `.contiguous()` after `.permute()` | Forces materialization, becomes a graph output | Remove if not needed, or use `.reshape()` |
| Dict construction returned from forward | Dynamo must track all dict entries as live | Simplify return values for inference |
| `tensor.item()` or `tensor.tolist()` | Data-dependent Python value | Avoid in compiled code paths |

### The fragmentation amplification effect

A single graph break doesn't just add 1 extra subgraph — it can create a cascade:
- Each break creates 2+ subgraphs (pre-break, post-break)
- The XLA bridge's `CapabilityBasedPartitioner` may further split each subgraph
- Small subgraphs force ALL intermediate tensors to be materialized as graph I/O
- Tile layout on materialized tensors can have catastrophic padding overhead
  (e.g., `memref<5898240x1×tile>` = 19x overhead on a dim-2 tensor)

In this case: 1 graph break pattern × 38 occurrences = 42 breaks = 8 subgraphs
= 12 GB OOM on a tensor that should never have been materialized.
