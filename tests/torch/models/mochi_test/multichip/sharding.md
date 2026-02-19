# Sharding and Multi-Device Parallelism for Mochi on Tenstorrent Hardware

This document is a comprehensive guide to sharding — what it is, why it matters, how it works
in the tt-xla codebase (PyTorch/XLA), and how we apply it to the Mochi video generation model.

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [The Torch ↔ JAX ↔ XLA Connection](#2-the-torch--jax--xla-connection)
3. [How Sharding Works in PyTorch/XLA (SPMD)](#3-how-sharding-works-in-pytorchxla-spmd)
4. [Practical Examples from the Codebase](#4-practical-examples-from-the-codebase)
5. [Megatron-Style Tensor Parallelism Deep Dive](#5-megatron-style-tensor-parallelism-deep-dive)
6. [Applying Sharding to Mochi](#6-applying-sharding-to-mochi)

---

## 1. Theoretical Foundations

### Why Sharding Matters

Large models like Mochi's DiT (10 billion parameters, ~20 GB in bfloat16) simply don't fit on a
single device. Even when they do fit, the activation memory during forward passes — particularly
for high-resolution video generation — can push memory requirements far beyond a single chip.

**Sharding** (also called **model parallelism**) solves this by distributing a model's weights
and/or computations across multiple devices. The goal is to:

- **Reduce per-device memory**: Each device holds only a fraction of the model
- **Enable larger models**: Models that exceed single-device capacity become runnable
- **Potentially increase throughput**: More devices means more compute, if communication overhead is low

### Types of Parallelism

#### Data Parallel (DP)

**Idea**: Replicate the entire model on every device, but split the *batch* across devices.

```
Device 0: Full model copy, processes batch[0:B/N]
Device 1: Full model copy, processes batch[B/N:2B/N]
...
Device N: Full model copy, processes batch[(N-1)B/N:B]
```

**Pros**: Simple to implement, no model changes needed.
**Cons**: Every device must hold the full model → no memory savings for model weights.
**When to use**: Large batch sizes with models that fit on a single device.
**NOT suitable for Mochi**: We use batch_size=1 (one video at a time), so there's nothing to split.

#### Tensor Parallel (TP)

**Idea**: Split individual weight tensors across devices. Each device holds a *shard* (slice)
of each weight matrix and computes its portion of the output.

```
Weight W (shape [M, N]):
  Device 0: W[0:M/D, :]     (column-parallel, splits rows)
  Device 1: W[M/D:2M/D, :]
  ...

  OR

  Device 0: W[:, 0:N/D]     (row-parallel, splits columns)
  Device 1: W[:, N/D:2N/D]
  ...
```

**Pros**: Linear memory reduction for weights. Works with batch_size=1.
**Cons**: Requires collective communication (ALL-REDUCE) after certain operations.
**When to use**: Large models with batch_size=1. The standard choice for transformer inference.
**This is what we use for Mochi's DiT.**

#### Pipeline Parallel (PP)

**Idea**: Split the model *by layers* across devices. Device 0 runs layers 0-11, Device 1 runs
layers 12-23, and so on. Data flows sequentially through the "pipeline".

```
Device 0: layers 0-11    →  Device 1: layers 12-23  →  Device 2: layers 24-35  →  Device 3: layers 36-47
```

**Pros**: Simple partitioning, no changes to individual layers.
**Cons**: "Pipeline bubble" — devices sit idle while waiting for their turn.
Only the device currently processing is active; the rest are idle.
**When to use**: When TP communication overhead is too high, or model has many sequential layers.

#### Context / Sequence Parallel (CP/SP)

**Idea**: Split the *sequence* (or temporal) dimension across devices. Each device processes a
subset of tokens or frames.

```
Video frames: [f0, f1, f2, ..., f23]
  Device 0: [f0, f1, f2, f3, f4, f5]
  Device 1: [f6, f7, f8, f9, f10, f11]
  Device 2: [f12, f13, f14, f15, f16, f17]
  Device 3: [f18, f19, f20, f21, f22, f23]
```

**Pros**: Natural for video (temporal coherence). Reduces activation memory.
**Cons**: Requires all-to-all communication during attention layers.
**When to use**: Long sequences, video generation. The tt-metal Mochi bringup team used this approach.

### Trade-offs Summary

| Strategy | Memory Savings | Communication | Impl. Complexity | Best For |
|----------|---------------|---------------|-------------------|----------|
| Data Parallel | None (weights) | ALL-REDUCE gradients | Low | Large batches |
| Tensor Parallel | Linear (weights) | ALL-REDUCE per layer | Medium | Large models, BS=1 |
| Pipeline Parallel | Linear (layers) | Point-to-point | Medium | Deep models |
| Context Parallel | Linear (activations) | All-to-all in attention | High | Long sequences/video |

---

## 2. The Torch ↔ JAX ↔ XLA Connection

Both PyTorch/XLA and JAX are supported in tt-xla. Understanding their relationship helps
clarify why sharding concepts are the same across both frameworks.

### The Shared Backend: XLA

**XLA** (Accelerated Linear Algebra) is a compiler for machine learning that optimizes
computation graphs. Both frameworks use XLA as their compilation backend:

```
PyTorch model
    ↓ torch_xla
    ↓ Traces operations, builds computation graph
    ↓
StableHLO (intermediate representation)  ←─── Both produce this
    ↓                                         ↑
    ↓                                    JAX model
    ↓                                        ↓ jax.jit
    ↓                                        ↓ Traces operations
    ↓
tt-mlir compiler
    ↓
Tenstorrent binary (runs on hardware)
```

- **PyTorch/XLA** (`torch_xla`): Intercepts PyTorch operations via lazy tensors, builds XLA HLO
  (High-Level Optimizer) graphs, then compiles them.
- **JAX**: Natively built on XLA. `jax.jit` compiles Python functions to XLA HLO directly.

### PJRT: The Shared Runtime

**PJRT** (Portable JAX Runtime) is the interface that connects XLA to hardware backends.
Despite the name containing "JAX", PJRT is used by *both* frameworks:

- `pjrt_plugin_tt.so` — Tenstorrent's PJRT plugin, built in this repository
- This single plugin serves both JAX and PyTorch/XLA
- When you call `xr.set_device_type("TT")` in PyTorch/XLA, it loads this PJRT plugin

### Shardy: The Shared Sharding Framework

**Shardy** is Google's sharding framework, embedded in StableHLO. Both PyTorch/XLA and JAX
express sharding intent via Shardy annotations:

- **In PyTorch/XLA**: `xs.mark_sharding(tensor, mesh, spec)` → generates Shardy annotations
  in the StableHLO graph. Enabled by setting `CONVERT_SHLO_TO_SHARDY=1`.
- **In JAX**: `jax.sharding.NamedSharding(mesh, PartitionSpec(...))` → same Shardy annotations.

This means the concepts are identical across frameworks — **Mesh**, **PartitionSpec** (or
sharding tuples), and **SPMD** (Single Program Multiple Data) — they're just different Python
APIs wrapping the same XLA primitives.

### Why This Matters for Mochi

Since we work with **PyTorch** (via `torch_xla`), all our sharding code uses the PyTorch/XLA
SPMD APIs. But the underlying compilation path is the same as JAX. If you see JAX sharding
examples in the codebase (under `tests/jax/`), the concepts transfer directly — only the API
surface differs.

---

## 3. How Sharding Works in PyTorch/XLA (SPMD)

### Step 1: Enable SPMD Mode

Before any sharding can happen, you must enable SPMD mode:

```python
import os
import torch_xla.runtime as xr

# Enable Shardy conversion in PyTorch/XLA's StableHLO output
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

# Enable SPMD (Single Program Multiple Data) mode
# WARNING: This cannot be disabled once set!
xr.use_spmd()
```

This is implemented in `tests/infra/utilities/torch_multichip_utils.py`:

```python
def enable_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
```

### Step 2: Create a Device Mesh

A **Mesh** defines the logical arrangement of devices:

```python
import numpy as np
from torch_xla.distributed.spmd import Mesh

num_devices = xr.global_runtime_device_count()  # e.g., 8 on QuietBox
device_ids = np.arange(num_devices)

# 1×8 mesh with axes "batch" and "model"
mesh = Mesh(device_ids, mesh_shape=(1, 8), axis_names=("batch", "model"))
```

The **mesh_shape** defines the logical grid dimensions. The **axis_names** give each dimension
a human-readable name used in sharding specs:

```
mesh_shape=(1, 8) with axis_names=("batch", "model"):

         model axis (8 devices)
        ┌───┬───┬───┬───┬───┬───┬───┬───┐
batch=0 │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
        └───┴───┴───┴───┴───┴───┴───┴───┘
```

### Step 3: Mark Tensor Sharding

Use `xs.mark_sharding()` to annotate how a tensor should be distributed:

```python
import torch_xla.distributed.spmd as xs

# Shard weight along dim 0 across "model" axis, replicate dim 1
# Weight shape: [output_features, input_features]
xs.mark_sharding(weight_tensor, mesh, ("model", None))
#                                      ^^^^^^^^  ^^^^
#                                      dim 0     dim 1
#                                      sharded   replicated
```

**Partition spec rules:**
- Each element corresponds to one tensor dimension
- A **string** (e.g., `"model"`) means that dimension is sharded across the named mesh axis
- **`None`** means that dimension is replicated (every device has a full copy)
- The dimension size must be evenly divisible by the mesh axis size

**Examples:**

```python
# Fully replicated (every device has the full tensor):
xs.mark_sharding(tensor, mesh, (None, None))

# Shard rows across "model" axis (column-parallel):
# Weight [M, N] → each device gets [M/8, N]
xs.mark_sharding(weight, mesh, ("model", None))

# Shard columns across "model" axis (row-parallel):
# Weight [M, N] → each device gets [M, N/8]
xs.mark_sharding(weight, mesh, (None, "model"))

# Shard batch across "batch" axis (data parallel):
xs.mark_sharding(input, mesh, ("batch", None, None))
```

### Step 4: Compile and Run

After marking sharding, compile and run as normal:

```python
device = torch_xla.device()
model = model.to(device)

# Mark sharding on weights AFTER moving to device
for weight, spec in shard_specs.items():
    xs.mark_sharding(weight, mesh, spec)

# Compile with TT backend
compiled_model = torch.compile(model, backend="tt")

# Run inference
with torch.no_grad():
    output = compiled_model(**inputs)
```

The compiler sees the Shardy annotations in StableHLO and generates a program that:
1. Distributes weight shards to the appropriate devices
2. Inserts ALL-REDUCE operations where needed
3. Routes data between devices automatically

### User-Facing Sharding APIs

The `python_package/tt_torch/sharding.py` module provides two additional APIs for
constraining intermediate tensor sharding:

```python
from tt_torch import sharding_constraint_hook, sharding_constraint_tensor

# Hook-based: apply constraint to a module's output
handle = sharding_constraint_hook(model.embed_tokens, mesh, ("batch", None, None))

# Direct: apply constraint to a tensor
constrained = sharding_constraint_tensor(tensor, mesh, (None, None, None))
```

These are useful when you need to control how the compiler shards *intermediate* tensors
(activations), not just the weight tensors you mark with `xs.mark_sharding()`.

---

## 4. Practical Examples from the Codebase

### Example 1: Tensor Parallel LLM (Qwen3-8B)

Source: `examples/pytorch/qwen3_tp.py`

This example demonstrates Megatron-style tensor parallelism for a Qwen3-8B embedding model:

```python
# 1. Enable SPMD
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()

# 2. Load model
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B", torch_dtype=torch.bfloat16)

# 3. Create mesh
num_devices = xr.global_runtime_device_count()
mesh = Mesh(np.arange(num_devices), (1, num_devices), ("batch", "model"))

# 4. Move to device
device = torch_xla.device()
model = model.to(device)

# 5. Apply TP sharding to each transformer layer
for layer in model.layers:
    # MLP: column-parallel gate+up, row-parallel down
    xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    # Attention: column-parallel QKV, row-parallel output
    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

# 6. Compile and run
compiled_model = torch.compile(model, backend="tt")
with torch.no_grad():
    output = compiled_model(**inputs)
```

### Example 2: Data Parallel + Tensor Parallel MNIST

Source: `examples/pytorch/multichip/n300/utils.py`

For tensor parallel on a simple MNIST model:

```python
def apply_tensor_parallel_sharding_mnist_linear(model, mesh):
    # fc1: weight [hidden, input] → shard output features (column-parallel)
    xs.mark_sharding(model.fc1.weight, mesh, ("model", None))

    # fc2: weight [hidden, hidden] → shard input features (row-parallel)
    xs.mark_sharding(model.fc2.weight, mesh, (None, "model"))

    # fc3: weight [num_classes, hidden] → shard output features (column-parallel)
    xs.mark_sharding(model.fc3.weight, mesh, ("model", None))
```

Note the alternating pattern: column → row → column. This minimizes ALL-REDUCE operations.

---

## 5. Megatron-Style Tensor Parallelism Deep Dive

The Megatron sharding pattern (from the Megatron-LM paper by NVIDIA) is the standard approach
for tensor-parallel transformer models. Here's how it works in detail.

### Column-Parallel Linear

**Splits the output dimension** (weight rows) across devices.

```
Weight W: [out_features, in_features] = [M, N]

Device 0 gets: W[0:M/D, :]        (rows 0 to M/D)
Device 1 gets: W[M/D:2M/D, :]     (rows M/D to 2M/D)
...
Device D-1 gets: W[(D-1)M/D:M, :] (rows (D-1)M/D to M)

Partition spec: ("model", None)
```

Each device receives the **full input X** (replicated) and computes:

```
Y_i = X @ W_i^T    (local matmul, no communication)
```

The outputs `Y_0, Y_1, ..., Y_{D-1}` are **distinct slices** of the full output.
No communication is needed — each device produces a unique chunk of the output.

### Row-Parallel Linear

**Splits the input dimension** (weight columns) across devices.

```
Weight W: [out_features, in_features] = [M, N]

Device 0 gets: W[:, 0:N/D]
Device 1 gets: W[:, N/D:2N/D]
...

Partition spec: (None, "model")
```

Each device receives the **corresponding slice of the input** and computes:

```
Y_i = X_i @ W_i^T    (local matmul)
```

The outputs `Y_0, Y_1, ..., Y_{D-1}` are **partial sums** of the full output.
An **ALL-REDUCE** operation sums them to get the final result:

```
Y = Y_0 + Y_1 + ... + Y_{D-1}
```

### The Megatron MLP Pattern

For a typical transformer MLP with gate/up/down projections:

```
Input X (replicated on all devices)
    │
    ├──→ gate_proj (column-parallel) ──→ gate_i (local)
    │                                        │
    ├──→ up_proj (column-parallel) ───→ up_i (local)
    │                                        │
    │                            gate_i * up_i (local, elementwise)
    │                                        │
    │                                        v
    │                          down_proj (row-parallel) ──→ partial_i
    │
    └──→ ALL-REDUCE(sum of all partial_i) ──→ Output Y
```

**Key insight**: Only ONE ALL-REDUCE per MLP block — at the very end after the down projection.
The gate, up, and elementwise operations all happen locally with zero communication.

### The Megatron Attention Pattern

For multi-head attention with head-parallel QKV:

```
Input X (replicated)
    │
    ├──→ Q_proj (column-parallel) ──→ Q_i (local, heads for device i)
    ├──→ K_proj (column-parallel) ──→ K_i (local, heads for device i)
    ├──→ V_proj (column-parallel) ──→ V_i (local, heads for device i)
    │
    │    Self-Attention(Q_i, K_i, V_i)  ← fully local, no CCL!
    │              │
    │              v
    │         Attn_out_i (local)
    │              │
    │         O_proj (row-parallel) ──→ partial_i
    │
    └──→ ALL-REDUCE(sum of all partial_i) ──→ Output Y
```

**Requirement**: `num_heads % num_devices == 0`
(each device must get a whole number of attention heads)

**Key insight**: QKV projection AND self-attention are completely local to each device.
Only ONE ALL-REDUCE per attention block — at the output projection.

### Why Megatron Minimizes Communication

Per transformer block, the total CCL (Collective Communication Library) operations are:
- **1 ALL-REDUCE** after MLP down projection
- **1 ALL-REDUCE** after attention output projection
- **= 2 ALL-REDUCE** total per transformer block

Compare with naive approaches that might need ALL-REDUCE after every linear layer (6+ per block).

---

## 6. Applying Sharding to Mochi

### Mochi Architecture Recap

Mochi is a video generation model with three main components:

| Component | Params | Size (bf16) | Architecture |
|-----------|--------|-------------|-------------|
| DiT (Diffusion Transformer) | 10B | ~20 GB | 48 AsymmDiT blocks, 24 heads |
| VAE (Decoder) | 362M | ~724 MB | Conv3D-based, GroupNorm |
| T5-XXL (Text Encoder) | 4.7B | ~9.4 GB | Standard encoder transformer |

### QuietBox Hardware (4× p150b cards)

```
4 cards × 2 chips/card = 8 chips total
8 chips × 16 GB/chip = 128 GB total DRAM
```

### DiT: Megatron-Style Tensor Parallelism (8 devices)

The DiT is the largest component and the primary target for sharding.

**Why TP works well for DiT:**
- 24 attention heads / 8 devices = **3 heads per device** (evenly divisible)
- All linear layer dimensions are divisible by 8
- 20 GB / 8 devices = **2.5 GB/device** for weights alone
- Megatron pattern requires only 2 ALL-REDUCEs per block (96 total for 48 blocks)

**Each MochiTransformerBlock has two parallel streams (visual + text):**

**Visual stream** (3072-dimensional, 24 heads × 128 dim/head):

| Layer | Weight Shape | Shard Spec | Per-Device |
|-------|-------------|------------|------------|
| `attn1.to_q` | [3072, 3072] | ("model", None) | [384, 3072] |
| `attn1.to_k` | [3072, 3072] | ("model", None) | [384, 3072] |
| `attn1.to_v` | [3072, 3072] | ("model", None) | [384, 3072] |
| `attn1.to_out[0]` | [3072, 3072] | (None, "model") | [3072, 384] |
| `ff.net[0].proj` (SwiGLU) | [16384, 3072] | ("model", None) | [2048, 3072] |
| `ff.net[2]` (down) | [3072, 8192] | (None, "model") | [3072, 1024] |

**Text stream** (1536-dimensional input, projected to 3072 for attention):

| Layer | Weight Shape | Shard Spec | Per-Device |
|-------|-------------|------------|------------|
| `attn1.add_q_proj` | [3072, 1536] | ("model", None) | [384, 1536] |
| `attn1.add_k_proj` | [3072, 1536] | ("model", None) | [384, 1536] |
| `attn1.add_v_proj` | [3072, 1536] | ("model", None) | [384, 1536] |
| `attn1.to_add_out` | [1536, 3072] | (None, "model") | [1536, 384] |
| `ff_context.net[0].proj` | [8192, 1536] | ("model", None) | [1024, 1536] |
| `ff_context.net[2]` | [1536, 4096] | (None, "model") | [1536, 512] |

**Note on the last block** (block 47): The last block has `context_pre_only=True`, meaning
`ff_context` is `None`. Skip text MLP sharding for this block. The text attention layers
(`add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`) still exist and should be sharded.

**Note on biases**: QKV projections have `bias=False`. Output projections (`to_out[0]`,
`to_add_out`) have `bias=True` — leave biases unsharded (replicated). For row-parallel layers,
the ALL-REDUCE produces the full output, and the (replicated) bias is added after.

### VAE Decoder: Spatial Input Sharding (4 devices)

The VAE decoder cannot use tensor parallelism because:
1. **Conv3D layers** have kernel spatial dimensions — splitting output channels creates
   cross-device dependencies at kernel boundaries
2. **GroupNorm** requires all channels within a group to be on the same device

**Strategy**: Replicate all model weights, shard the **input spatially** along the height
dimension.

**Divisibility analysis** for input shape `[1, 12, 4, 60, 106]`:

| Dimension | Size | Divisible by 4? | Divisible by 8? |
|-----------|------|-----------------|-----------------|
| Batch | 1 | No | No |
| Channels | 12 | Yes (3) | No |
| Time | 4 | Yes (1) | No |
| **Height** | **60** | **Yes (15)** | No |
| Width | 106 | No | No |

**Decision**: Use **4 devices**, shard on **height** (dim 3).
- Input: 60 / 4 = 15 height per device
- After upsampling stages (each 2×): 30, 60, 120, 240, 480 — all divisible by 4
- Output: [1, 3, 24, 480, 848] → 480 / 4 = 120 height per device

```python
mesh = Mesh(np.arange(4), (1, 4), ("batch", "spatial"))
# Weights: no sharding (replicated automatically)
# Input: shard height dimension
xs.mark_sharding(input_tensor, mesh, (None, None, None, "spatial", None))
```

### Memory Budget

```
DiT (tensor parallel across 8 devices):
  Weights:     20 GB / 8 = 2.5 GB/device
  Activations: Sharded proportionally
  Total:       Fits comfortably in 16 GB/chip

VAE Decoder (replicated on 4 devices):
  Weights:     724 MB (replicated on each device)
  Activations: Sharded spatially (height / 4)
  Total:       Fits within 16 GB/chip with spatial sharding

T5-XXL (separate, not sharded here):
  Weights:     ~9.4 GB
  Note:        Run on CPU or separate device set
```
