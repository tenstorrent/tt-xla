---
name: tt-multi-chip
description: Add multi-chip tensor parallelism support to tt-forge-models loaders using get_mesh_config and load_shard_spec. Covers mesh topology, weight sharding strategies, and validation.
argument-hint: <model-name-or-loader-path>
---

# Multi-Chip Model Support — tt-forge-models

Guides the implementation of `get_mesh_config()` and `load_shard_spec()` in model loaders to enable tensor parallelism across multiple Tenstorrent devices. These two methods are all that a model loader needs to support multi-chip execution.

## External Resources

- [tt-forge-models base.py](third_party/tt_forge_models/base.py) — Default stubs
- [PyTorch XLA SPMD](https://pytorch.org/xla/release/2.5/index.html#pytorch-xla-spmd) — Underlying sharding API

## Overview

Multi-chip support in tt-forge-models follows a two-step pattern:

1. **`get_mesh_config(num_devices)`** — Define how devices are arranged in a 2D mesh
2. **`load_shard_spec(model)`** — Define how each weight tensor is partitioned across that mesh

The test runner calls these methods, constructs a PyTorch XLA mesh device, and applies sharding via `xs.mark_sharding(tensor, mesh, shard_spec)`.

## Prerequisites

Before adding multi-chip support:

1. The model loader already works on single device (load_model + load_inputs pass)
2. You know the model architecture (attention heads, MLP structure, layer names)
3. Understand the target hardware:
   - **N300/T3000**: 2–8 devices, mesh typically `(1, N)` or `(2, N/2)`
   - **Galaxy (TG)**: 32 devices, mesh typically `(4, 8)` or `(8, 4)`

## Step 1: Implement `get_mesh_config`

### API Signature

```python
def get_mesh_config(self, num_devices: int) -> tuple:
    """Returns (mesh_shape, mesh_axis_names).

    Args:
        num_devices: Total number of available devices.

    Returns:
        mesh_shape: Tuple[int, int] — 2D shape (batch_dim, model_dim).
        mesh_axis_names: Tuple[str, str] — Names for the mesh axes,
                         conventionally ("batch", "model").
    """
```

### Mesh Shape Rules

The mesh shape `(R, C)` must satisfy `R * C == num_devices`.

| Mesh Shape | Meaning | When to Use |
|-----------|---------|-------------|
| `(1, N)` | All devices on the model axis | Default for most models. All tensor parallelism, no data parallelism. |
| `(2, N/2)` | 2 rows for batch, rest for model | When N attention heads don't divide evenly by N devices. |
| `(4, 8)` | Galaxy layout | Llama 70B/405B on 32-device Galaxy. |
| `(8, 4)` | Galaxy layout (transposed) | Qwen 72B on Galaxy. |

### Critical Constraint: Attention Head Divisibility

The number of attention heads **must divide evenly** by the model axis size (the second dimension of mesh_shape). This is because each device handles `num_heads / model_dim` attention heads.

```python
assert config.num_attention_heads % mesh_shape[1] == 0, \
    f"Cannot split {config.num_attention_heads} heads across {mesh_shape[1]} devices"
```

If heads don't divide by `num_devices`, try `(2, num_devices // 2)`:
```python
def get_mesh_config(self, num_devices: int):
    if self.config.num_attention_heads % num_devices == 0:
        mesh_shape = (1, num_devices)
    elif num_devices % 2 == 0 and self.config.num_attention_heads % (num_devices // 2) == 0:
        mesh_shape = (2, num_devices // 2)
    else:
        raise ValueError(
            f"Cannot evenly distribute {self.config.num_attention_heads} "
            f"heads across {num_devices} devices"
        )
    return mesh_shape, ("batch", "model")
```

### Implementation Patterns

Refer to `references/mesh_config_patterns.md` for complete examples from Llama, Qwen, Mistral.

## Step 2: Implement `load_shard_spec`

### API Signature

```python
def load_shard_spec(self, model) -> dict:
    """Returns a mapping from weight tensors to shard specs.

    Args:
        model: The loaded model instance (must already be on device).

    Returns:
        Dict mapping torch.nn.Parameter → Tuple[Optional[str], ...]:
            - Each tuple has one entry per weight dimension
            - "model" → shard along the model mesh axis
            - "batch" → shard along the batch mesh axis
            - None → replicate (don't shard this dimension)
        Or None if the model is too small to benefit from sharding.
    """
```

### Shard Spec Tuple Convention

For a 2D weight tensor of shape `(out_features, in_features)`:

| Spec | Meaning |
|------|---------|
| `("model", "batch")` | Shard dim 0 across model axis, dim 1 across batch axis |
| `("batch", "model")` | Shard dim 0 across batch axis, dim 1 across model axis |
| `("model", None)` | Shard dim 0 across model axis, replicate dim 1 |
| `(None, "model")` | Replicate dim 0, shard dim 1 across model axis |

For a 1D tensor (e.g., bias of shape `(features,)`):

| Spec | Meaning |
|------|---------|
| `("model",)` | Shard across model axis |
| `("batch",)` | Shard across batch axis |
| `(None,)` | Replicate |

### Standard Sharding Rules for Transformer Models

Most transformer-based models follow this pattern:

```python
def load_shard_spec(self, model):
    shard_specs = {}

    for layer in model.model.layers:
        # MLP: gate/up projections shard output dim on model axis
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        # MLP: down projection shards input dim on model axis (transpose)
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        # Attention: Q/K/V shard output dim on model axis
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        # Attention: output projection shards input dim on model axis
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

    # LM head shards output dim on model axis
    shard_specs[model.lm_head.weight] = ("model", "batch")

    return shard_specs
```

### When Models Have Biases

Some models (e.g., Qwen) have biases on attention projections. Add 1D specs:

```python
shard_specs[layer.self_attn.q_proj.bias] = ("model",)
shard_specs[layer.self_attn.k_proj.bias] = ("model",)
shard_specs[layer.self_attn.v_proj.bias] = ("model",)
```

### Sharding Strategies

#### FSDP (Fully Sharded Data Parallel)

Shards weights across **both** mesh axes. Every device holds a unique slice of every weight. Maximizes memory distribution.

```python
# FSDP: use both "model" and "batch" axes
shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
shard_specs[layer.input_layernorm.weight] = ("batch",)
shard_specs[model.model.embed_tokens.weight] = (None, "batch")
```

#### Megatron (Tensor Parallelism Only)

Shards weights only on the `"model"` axis. The batch axis is `None` (replicated). Each device holds a full copy along the batch axis.

```python
# Megatron: "model" axis only, other axis is None
shard_specs[layer.mlp.up_proj.weight] = ("model", None)
shard_specs[layer.input_layernorm.weight] = (None,)
shard_specs[model.model.embed_tokens.weight] = (None, None)
```

### Small Model Guard

Models that fit on a single device don't need sharding. Return `None`:

```python
def load_shard_spec(self, model):
    if self._variant in [ModelVariant.SMALL_1B, ModelVariant.TINY_500M]:
        return None
    # ... sharding specs for larger variants
```

## Step 3: Validate Multi-Chip Support

### Inspection Checklist

1. **Layer names**: Print `model.named_parameters()` to verify the exact attribute paths:
   ```python
   for name, param in model.named_parameters():
       print(f"{name}: {param.shape}")
   ```
   This reveals whether the model uses `self_attn` vs `attention`, `mlp.up_proj` vs `feed_forward.up_proj`, etc.

2. **Attention heads**: Check `model.config.num_attention_heads` divides by the model axis.

3. **Key-value heads**: For GQA models, also check `model.config.num_key_value_heads` divides by the model axis.

4. **Bias presence**: Check if projections have `.bias` attributes (not all models do).

### Integration Test Pattern

The test runner exercises multi-chip via:

```python
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

num_devices = xr.global_runtime_device_count()
mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
mesh = xs.Mesh(range(num_devices), mesh_shape, mesh_names)

model = loader.load_model()
shard_specs = loader.load_shard_spec(model)

if shard_specs is not None:
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `num_heads % model_axis != 0` | Mesh model axis doesn't divide attention heads | Adjust mesh shape or use `(2, N/2)` |
| `KeyError: layer.xxx.weight` | Wrong attribute path for this model architecture | Print `named_parameters()` to find correct path |
| `Tuple length mismatch` | Shard spec tuple length != weight tensor dimensions | 2D weights need 2-tuples, 1D need 1-tuples |
| `None returned for large model` | Guard clause too aggressive | Only skip sharding for genuinely small variants |

## Architecture-Specific Layer Paths

Different HuggingFace model architectures use different attribute names. Here's a quick reference:

| Architecture | Layers | MLP Up | MLP Down | MLP Gate | Q/K/V/O Proj |
|-------------|--------|--------|----------|----------|-------------|
| Llama, Qwen, Mistral | `model.model.layers` | `mlp.up_proj` | `mlp.down_proj` | `mlp.gate_proj` | `self_attn.{q,k,v,o}_proj` |
| Falcon | `model.transformer.h` | `mlp.dense_h_to_4h` | `mlp.dense_4h_to_h` | — | `self_attention.{query_key_value,dense}` |
| GPT-2 | `model.transformer.h` | `mlp.c_fc` | `mlp.c_proj` | — | `attn.c_attn` (combined QKV) |
| BERT | `model.bert.encoder.layer` | `intermediate.dense` | `output.dense` | — | `attention.self.{query,key,value}` |
| Gemma | `model.model.layers` | `mlp.up_proj` | `mlp.down_proj` | `mlp.gate_proj` | `self_attn.{q,k,v,o}_proj` |
| Pixtral (vision) | `model.model.vision_tower.transformer.layers` | `feed_forward.up_proj` | `feed_forward.down_proj` | `feed_forward.gate_proj` | `attention.{q,k,v,o}_proj` |

## Reference Implementations

See `references/mesh_config_patterns.md` for complete `get_mesh_config` and `load_shard_spec` examples extracted from production loaders (Llama, Qwen, Mistral).
