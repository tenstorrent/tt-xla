---
name: tt-multi-chip
description: Add multi-chip tensor parallelism support to tt-forge-models loaders. Covers PyTorch (get_mesh_config / load_shard_spec) and JAX (get_input_activations_partition_spec / load_parameters_partition_spec) approaches, mesh topology, sharding, and validation.
argument-hint: <model-name-or-loader-path>
---

# Multi-Chip Model Support — tt-forge-models

Guides the implementation of multi-chip partitioning in model loaders to enable tensor and
data parallelism across multiple Tenstorrent devices.

Two framework-specific APIs exist:

| Framework | Methods to implement |
|-----------|---------------------|
| **PyTorch** | `get_mesh_config()` + `load_shard_spec()` |
| **JAX / EasyDeL** | `get_input_activations_partition_spec()` + `load_parameters_partition_spec()` |

## External Resources

- [tt-forge-models base.py](third_party/tt_forge_models/base.py) — Default stubs
- [PyTorch XLA SPMD](https://pytorch.org/xla/release/2.5/index.html#pytorch-xla-spmd) — Underlying sharding API (PyTorch path)
- [JAX sharding](https://jax.readthedocs.io/en/latest/jax.sharding.html) — JAX `PartitionSpec` / `Mesh` API

## Overview

Multi-chip splits model weights and/or activations across devices so large models fit in memory
and run in parallel.

**PyTorch path** (two methods):
1. **`get_mesh_config(num_devices)`** — Define how devices are arranged in a 2D mesh
2. **`load_shard_spec(model, ...)`** — Define how each weight tensor is partitioned across that mesh (`ForgeModel` stubs `load_shard_spec(self, model)`; production loaders often add **`strategy`** / **`batch_axis`** — see Step 2)

The test runner constructs a PyTorch XLA mesh device and applies sharding via
`xs.mark_sharding(tensor, mesh, shard_spec)`.

**JAX path** (two methods):
1. **`get_input_activations_partition_spec(mesh, parallelism, axis_name)`** — Define how inputs are partitioned
2. **`load_parameters_partition_spec(model, parallelism, axis_name, ...)`** — Define how model parameters are partitioned

The test runner constructs a JAX mesh and applies the partition specs when distributing the
model and inputs across devices.

**JAX EasyDel single-device:** `test_all_models_jax` routes EasyDel loaders through **`DynamicJaxMultiChipModelTester`** even when **`num_devices == 1`**. Implement `get_input_activations_partition_spec` / `load_parameters_partition_spec` so replicated partitioning on a one-device mesh is correct, not only for TP/DP.

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

### Base vs production signature

`third_party/tt_forge_models/base.py` declares `load_shard_spec(self, model)`. Production loaders (e.g. Llama) often **extend** the signature:

```python
def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
    """strategy: 'fsdp' uses both mesh axes; 'megatron' shards on the model axis only.
    batch_axis: label for the non-model axis — use 'data' when DP input sharding uses a 'data' mesh name."""
```

Preserve these parameters when porting from a reference loader; they keep FSDP/Megatron modes and DP axis naming consistent with `tests/runner/utils/dynamic_loader.py` (`load_shard_spec_data_parallel` uses `"data"` on inputs).

### Return value

```python
def load_shard_spec(self, model, ...) -> Optional[dict]:
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

### Standard Transformers (Dense MLP)

| Architecture | Layers | MLP Up | MLP Down | MLP Gate | Q/K/V/O Proj |
|-------------|--------|--------|----------|----------|-------------|
| Llama, Qwen, Mistral | `model.model.layers` | `mlp.up_proj` | `mlp.down_proj` | `mlp.gate_proj` | `self_attn.{q,k,v,o}_proj` |
| Gemma | `model.model.layers` | `mlp.up_proj` | `mlp.down_proj` | `mlp.gate_proj` | `self_attn.{q,k,v,o}_proj` |
| Falcon | `model.transformer.h` | `mlp.dense_h_to_4h` | `mlp.dense_4h_to_h` | — | `self_attention.{query_key_value,dense}` |
| GPT-2 | `model.transformer.h` | `mlp.c_fc` | `mlp.c_proj` | — | `attn.c_attn` (combined QKV) |
| BERT | `model.bert.encoder.layer` | `intermediate.dense` | `output.dense` | — | `attention.self.{query,key,value}` |

### Combined / Fused Projection Architectures

| Architecture | Layers | MLP | Attention |
|-------------|--------|-----|-----------|
| Phi-4 | `model.model.layers` | `mlp.gate_up_proj` (fused) + `mlp.down_proj` | `self_attn.qkv_proj` (fused QKV) + `self_attn.o_proj` |
| OLM-OCR (vision) | `model.model.visual.blocks` | variant-specific: `mlp.fc1`/`fc2` or `mlp.gate_proj`/`up_proj`/`down_proj` | `attn.qkv` (fused QKV) + `attn.proj` |

### Multimodal Models (Vision + Language)

| Architecture | Vision Layers | Language Layers |
|-------------|---------------|-----------------|
| Pixtral (Mistral3) | `model.model.vision_tower.transformer.layers` — `feed_forward.{up,gate,down}_proj`, `attention.{q,k,v,o}_proj` | `model.model.language_model.layers` — standard Mistral paths |
| Gemma3 | `model.vision_tower.vision_model.encoder.layers` — `self_attn.{q,k,v}_proj`/`out_proj`, `mlp.fc1`/`fc2` | `model.language_model.layers` — standard Gemma paths |
| OLM-OCR | `model.model.visual.blocks` — `attn.qkv`/`attn.proj`, variant MLPs | `model.model.language_model.layers` — standard Qwen paths |

### MoE (Mixture of Experts) Models

| Architecture | Layers | Router | Expert MLPs | Attention |
|-------------|--------|--------|-------------|-----------|
| GPT-OSS | `model.model.layers` | `mlp.router.weight` — replicated `(None, "batch")` | `mlp.experts.gate_up_proj` 3D `("model", "batch", None)`, `mlp.experts.down_proj` 3D `("model", None, "batch")` | `self_attn.{q,k,v}_proj` + biases |

**Note:** Vision-only models (ResNet, ViT, DETR, YOLO, etc.) do not implement multi-chip — single device only. Multi-chip is primarily for LLMs and multimodal models that exceed single-chip memory.

See `references/mesh_config_patterns.md` for generic `load_shard_spec` patterns for each type (standard, fused QKV, multimodal, MoE).

## Data Parallel Note

**PyTorch DP:** the tester bypasses `get_mesh_config` — uses fixed `(1, N), ("model", "data")` mesh and `load_shard_spec_data_parallel`. Loader authors only need `get_mesh_config`/`load_shard_spec` for **tensor parallelism**.

**JAX DP:** handled explicitly via `parallelism` parameter — inputs sharded on batch, parameters replicated.

---

# JAX / EasyDeL Multi-Chip Support

JAX models use a different API from PyTorch. Instead of `get_mesh_config` / `load_shard_spec`,
JAX loaders implement `get_input_activations_partition_spec` and
`load_parameters_partition_spec`. These work with `jax.sharding.PartitionSpec` and EasyDeL's
partition rule system.

## Prerequisites (JAX)

1. The model loader already works on single device using EasyDeL
2. `from easydel.modules.<arch> import <Arch>Config` exposes `.get_partition_rules()`
3. You know the parallelism mode: `Parallelism.DATA_PARALLEL`, `Parallelism.TENSOR_PARALLEL`, or `Parallelism.SINGLE_DEVICE`

## Step J1: Implement `get_input_activations_partition_spec`

### API Signature

```python
def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    """Returns partition specs for input activations.

    Args:
        mesh: JAX device mesh.
        parallelism: Parallelism enum (DATA_PARALLEL, TENSOR_PARALLEL, SINGLE_DEVICE).
        axis_name: Name of the mesh axis used for sharding (default "X").

    Returns:
        Tuple of PartitionSpec — one per input tensor.
    """
```

### Standard Pattern

Nearly all JAX loaders follow the same logic:

```python
from jax.sharding import PartitionSpec
import numpy as np

def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if (
        parallelism.name == Parallelism.TENSOR_PARALLEL.name
        or np.prod(list(mesh.shape.values())) == 1
    ):
        return (PartitionSpec(),)
    return (PartitionSpec(axis_name),)
```

- **Tensor parallel or single device**: inputs are replicated (no sharding) → `PartitionSpec()`
- **Data parallel**: inputs are sharded along the batch dimension → `PartitionSpec(axis_name)`

**Encoder-decoder models** (e.g., Whisper) return multiple specs — one per input:
```python
def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if parallelism.name == Parallelism.TENSOR_PARALLEL.name or ...:
        return (PartitionSpec(), PartitionSpec())
    return (PartitionSpec(axis_name), PartitionSpec(axis_name))
```

## Step J2: Implement `load_parameters_partition_spec`

### API Signature

```python
def load_parameters_partition_spec(
    self,
    model_for_multichip,
    parallelism,
    axis_name="X",
    cpu_mesh=None,
    input_activations_partition_specs=None,
    inputs=None,
    dtype_override=None,
):
    """Returns partition specs for all model parameters.

    Args:
        model_for_multichip: The loaded EasyDeL model.
        parallelism: Parallelism enum.
        axis_name: Name of the mesh axis for sharding.
        cpu_mesh: Optional CPU mesh (used for Flax Linen models).
        input_activations_partition_specs: Specs from get_input_activations_partition_spec.
        inputs: Sample inputs (used for Flax Linen parameter initialization).
        dtype_override: Optional dtype override.

    Returns:
        Parameter partition specs (format depends on backend).
    """
```

### Standard EasyDeL Pattern

This is the most common pattern (used by Llama, Qwen, Phi, Falcon, Mistral, Mamba, GPT-2, GPT-J):

```python
import flax.nnx as nnx
from jax.sharding import PartitionSpec

def load_parameters_partition_spec(
    self, model_for_multichip, parallelism, axis_name="X",
    cpu_mesh=None, input_activations_partition_specs=None,
    inputs=None, dtype_override=None,
):
    state = nnx.split(model_for_multichip)[1]

    if (
        parallelism.name == Parallelism.DATA_PARALLEL.name
        or parallelism.name == Parallelism.SINGLE_DEVICE.name
    ):
        partition_rules = ((r".*", PartitionSpec()),)
    else:
        from easydel.modules.<arch> import <Arch>Config
        arch_config = <Arch>Config()
        partition_rules = arch_config.get_partition_rules()

    from infra.utilities import make_easydel_parameters_partition_specs
    return make_easydel_parameters_partition_specs(
        model_state=state, partition_rules=partition_rules, axis_name=axis_name
    )
```

### Key pieces:

1. **`nnx.split(model)[1]`** — Extracts the model state (parameters) from the Flax NNX model
2. **Data parallel / single device** — All parameters are replicated: `((r".*", PartitionSpec()),)`
3. **Tensor parallel** — Use the architecture-specific partition rules from EasyDeL's config class
4. **`make_easydel_parameters_partition_specs`** — Infrastructure utility that applies regex-based partition rules to the model state tree

### EasyDeL Config → Partition Rules Mapping

Each EasyDeL model architecture has a config class with `get_partition_rules()`. Only **17** of the 52 JAX loaders use EasyDeL; the remaining 35 use HuggingFace Flax or custom Flax Linen (see "Non-EasyDeL JAX Loaders" below).

| Model | Config Import | Config Class |
|-------|--------------|-------------|
| Llama | `easydel.modules.llama` | `LlamaConfig` |
| Qwen 2.5 | `easydel.modules.qwen2` | `Qwen2Config` |
| Qwen 2.5-Coder | `easydel.modules.qwen2` | `Qwen2Config` |
| Qwen 3 | `easydel.modules.qwen3` | `Qwen3Config` |
| Phi 1/1.5/2 | `easydel.modules.phi` | `PhiConfig` |
| Phi 3 | `easydel.modules.phi3` | `Phi3Config` |
| Falcon | `easydel.modules.falcon` | `FalconConfig` |
| Mistral | `easydel.modules.mistral` | `MistralConfig` |
| Mamba | `easydel.modules.mamba` | `MambaConfig` |
| Mamba2 | `easydel.modules.mamba2` | `Mamba2Config` |
| GPT-2 | `easydel.modules.gpt2` | `GPT2Config` |
| GPT-J | `easydel.modules.gpt_j` | `GPTJConfig` |
| GLM | `easydel.modules.glm` | `GLMConfig` |
| StableLM | `easydel.modules.stablelm` | `StableLmConfig` |
| Whisper | `easydel.modules.whisper` | `WhisperConfig` |

### Non-EasyDeL JAX Loaders (HuggingFace Flax / Custom Flax Linen)

35 of 52 JAX loaders do **not** use EasyDeL. These fall into two categories:

**HuggingFace Flax models** (ResNet, ViT, CLIP, DINOv2, BERT, BART, T5, etc.):
- Loaded via `Flax<Model>ForXxx.from_pretrained()`
- Currently **no multi-chip partition methods** — single device only
- Use `cast_hf_model_to_type` from `tools/jax_utils` for dtype casting
- Return dict/BatchEncoding inputs from HF processors

**Custom Flax Linen models** (MNIST, AlexNet):
- Use `make_flax_linen_parameters_partition_specs_on_cpu` and `initialize_flax_linen_parameters_on_cpu` from `infra.utilities`
- Multi-chip variants use `load_multichip_model` (separate from `load_model`)
- May return raw JAX arrays (not dicts) from `load_inputs`

When adding multi-chip to a **non-EasyDeL** JAX loader, use the Flax Linen pattern below.

### Flax Linen Pattern (Non-EasyDeL)

For models using Flax Linen directly (MNIST, AlexNet, custom architectures), use
the infrastructure utilities for parameter initialization and partitioning:

```python
def load_parameters_partition_spec(
    self, model_for_multichip, parallelism, axis_name="X",
    cpu_mesh=None, input_activations_partition_specs=None,
    inputs=None, dtype_override=None,
):
    from infra.utilities import (
        make_flax_linen_parameters_partition_specs_on_cpu,
        initialize_flax_linen_parameters_on_cpu,
    )
    init_params = initialize_flax_linen_parameters_on_cpu(
        model=model_for_multichip, cpu_mesh=cpu_mesh, inputs=inputs
    )
    return make_flax_linen_parameters_partition_specs_on_cpu(
        params=init_params, partition_rules=((r".*", PartitionSpec()),)
    )
```

## Step J3: Validate JAX Multi-Chip

### Test Config Registration

Register in the JAX-specific **inference** config files with `EXPECTED_PASSING` so real
errors are visible (not hidden by xfail). Only downgrade to `KNOWN_FAILURE_XFAIL` after
observing and diagnosing a specific failure:

```yaml
# tests/runner/test_config/jax/test_config_inference_tensor_parallel.yaml
test_config:
  llama/causal_lm/jax-3B_v2-tensor_parallel-inference:
    status: EXPECTED_PASSING
```

```yaml
# tests/runner/test_config/jax/test_config_inference_data_parallel.yaml
test_config:
  llama/causal_lm/jax-3B_v2-data_parallel-inference:
    status: EXPECTED_PASSING
```

### Run Multi-Chip Tests

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_jax[llama/causal_lm/jax-3B_v2-tensor_parallel-inference]"
pytest -svv "tests/runner/test_models.py::test_all_models_jax[llama/causal_lm/jax-3B_v2-data_parallel-inference]"
```

### Tester Invocation Details

**Important:** `DynamicJaxMultiChipModelTester` calls these methods with specific signatures that may differ from what you'd expect:

- `get_input_activations_partition_spec(mesh, axis_name=..., parallelism=...)` — all keyword args after mesh
- `load_parameters_partition_spec(model, cpu_mesh=..., input_activations_partition_specs=..., inputs=..., parallelism=...)` — note: **`axis_name` is not passed** by the tester; implementations must use the default value `"X"`
- `load_multichip_model(axis_name=..., num_devices=..., train_mode=...)` — called when defined on the loader, instead of `load_model`

### Common JAX Multi-Chip Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `PartitionSpec axis name not found in mesh` | `axis_name` doesn't match the mesh axis | Ensure mesh axis names match what's passed to partition specs |
| `make_easydel_parameters_partition_specs failed` | Partition rules regex doesn't match parameter names | Print `state` tree to verify parameter name patterns |
| `Batch size not divisible by device count` | `load_inputs` doesn't handle `mesh` parameter | Add mesh-aware batch size calculation in `load_inputs` |
| `nnx.split returned unexpected structure` | Model wasn't loaded via EasyDeL properly | Verify `AutoEasyDeLModelForCausalLM.from_pretrained` succeeded |

## JAX vs PyTorch Multi-Chip Comparison

| Aspect | PyTorch | JAX / EasyDeL |
|--------|---------|---------------|
| Device mesh | `xs.Mesh` from PyTorch XLA | `jax.sharding.Mesh` |
| Methods to implement | `get_mesh_config` + `load_shard_spec` | `get_input_activations_partition_spec` + `load_parameters_partition_spec` |
| Sharding granularity | Per-tensor `{param: tuple}` mapping | Regex-based partition rules applied to full state tree |
| Parallelism control | Implicit via mesh shape | Explicit `Parallelism` enum argument |
| Model state extraction | Direct `model.named_parameters()` | `nnx.split(model)[1]` for Flax NNX |
| Config source | `model.config` from HuggingFace | `easydel.modules.<arch>.<Arch>Config` |
| Infrastructure utility | `xs.mark_sharding()` | `make_easydel_parameters_partition_specs()` |

---

## Reference Implementations

See `references/mesh_config_patterns.md` for complete PyTorch `get_mesh_config` and `load_shard_spec` examples extracted from production loaders (Llama, Qwen, Mistral).

See `references/jax_partition_patterns.md` for complete JAX `get_input_activations_partition_spec` and `load_parameters_partition_spec` examples from production EasyDeL loaders.

YAML test keys must use the full **`rel_path`** to the loader (e.g. `llama/causal_lm/jax-...`), not a shortened `<model>/jax-...` alias — see `../tt-model-bringup/references/test_ids_and_yaml.md`.
