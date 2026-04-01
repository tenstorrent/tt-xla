# Multi-Chip Quick-Start Guide

Minimal steps to add multi-chip support to an existing ModelLoader.
Covers both PyTorch (`get_mesh_config` / `load_shard_spec`) and
JAX (`get_input_activations_partition_spec` / `load_parameters_partition_spec`).

## Step 1: Inspect Your Model

```python
from third_party.tt_forge_models.<model>.pytorch import ModelLoader

loader = ModelLoader()
model = loader.load_model()

# Print all 2D weight parameters to find layer structure
for name, param in model.named_parameters():
    if param.dim() >= 2:
        print(f"{name}: shape={list(param.shape)}")
```

Look for these patterns in the output:
- `model.model.layers.0.self_attn.q_proj.weight` → Standard transformer
- `model.transformer.h.0.attn.c_attn.weight` → GPT-2 style (combined QKV)
- `model.bert.encoder.layer.0.attention.self.query.weight` → BERT style

## Step 2: Check Attention Heads

```python
config = model.config
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
print(f"hidden_size: {config.hidden_size}")
```

For multi-chip to work: `num_attention_heads % model_axis_size == 0`

## Step 3: Add get_mesh_config

### Minimal Version (most models)

```python
def get_mesh_config(self, num_devices: int):
    assert self.config.num_attention_heads % num_devices == 0, \
        f"{self.config.num_attention_heads} heads not divisible by {num_devices}"
    return (1, num_devices), ("batch", "model")
```

### Robust Version (with fallback)

```python
def get_mesh_config(self, num_devices: int):
    heads = self.config.num_attention_heads
    if num_devices == 32:  # Galaxy
        mesh_shape = (4, 8) if heads % 8 == 0 else (8, 4)
    elif heads % num_devices == 0:
        mesh_shape = (1, num_devices)
    elif num_devices % 2 == 0 and heads % (num_devices // 2) == 0:
        mesh_shape = (2, num_devices // 2)
    else:
        raise ValueError(f"Cannot split {heads} heads across {num_devices} devices")
    return mesh_shape, ("batch", "model")
```

## Step 4: Add load_shard_spec

### Standard Transformer (Llama-like architecture)

```python
def load_shard_spec(self, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    shard_specs[model.lm_head.weight] = ("model", "batch")
    return shard_specs
```

### With Biases (Qwen-like)

```python
def load_shard_spec(self, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    shard_specs[model.lm_head.weight] = ("model", "batch")
    return shard_specs
```

### With Small Variant Guard

```python
def load_shard_spec(self, model):
    if self._variant in [ModelVariant.SMALL]:
        return None
    # ... specs for larger variants
```

## Step 5: Ensure load_config Exists

Both `get_mesh_config` and `load_shard_spec` often need `self.config`. Make sure your
loader has `load_config` and stores it:

```python
def load_config(self):
    self.config = AutoConfig.from_pretrained(
        self._variant_config.pretrained_model_name
    )
    return self.config
```

And call it in `load_model`:

```python
def load_model(self, *, dtype_override=None, **kwargs):
    # ... load model ...
    self.config = model.config  # Cache config after loading
    return model
```

## How the Test Runner Uses These

The DynamicModelLoader in `tests/runner/utils/dynamic_loader.py` wraps these:

```python
# From the test runner
num_devices = xr.global_runtime_device_count()
mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
mesh = xs.Mesh(range(num_devices), mesh_shape, mesh_names)

model = loader.load_model()
shard_specs = loader.load_shard_spec(model)
if shard_specs:
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
```

Both methods are completely optional — if not implemented, the base class returns
`(None, ())` and `None` respectively, and the model runs on a single device.

---

# JAX / EasyDeL Multi-Chip Quick-Start

## Step 1: Verify EasyDeL Config Exists

```python
from easydel.modules.<arch> import <Arch>Config
config = <Arch>Config()
rules = config.get_partition_rules()
print(rules)
```

If `get_partition_rules()` exists, you can use the standard pattern.
If not, use the trivial replicated rules: `((r".*", PartitionSpec()),)`.

## Step 2: Add Imports to Your Loader

```python
import flax.nnx as nnx
from jax.sharding import PartitionSpec
import numpy as np
from ....config import Parallelism
```

## Step 3: Add get_input_activations_partition_spec

```python
def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
    if (
        parallelism.name == Parallelism.TENSOR_PARALLEL.name
        or np.prod(list(mesh.shape.values())) == 1
    ):
        return (PartitionSpec(),)
    return (PartitionSpec(axis_name),)
```

For encoder-decoder models (multiple inputs), return a tuple with one spec per input.

## Step 4: Add load_parameters_partition_spec

```python
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

## Step 5: Update load_inputs for Mesh-Aware Batch Size

Ensure `load_inputs` accepts `mesh=None` and adjusts batch size:

```python
def load_inputs(self, dtype_override=None, mesh=None):
    if mesh is not None:
        num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
        batch_size = 8
        if batch_size % num_devices != 0:
            batch_size = num_devices * (batch_size // num_devices + 1)
    else:
        batch_size = 8

    # ... tokenize and repeat to batch_size ...
```

## How the Test Runner Uses These (JAX)

The test infrastructure calls these methods to distribute the model:

1. Creates a mesh from available devices
2. Calls `get_input_activations_partition_spec(mesh, parallelism)` for input sharding
3. Calls `load_parameters_partition_spec(model, parallelism)` for parameter sharding
4. Applies partition specs and runs inference

Both methods are completely optional — if not implemented, the model runs on a single device.
