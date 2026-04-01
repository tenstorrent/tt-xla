# Shard Spec Quick-Start Guide

Minimal steps to add `get_mesh_config` and `load_shard_spec` to an existing ModelLoader.

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
