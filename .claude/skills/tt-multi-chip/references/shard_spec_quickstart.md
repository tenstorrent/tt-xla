# Multi-Chip Quick-Start Guide

Minimal steps to add multi-chip support to an existing ModelLoader.

## PyTorch: Step-by-Step

### 1. Inspect your model

```python
from third_party.tt_forge_models.<path.to.pytorch.package> import ModelLoader
loader = ModelLoader()
model = loader.load_model()
for name, param in model.named_parameters():
    if param.dim() >= 2:
        print(f"{name}: shape={list(param.shape)}")
```

### 2. Check attention heads

```python
config = model.config
print(f"num_attention_heads: {config.num_attention_heads}")
print(f"num_key_value_heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
```

Constraint: `num_attention_heads % model_axis_size == 0`

### 3. Add `get_mesh_config` + `load_shard_spec`

See `mesh_config_patterns.md` for generic patterns by model type (standard transformer,
fused projections, multimodal, MoE). Pick the matching type and adapt layer paths from Step 1.

### 4. Ensure `load_config` caches `self.config`

```python
def load_config(self):
    self.config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
    return self.config
```

### How the test runner uses these (PyTorch)

```python
mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
mesh = xs.Mesh(range(num_devices), mesh_shape, mesh_names)
shard_specs = loader.load_shard_spec(model)
if shard_specs:
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
```

**Data parallel note:** `DynamicTorchModelTester` bypasses `get_mesh_config` for DP — it
uses a fixed `(1, N), ("model", "data")` mesh and `load_shard_spec_data_parallel` from
`TorchDynamicLoader`. Loader authors only need `get_mesh_config`/`load_shard_spec` for TP.

---

## JAX: Step-by-Step

### 1. Verify EasyDeL config exists (if EasyDeL model)

```python
from easydel.modules.<arch> import <Arch>Config
rules = <Arch>Config().get_partition_rules()
```

If no `get_partition_rules()`, use replicated: `((r".*", PartitionSpec()),)`.

### 2. Add partition methods

See `jax_partition_patterns.md` for the three types (EasyDeL, Flax Linen, HF Flax).

### 3. Update `load_inputs` for mesh-aware batch size

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

### How the test runner uses these (JAX)

1. Calls `get_input_activations_partition_spec(mesh, axis_name=..., parallelism=...)`
2. Calls `load_parameters_partition_spec(model, cpu_mesh=..., inputs=..., parallelism=...)` — `axis_name` is **not** passed, so implementations must default to `"X"`
3. Calls `load_multichip_model(...)` when defined, instead of `load_model`

Both methods are optional — if not implemented, the model runs on a single device.
