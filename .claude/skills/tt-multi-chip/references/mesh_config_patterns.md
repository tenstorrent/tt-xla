# Multi-Chip Reference Patterns

Generic patterns for `get_mesh_config` and `load_shard_spec`. Always run
`model.named_parameters()` first to discover actual layer paths for your model.

---

## get_mesh_config — Generic (covers all models)

```python
def get_mesh_config(self, num_devices: int):
    heads = self.config.num_attention_heads  # or self.config.text_config.num_attention_heads for multimodal
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

---

## load_shard_spec — By Model Type

### Type 1: Standard Transformer (Llama, Qwen, Mistral, Gemma, etc.)

Covers models with separate Q/K/V/O projections and gate/up/down MLP. If the model has
biases (e.g. Qwen), add 1D specs like `("model",)` for each bias. Production loaders
may add `strategy="fsdp"` / `batch_axis="batch"` parameters for FSDP vs Megatron modes.

```python
def load_shard_spec(self, model):
    if self._variant in [ModelVariant.SMALL]:
        return None
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        # If model has biases: shard_specs[layer.self_attn.q_proj.bias] = ("model",)
    shard_specs[model.lm_head.weight] = ("model", "batch")
    return shard_specs
```

### Type 2: Fused Projections (e.g. combined QKV or gate+up)

Same approach — the fused weight is just sharded as a single tensor:

```python
shard_specs[layer.self_attn.qkv_proj.weight] = ("model", "batch")   # fused QKV
shard_specs[layer.mlp.gate_up_proj.weight] = ("model", "batch")     # fused gate+up
```

### Type 3: Multimodal (Vision + Language)

Shard both components. Vision encoders often use different naming (`out_proj`, `fc1`/`fc2`,
`attn.qkv`). Discover with `named_parameters()`.

```python
def load_shard_spec(self, model):
    if self._variant != ModelVariant.LARGE:
        return None
    shard_specs = {}

    # Vision encoder (paths vary: vision_tower, model.visual, etc.)
    for layer in model.vision_tower.vision_model.encoder.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.out_proj.weight] = ("batch", "model")
        shard_specs[layer.mlp.fc1.weight] = ("model", "batch")
        shard_specs[layer.mlp.fc2.weight] = ("batch", "model")

    # Language model
    for layer in model.language_model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    return shard_specs
```

### Type 4: MoE (Mixture of Experts)

Expert weights are 3D `[num_experts, hidden, intermediate]` — shard expert dim on `"model"`.
Router is replicated. `num_experts % model_axis_size == 0` must hold.

```python
def load_shard_spec(self, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.mlp.router.weight] = (None, "batch")
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
        # Attention: standard 2D sharding (same as Type 1)
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    return shard_specs
```

---

## Cheat Sheet

1. **Discover layers:** `for name, p in model.named_parameters(): print(f"{name}: {list(p.shape)}")`
2. **Sharding rule:** output projections (Q, K, V, gate, up, fused QKV) → `("model", ...)`, input projections (O, down) → `(..., "model")`, biases → `("model",)`, MoE experts dim 0 → `"model"`
3. **Head divisibility:** `num_attention_heads % model_axis_size == 0`
4. **Small variant guard:** return `None` if model fits on one device
5. **Multimodal:** shard both towers — vision and language use different naming
