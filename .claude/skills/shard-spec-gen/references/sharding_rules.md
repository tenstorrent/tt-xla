# Megatron Tensor Parallel Sharding Rules

Weight shape is `[out_features, in_features]`. Mesh axis name is typically `"model"`.

## Attention (standard q/k/v/o)
```python
shard_specs[attention.q_proj.weight] = ("model", None)   # column-parallel
shard_specs[attention.k_proj.weight] = ("model", None)
shard_specs[attention.v_proj.weight] = ("model", None)
shard_specs[attention.o_proj.weight] = (None, "model")   # row-parallel
```

## MLP (standard gate/up/down)
```python
shard_specs[mlp.gate_proj.weight] = ("model", None)
shard_specs[mlp.up_proj.weight]   = ("model", None)
shard_specs[mlp.down_proj.weight] = (None, "model")
```

## MoE (iterate over experts; handle shared_expert if present)
```python
for expert in mlp.experts:
    shard_specs[expert.gate_proj.weight] = ("model", None)
    shard_specs[expert.up_proj.weight]   = ("model", None)
    shard_specs[expert.down_proj.weight] = (None, "model")
if hasattr(mlp, "shared_expert") and mlp.shared_expert is not None:
    shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", None)
    shard_specs[mlp.shared_expert.up_proj.weight]   = ("model", None)
    shard_specs[mlp.shared_expert.down_proj.weight] = (None, "model")
```

## Fallback: heads not divisible by num_devices
Shard inputs on the batch axis instead:
```python
shard_specs[args[0]] = ("batch", None, None)   # hidden_states
# plus all weight specs above
```

## Notes
- Use exact attribute names from Step 4 — don't assume `q_proj`, `gate_proj`, etc.
- `decoder_layer` = apply both attention and MLP rules to the same layer
