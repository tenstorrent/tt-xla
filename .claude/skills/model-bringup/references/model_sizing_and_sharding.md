# Model Sizing and Sharding Strategy

During bringup, you must determine whether a model fits on a single device or needs to be sharded across multiple devices. This reference covers device memory budgets, model size estimation, and sharding strategy for tt-forge-models loaders.

## Device memory budgets

| Architecture | DRAM per chip | Typical use |
|-------------|---------------|-------------|
| Wormhole (WH) — n150, p150, n300 | **12 GB** | Most CV models, small LLMs (1B-8B single chip) |
| Blackhole (BH) | **32 GB** | Larger LLMs, multi-modal models |

**Important**: These are total DRAM budgets. Actual available memory is less due to runtime overhead, activations, and intermediate tensors. As a rough guide, assume ~70-80% of DRAM is usable for model weights.

## Estimating model size

Calculate total model parameters and weight memory:

```python
model = loader.load_model()
total_params = sum(p.numel() for p in model.parameters())
# Size in bytes depends on dtype: float32=4B, bfloat16=2B, bfp8=1B
size_f32_gb = (total_params * 4) / (1024**3)
size_bf16_gb = (total_params * 2) / (1024**3)
size_bfp8_gb = (total_params * 1) / (1024**3)
print(f"Parameters: {total_params:,}")
print(f"Size: {size_f32_gb:.1f} GB (f32), {size_bf16_gb:.1f} GB (bf16), {size_bfp8_gb:.1f} GB (bfp8)")
```

### Quick reference for common model sizes

| Model | Parameters | Size (bf16) | Fits on WH? | Fits on BH? |
|-------|-----------|-------------|-------------|-------------|
| ResNet-50 | 25M | 0.05 GB | Yes | Yes |
| BERT-base | 110M | 0.2 GB | Yes | Yes |
| Llama 3.2 1B | 1.2B | 2.4 GB | Yes | Yes |
| Llama 3.2 3B | 3.2B | 6.4 GB | Tight | Yes |
| Llama 3.1 8B | 8B | 16 GB | No (needs TP) | Yes |
| Llama 3.1 70B | 70B | 140 GB | No (needs TP) | No (needs TP) |

## Decision tree: single device vs. tensor parallel

```
1. Calculate model_size_gb (in target dtype, typically bf16)
2. Compare to device DRAM:
   - model_size_gb < 0.7 * device_dram_gb → single_device
   - model_size_gb >= 0.7 * device_dram_gb → needs tensor_parallel or bfp8
3. If borderline:
   - Try enable_weight_bfp8_conversion first (halves weight memory vs bf16)
   - If still too large → tensor_parallel
4. For tensor_parallel:
   - num_chips_needed = ceil(model_size_gb / (0.7 * device_dram_gb))
   - Round up to available mesh size (2, 4, 8, 32)
```

## Sharding strategy: minimize CCLs

**Key principle**: Collective communication (CCLs — all-reduce, all-gather) is expensive. Shard only what's necessary to fit in memory. If you have replicated bandwidth (i.e., the model fits with fewer shards than available chips), use the extra chips to increase batch size instead.

### What to shard

For a standard transformer LLM, the large weight matrices are:
- **MLP**: `up_proj`, `gate_proj`, `down_proj` (or `dense_h_to_4h`, `dense_4h_to_h`)
- **Attention**: `q_proj`, `k_proj`, `v_proj`, `o_proj`

These are the only tensors worth sharding — they dominate parameter count.

### What NOT to shard (or replicate)

- **Embeddings** (`embed_tokens`) — typically replicated (`(None, None)`) unless the vocabulary is extremely large
- **Layer norms** (`input_layernorm`, `post_attention_layernorm`, `norm`) — tiny tensors, replicate them
- **Biases** — tiny, replicate

### Sharding strategies in load_shard_spec

Two strategies are used in the codebase:

**Megatron strategy** — shard only on the `"model"` axis. Simpler, fewer CCLs:
```python
# MLP: column-parallel up/gate, row-parallel down
shard_specs[layer.mlp.up_proj.weight] = ("model", None)
shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
shard_specs[layer.mlp.down_proj.weight] = (None, "model")

# Attention: column-parallel QKV, row-parallel O
shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
shard_specs[layer.self_attn.o_proj.weight] = (None, "model")

# Norms and embeddings: fully replicated
shard_specs[model.model.norm.weight] = (None,)
shard_specs[model.model.embed_tokens.weight] = (None, None)
```

**FSDP strategy** — shard across both `"batch"` and `"model"` axes. More aggressive memory saving but more CCLs:
```python
shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
# ... etc
shard_specs[model.model.norm.weight] = ("batch",)
```

**Recommendation**: Start with Megatron (simpler, fewer CCLs). Only move to FSDP if memory is still too tight.

### Mesh configuration

The mesh shape determines how devices are organized. Returned by `get_mesh_config(num_devices)`:

```python
def get_mesh_config(self, num_devices):
    # Standard: 1D mesh, all devices on model axis
    mesh_shape = (1, num_devices)
    return mesh_shape, ("batch", "model")
```

For very large models (70B+) on many devices:
```python
def get_mesh_config(self, num_devices):
    if num_devices == 32:  # Galaxy
        mesh_shape = (4, 8)
    else:
        mesh_shape = (2, num_devices // 2)
    return mesh_shape, ("batch", "model")
```

### When to increase batch instead of sharding further

If `num_chips * 0.7 * device_dram_gb > 2 * model_size_gb`, you have excess memory capacity. Instead of leaving it idle:
1. Keep the model sharding as-is
2. Increase `batch_size` in the YAML config to use the extra memory for throughput
3. This is more efficient than over-sharding (which adds CCL overhead for no benefit)

## Implementing sharding in a new loader

### Step 1: Determine if sharding is needed

```python
def load_shard_spec(self, model, strategy="megatron"):
    # Small models don't need sharding
    if self._variant in [SMALL_VARIANT_A, SMALL_VARIANT_B]:
        return None
    # ... shard spec for larger variants
```

### Step 2: Identify the weight names

Inspect the model architecture:
```python
for name, param in model.named_parameters():
    print(f"{name}: {param.shape} ({param.numel():,} params)")
```

Look for the large 2D weight matrices — those are your sharding targets.

### Step 3: Apply the standard pattern

For HuggingFace transformer models, the weight names follow a consistent pattern:
- `model.layers[i].self_attn.{q,k,v,o}_proj.weight`
- `model.layers[i].mlp.{up,gate,down}_proj.weight` (or `{dense_h_to_4h,dense_4h_to_h}`)
- `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`

### Step 4: Set supported_archs in YAML

```yaml
  model/pytorch-variant-tensor_parallel-inference:
    supported_archs: [n300-llmbox]  # or [galaxy-wh-6u] for very large models
    status: EXPECTED_PASSING
```

## Bringup sizing checklist

1. **Calculate model size** — `sum(p.numel() for p in model.parameters())`, convert to GB in target dtype
2. **Check device budget** — WH: 12GB, BH: 32GB (use 70% as effective)
3. **Decide single vs. TP** — if model fits, single_device; otherwise tensor_parallel
4. **If TP needed**: implement `get_mesh_config()` and `load_shard_spec()` in the loader
5. **Start with Megatron** — shard only on `"model"` axis, minimize CCLs
6. **If extra memory**: increase batch size rather than over-sharding
7. **If borderline**: try `enable_weight_bfp8_conversion: true` in YAML before adding TP
