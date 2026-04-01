# Multi-Chip Reference Implementations

Production-tested patterns from real model loaders in `tt-forge-models`.

---

## Llama (3.x family)

Llama supports both FSDP and Megatron strategies, with special handling for 70B/405B
models that need more batch parallelism on Galaxy.

### get_mesh_config

```python
def get_mesh_config(self, num_devices: int):
    if self._variant in [
        ModelVariant.LLAMA_3_1_70B,
        ModelVariant.LLAMA_3_1_70B_INSTRUCT,
        ModelVariant.LLAMA_3_3_70B_INSTRUCT,
        ModelVariant.LLAMA_3_1_405B,
        ModelVariant.LLAMA_3_1_405B_INSTRUCT,
    ]:
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        else:  # wh/bh llmbox
            mesh_shape = (2, num_devices // 2)
    else:
        mesh_shape = (1, num_devices)

    return mesh_shape, ("batch", "model")
```

**Key decisions:**
- 70B/405B use `(4, 8)` on Galaxy because they have 64/128 attention heads, which divide by 8
- Smaller models (8B, 3B) use `(1, N)` — all devices for tensor parallelism

### load_shard_spec (FSDP strategy)

```python
def load_shard_spec(self, model, strategy="fsdp", batch_axis="batch"):
    # Small models don't need sharding
    if self._variant in [
        ModelVariant.LLAMA_3_2_1B,
        ModelVariant.LLAMA_3_2_1B_INSTRUCT,
        ModelVariant.LLAMA_3_2_3B,
        ModelVariant.LLAMA_3_2_3B_INSTRUCT,
        ModelVariant.HUGGYLLAMA_7B,
    ]:
        return None

    shard_specs = {}

    if strategy == "fsdp":
        shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
        shard_specs[model.lm_head.weight] = ("model", batch_axis)
        shard_specs[model.model.norm.weight] = (batch_axis,)
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.gate_proj.weight] = ("model", batch_axis)
            shard_specs[layer.mlp.down_proj.weight] = (batch_axis, "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
            shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
            shard_specs[layer.input_layernorm.weight] = (batch_axis,)
            shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

    elif strategy == "megatron":
        shard_specs[model.model.embed_tokens.weight] = (None, None)
        shard_specs[model.lm_head.weight] = ("model", None)
        shard_specs[model.model.norm.weight] = (None,)
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", None)
            shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
            shard_specs[layer.mlp.down_proj.weight] = (None, "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
            shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
            shard_specs[layer.input_layernorm.weight] = (None,)
            shard_specs[layer.post_attention_layernorm.weight] = (None,)

    return shard_specs
```

**Key decisions:**
- FSDP shards embeddings and layernorms on the batch axis for maximal memory distribution
- Megatron leaves batch axis as None (replicated) — simpler but less memory efficient
- `batch_axis` parameter allows swapping to `"data"` when combined with data-parallel input sharding

---

## Qwen 2.5

Qwen uses the standard transformer pattern but includes biases on attention projections.

### get_mesh_config

```python
def get_mesh_config(self, num_devices: int):
    if num_devices == 32:  # Galaxy
        mesh_shape = (8, 4)
    elif self.config.num_attention_heads % num_devices == 0:
        mesh_shape = (1, num_devices)
    elif (
        self.config.num_attention_heads % (num_devices // 2) == 0
        and num_devices % 2 == 0
    ):
        mesh_shape = (2, num_devices // 2)
    else:
        raise ValueError(
            f"Cannot evenly distribute {self.config.num_attention_heads} "
            f"heads across {num_devices} devices"
        )
    return mesh_shape, ("batch", "model")
```

**Key decision:** Galaxy uses `(8, 4)` rather than `(4, 8)` — this is model-specific based on how many attention heads divide evenly.

### load_shard_spec

```python
def load_shard_spec(self, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)       # Qwen has biases
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    shard_specs[model.lm_head.weight] = ("model", "batch")

    return shard_specs
```

**Key difference from Llama:** Qwen includes 1D bias specs `("model",)` for Q/K/V biases.

---

## Mistral

Mistral shows how to handle models with conditional architecture (regular vs. multimodal Pixtral).

### get_mesh_config

```python
def get_mesh_config(self, num_devices: int):
    mesh_shape = (1, num_devices)
    if self._variant not in [ModelVariant.MINISTRAL_3B]:
        assert (
            self.config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
    return mesh_shape, ("batch", "model")
```

**Key decision:** Small variants (3B) are excluded from the assertion because they may not need/support multi-chip.

### load_shard_spec

```python
def load_shard_spec(self, model):
    if self._variant in [ModelVariant.MINISTRAL_3B]:
        return None

    shard_specs = {}

    if self._variant in self._USE_Mistral3ForConditionalGeneration_VARIANTS:
        # Multimodal Pixtral: shard both vision tower and language model
        for layer in model.model.vision_tower.transformer.layers:
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")
            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        for layer in model.model.language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    else:
        # Standard Mistral: same pattern as Llama
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    return shard_specs
```

**Key difference:** Multimodal models need to shard both the vision tower and language model separately, with different layer paths.

---

## Cheat Sheet: Adding Multi-Chip to a New Model

1. **Load the model and print parameter names:**
   ```python
   for name, p in model.named_parameters():
       if "weight" in name and p.dim() == 2:
           print(f"{name}: {p.shape}")
   ```

2. **Identify the transformer layers list.** Look for patterns like:
   - `model.model.layers` (Llama, Qwen, Mistral, Gemma)
   - `model.transformer.h` (GPT-2, Falcon)
   - `model.bert.encoder.layer` (BERT)

3. **Map projections to shard specs** using the rule:
   - Output projections (Q, K, V, gate, up): `("model", "batch")`
   - Input projections (O, down): `("batch", "model")`
   - 1D biases: `("model",)` if on a model-sharded dim

4. **Check attention head divisibility** and implement `get_mesh_config` with fallback mesh shapes.

5. **Guard small variants** by returning `None` from `load_shard_spec`.
