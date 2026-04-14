# How DeepSeek-V3 Gets Sharded with `enable_tensor_parallel=True`

## Context

This is an investigation into the full sharding flow for `deepseek-ai/DeepSeek-V3` when
`enable_tensor_parallel=True` is passed via `additional_config` in the tt-xla vLLM plugin.

---

## 1. Entry Point and Mesh Setup

**Flow:**
1. Test/user passes `additional_config={"enable_tensor_parallel": True, "use_2d_mesh": True}` to `vllm.LLM()`
2. `TTWorker.__init__()` creates `TTConfig(**vllm_config.additional_config)` and sets `use_spmd=True`
3. Worker sets `tensor_parallel_size=1`, `pipeline_parallel_size=1`, `world_size=1` (SPMD replaces explicit TP)
4. Worker calls `xr.use_spmd()` and sets `CONVERT_SHLO_TO_SHARDY=1`
5. `TTModelRunner.__init__()` creates the mesh (`model_runner.py:245-249`):
   ```python
   num_devices = xr.global_runtime_device_count()
   mesh_shape = determine_mesh_shape(num_devices, use_2d_mesh=True)
   device_ids = np.array(range(num_devices))
   self.mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
   ```

**Mesh shapes** (`vllm_utils.py:9-41`):

| Devices | 2D Mesh Shape | Target Hardware |
|---------|--------------|-----------------|
| 2       | (1, 2)       | N300            |
| 4       | (2, 2)       |                 |
| 8       | (2, 4)       |                 |
| 16      | (4, 4)       |                 |
| 32      | (4, 8)       | LLMBox          |

1D fallback: `(1, num_devices)`. Axes are always `("batch", "model")`.

---

## 2. Model Loading and Sharding Application

Order of operations (`model_runner.py:1738-1764`):
1. Load model weights on CPU (HuggingFace snapshot_download)
2. Convert FP8 parameters (`float8_e4m3fn`, `float8_e5m2`) to `bfloat16` -- DeepSeek-V3 ships fp8 weights
3. `replace_modules(model)` -- replaces `RMSNorm` with `TTRMSNorm` (`overrides.py`)
4. `model.to(self.device)` -- move to XLA device
5. Sweep stray CPU tensors (e.g. MLA derived tensors `W_UV`, `W_UK_T`) to device
6. **`shard_model(model, self.mesh)`** -- apply SPMD sharding constraints

---

## 3. How `shard_model()` Works

**File:** `vllm_distributed_utils.py:285-318`

Recursively walks the module tree via `named_children()`. For each module, checks
`module.__class__.__qualname__` (exact string match, NOT isinstance) against
`MODULE_TYPE_TO_WRAPPING_FUNC`:

| Module Type                    | Partition Function                          | Weight Partition Spec      |
|-------------------------------|---------------------------------------------|---------------------------|
| `MergedColumnParallelLinear`  | Split into sub-weights, replace module      | Each sub-weight: `("batch", "model")` |
| `QKVParallelLinear`           | Split into Q/K/V, replace module            | Each: `("batch", "model")` |
| `ColumnParallelLinear`        | `mark_sharding` on weight in-place          | `("batch", None)`         |
| `RowParallelLinear`           | `mark_sharding` on weight in-place          | `("model", "batch")`      |
| `ParallelLMHead`             | `mark_sharding` on weight in-place          | `("batch", None)`         |
| `VocabParallelEmbedding`     | `mark_sharding` + forward hook on output    | Weight: `(None, "batch")`, Output: `(None, None, None)` |

**Types NOT in the mapping** (and therefore NOT sharded):
`ReplicatedLinear`, `SharedFusedMoE`, `FusedMoE`, `RMSNorm`/`TTRMSNorm`, rotary embeddings, any plain tensor attributes.

---

## 4. DeepSeek-V3 Architecture and Per-Layer Sharding

**Model file:** `vllm/model_executor/models/deepseek_v2.py` (DeepseekV3ForCausalLM extends DeepseekV2ForCausalLM)

**Key config values:** `hidden_size=7168`, `num_heads=128`, `q_lora_rank=1536`, `kv_lora_rank=512`,
`qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, `n_routed_experts=256`,
`n_shared_experts=1`, `num_experts_per_tok=8`, `first_k_dense_replace=3`, `moe_intermediate_size=2048`,
`intermediate_size=18432`, 61 decoder layers.

### 4a. Top-Level Layers

| Component      | Module Type               | Sharded? | Partition Spec                                    |
|---------------|---------------------------|----------|--------------------------------------------------|
| `embed_tokens` | `VocabParallelEmbedding`  | YES      | Weight: `(None, "batch")`, Output hook: `(None, None, None)` |
| `lm_head`      | `ParallelLMHead`          | YES      | `("batch", None)`                                |
| `norm`         | `RMSNorm` -> `TTRMSNorm`  | NO       | Replicated                                        |

### 4b. Attention -- DeepseekV2MLAAttention (all 61 layers)

Since `q_lora_rank=1536` is not None, the **fused QKV path** is used:

| Component          | Module Type                    | Weight Shape         | Sharded? | Partition Spec                |
|-------------------|-------------------------------|---------------------|----------|-------------------------------|
| `fused_qkv_a_proj` | `MergedColumnParallelLinear`  | `(2112, 7168)` split into `(1536, 7168)` + `(576, 7168)` | **YES** | Each sub-weight: `("batch", "model")` |
| `q_a_layernorm`    | `RMSNorm` -> `TTRMSNorm`     | `(1536,)`           | NO       | Replicated                    |
| `q_b_proj`         | `ColumnParallelLinear`        | `(24576, 1536)`     | **YES** | `("batch", None)`            |
| `kv_a_layernorm`   | `RMSNorm` -> `TTRMSNorm`     | `(512,)`            | NO       | Replicated                    |
| `kv_b_proj`        | `ColumnParallelLinear`        | `(32768, 512)`      | **YES** | `("batch", None)`            |
| `o_proj`           | `RowParallelLinear`           | `(7168, 16384)`     | **YES** | `("model", "batch")`         |

**Note:** `fused_qkv_a_proj` is created with `disable_tp=True` at the vLLM level, meaning vLLM's
internal TP logic does not split this weight. However, `shard_model()` still applies SPMD sharding
because it matches on module type name, not the `disable_tp` flag.

**Note:** MLA's `process_weights_after_loading` creates derived tensors `W_UV` and `W_UK_T` as plain
attributes (not `nn.Parameter`). These are NOT visited by `shard_model()` and remain replicated.

### 4c. Dense MLP (Layers 0-2)

| Component       | Module Type                    | Weight Shape        | Sharded? | Partition Spec                  |
|----------------|-------------------------------|--------------------|---------|---------------------------------|
| `gate_up_proj`  | `MergedColumnParallelLinear`  | `(36864, 7168)` split into 2x `(18432, 7168)` | **YES** | Each: `("batch", "model")`     |
| `down_proj`     | `RowParallelLinear`           | `(7168, 18432)`    | **YES** | `("model", "batch")`           |

### 4d. MoE Layers (Layers 3-60, 58 layers)

| Component                     | Module Type                    | Weight Shape             | Sharded? | Partition Spec                  |
|------------------------------|-------------------------------|-------------------------|---------|--------------------------------|
| `gate` (router)              | `ReplicatedLinear`            | `(256, 7168)`           | **NO**  | Replicated                     |
| `gate.e_score_correction_bias` | `nn.Parameter`              | `(256,)`                | **NO**  | Replicated                     |
| `experts` (routed)           | `SharedFusedMoE`              | --                      | **NO**  | Not in mapping                 |
| `experts.w13_weight`         | Fused `nn.Parameter`          | `(256, 4096, 7168)`    | **NO**  | Replicated (~15 GB/layer)      |
| `experts.w2_weight`          | Fused `nn.Parameter`          | `(256, 7168, 2048)`    | **NO**  | Replicated (~7.5 GB/layer)     |
| `shared_experts.gate_up_proj` | `MergedColumnParallelLinear` | `(4096, 7168)` split    | **YES** | Each: `("batch", "model")`     |
| `shared_experts.down_proj`   | `RowParallelLinear`           | `(7168, 2048)`          | **YES** | `("model", "batch")`           |

### 4e. Per-Layer Norms

| Component                   | Sharded? | Notes                          |
|----------------------------|----------|--------------------------------|
| `input_layernorm`           | NO       | Replaced with TTRMSNorm        |
| `post_attention_layernorm`  | NO       | Replaced with TTRMSNorm        |

---

## 5. MLA Attention Backend

When `use_mla=True` (auto-detected for DeepSeek-V3), `platform.py:122` routes to `TTMLAAttentionBackend`:

- **KV cache shape:** 4D `(num_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim)` = `(num_blocks, 1, block_size, 576)`
- **Prefill:** `torch.ops.tt.flash_mla_prefill` custom op
- **Decode:** `torch.ops.tt.paged_flash_multi_latent_attention_decode` custom op (absorb-attention approach)
- Sparse MLA (DeepSeek-V3.2) falls back to dense MLA with a warning

---

## 6. KV Cache Sharding

**Standard attention (5D):** `model_runner.py:2329-2333` shards with `(None, None, "batch", None, None)`

**MLA attention (4D):** There is a **bug** at `model_runner.py:2332`:
```python
assert cache.ndim == 5, "KV cache tensor must be 5D."
xs.mark_sharding(cache, self.mesh, (None, None, "batch", None, None))
```
This assertion will **fail** for MLA's 4D cache shape. The recent commit `f81d84404` added MLA
cache allocation (lines 2299-2305) but did NOT update the sharding section below. MLA KV cache
sharding is currently broken when `enable_tensor_parallel=True`.

---

## 7. Summary: What IS vs IS NOT Sharded

### Sharded (distributed across devices)
- Embedding layer (`VocabParallelEmbedding`)
- LM head (`ParallelLMHead`)
- All `fused_qkv_a_proj` projections (61 layers) -- split and sharded
- All `q_b_proj`, `kv_b_proj` projections (61 layers)
- All `o_proj` projections (61 layers)
- Dense MLP `gate_up_proj` + `down_proj` (layers 0-2)
- Shared expert MLP `gate_up_proj` + `down_proj` (layers 3-60)

### NOT Sharded (fully replicated on every device)
- **All 256 routed expert weights** (`w13_weight`, `w2_weight`) across 58 MoE layers
- All MoE gate/router weights (58 layers)
- All RMSNorm/TTRMSNorm weights
- MLA derived tensors (`W_UV`, `W_UK_T`)
- MLA KV cache (due to the 5D assertion bug)

---

## 8. Memory Implications

For BFloat16 DeepSeek-V3 (~671B params):

| Category | Per-Layer Memory | Layers | Total | Sharded? |
|----------|-----------------|--------|-------|----------|
| Routed expert `w13_weight` | ~15.0 GB | 58 | ~870 GB | **NO** |
| Routed expert `w2_weight` | ~7.5 GB | 58 | ~435 GB | **NO** |
| MoE gate | ~3.5 MB | 58 | ~0.2 GB | **NO** |
| Attention projections | ~300 MB | 61 | ~18 GB | YES |
| Dense MLP | ~400 MB | 3 | ~1.2 GB | YES |
| Shared expert MLP | ~56 MB | 58 | ~3.3 GB | YES |
| Embed + LM head | -- | -- | ~0.2 GB | YES |

**The routed expert weights (~1,305 GB total) dominate the model and are fully replicated.**
Tensor parallelism as currently implemented distributes only the attention and shared expert
weights (~22.7 GB), which is ~1.7% of total model weight. The per-device memory footprint is
NOT meaningfully reduced for DeepSeek-V3.

---

## 9. Key Files

| File | Purpose |
|------|---------|
| `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` | `shard_model()`, `MODULE_TYPE_TO_WRAPPING_FUNC`, partition functions |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Mesh creation, shard_model call, KV cache sharding (with MLA bug) |
| `integrations/vllm_plugin/vllm_tt/vllm_utils.py` | `determine_mesh_shape()` |
| `integrations/vllm_plugin/vllm_tt/platform.py` | `TTConfig`, MLA backend routing |
| `integrations/vllm_plugin/vllm_tt/attention.py` | `TTMLAAttentionBackend`, 4D KV cache shape |
| `integrations/vllm_plugin/vllm_tt/overrides.py` | RMSNorm -> TTRMSNorm replacement |
| `venv/.../vllm/model_executor/models/deepseek_v2.py` | DeepSeek-V3 model definition (module hierarchy) |
