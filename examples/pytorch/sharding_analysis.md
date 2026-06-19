# Sharding analysis - Qwen3.6-27B

> Living document. Built up across the skill's phases, finalized in phase 8.

## 1. Model & target
- Model / variant: `Qwen/Qwen3.6-27B` (HF model_type: `qwen3_5`, architectures: `Qwen3_5ForConditionalGeneration`)
- Load script: `examples/pytorch/qwen3_6_27b_tp_decode.py`
- dtype: `bfloat16`
- Device count: **4** (QB2)
- Target mesh shape: `(1, 4)` with axes `("batch", "model")`
- Currently running with **4 layers** (3 DeltaNet + 1 full attention, 4.1B params) for bringup

## 2. Architecture

**Total parameters: 26.9B** (full model), **4.1B** (4-layer bringup)

64 decoder layers, repeating pattern: 3× linear_attention (DeltaNet) + 1× full_attention, ×16 groups.
- **48 DeltaNet layers** (indices 0,1,2,4,5,6,8,...,62)
- **16 Full Attention layers** (indices 3,7,11,...,63)
- All layers use **dense MLP** (`Qwen3_5MLP`, not MoE)

### DeltaNet layer (×48, representative: layer 0)

| Parameter | Shape | Params | Sharding status |
|---|---|---|---|
| `linear_attn.in_proj_qkv.weight` | (10240, 5120) | 52.4M | **Replicated** (conv1d limitation) |
| `linear_attn.in_proj_z.weight` | (6144, 5120) | 31.5M | **Replicated** (conv1d limitation) |
| `linear_attn.in_proj_b.weight` | (48, 5120) | 246K | **Replicated** (conv1d limitation) |
| `linear_attn.in_proj_a.weight` | (48, 5120) | 246K | **Replicated** (conv1d limitation) |
| `linear_attn.out_proj.weight` | (5120, 6144) | 31.5M | **Replicated** (conv1d limitation) |
| `linear_attn.conv1d.weight` | (10240, 1, 4) | 41K | **Replicated** (root cause) |
| `linear_attn.dt_bias` | (48,) | 48 | Replicated (tiny) |
| `linear_attn.A_log` | (48,) | 48 | Replicated (tiny) |
| `linear_attn.norm.weight` | (128,) | 128 | Replicated (per-head-dim norm) |
| `mlp.gate_proj.weight` | (17408, 5120) | 89.1M | **Sharded** `("model", None)` |
| `mlp.up_proj.weight` | (17408, 5120) | 89.1M | **Sharded** `("model", None)` |
| `mlp.down_proj.weight` | (5120, 17408) | 89.1M | **Sharded** `(None, "model")` |
| `input_layernorm.weight` | (5120,) | 5.1K | Replicated (norm) |
| `post_attention_layernorm.weight` | (5120,) | 5.1K | Replicated (norm) |

### Full Attention layer (×16, representative: layer 3)

| Parameter | Shape | Params | Sharding status |
|---|---|---|---|
| `self_attn.q_proj.weight` | (12288, 5120) | 62.9M | **Sharded** `("model", None)` |
| `self_attn.k_proj.weight` | (1024, 5120) | 5.2M | **Sharded** `("model", None)` |
| `self_attn.v_proj.weight` | (1024, 5120) | 5.2M | **Sharded** `("model", None)` |
| `self_attn.o_proj.weight` | (5120, 6144) | 31.5M | **Sharded** `(None, "model")` |
| `self_attn.q_norm.weight` | (256,) | 256 | Replicated (per-head-dim norm) |
| `self_attn.k_norm.weight` | (256,) | 256 | Replicated (per-head-dim norm) |
| `mlp.gate_proj.weight` | (17408, 5120) | 89.1M | **Sharded** `("model", None)` |
| `mlp.up_proj.weight` | (17408, 5120) | 89.1M | **Sharded** `("model", None)` |
| `mlp.down_proj.weight` | (5120, 17408) | 89.1M | **Sharded** `(None, "model")` |
| `input_layernorm.weight` | (5120,) | 5.1K | Replicated (norm) |
| `post_attention_layernorm.weight` | (5120,) | 5.1K | Replicated (norm) |

### Embedding & head

| Parameter | Shape | Params | Sharding status |
|---|---|---|---|
| `model.embed_tokens.weight` | (248320, 5120) | 1.27B | Replicated (single-token lookup) |
| `model.norm.weight` | (5120,) | 5.1K | Replicated |
| `lm_head.weight` | (248320, 5120) | 1.27B | **Sharded** `("model", None)` |

### KV Cache (StaticCache, 16 TTStaticLayer for full_attention layers)

| Tensor | Shape | Sharding |
|---|---|---|
| `layer.keys` | (1, 4, max_cache_len, 256) | **Sharded** `(None, "model", None, None)` |
| `layer.values` | (1, 4, max_cache_len, 256) | **Sharded** `(None, "model", None, None)` |

### Divisibility (all ÷ 4)

All sharded dimensions divide cleanly by 4: 10240/4=2560, 6144/4=1536, 48/4=12, 17408/4=4352, 12288/4=3072, 1024/4=256, 248320/4=62080. ✓

## 3. Recommended strategy (summary)

**Megatron-style tensor parallelism** on mesh `(1, 4)` with axis `"model"`. MLP and full attention projections use standard column→row pairs with 1 all-reduce each. `lm_head` is sharded column-parallel. DeltaNet `linear_attn` projections are **fully replicated** due to a compiler limitation with depthwise conv1d (see §6). **Total: 81 all-reduces per forward pass** (64 MLP + 16 attention + 1 lm_head).

## 4. Per-component design

### Full Attention (×16 layers)
- **Chosen sharding:** `q/k/v_proj ("model", None)` column-parallel; `o_proj (None, "model")` row-parallel
- **Why:** Standard Megatron §3 attention TP — each device gets a subset of heads, one all-reduce after o_proj. Matches `examples/pytorch/llama.py`.
- **Rejected alternatives:** Data parallelism — no compute split, doesn't help with 1-batch decode.
- **CCLs:** 1 all-reduce per layer (hidden_size × batch × seq = 5120 × 1 × 1 × 2B = 10 KB per decode step)

### DeltaNet (×48 layers) — BLOCKED by compiler limitation
- **Ideal sharding:** `in_proj_qkv, in_proj_z, in_proj_b, in_proj_a ("model", None)` column-parallel; `conv1d ("model", None, None)`; `out_proj (None, "model")` row-parallel
- **Actual sharding:** All `linear_attn` projections **replicated**. Only the MLP within DeltaNet layers is sharded.
- **Why replicated:** The depthwise `conv1d` (groups=10240) causes a `feature_group_count` mismatch when channel dim is partitioned. Shardy incorrectly scales `feature_group_count` to the full unpartitioned value (10240) while the per-device input only has 2560 channels. StableHLO verifier rejects: `input_feature_dim (2560) must be divisible by feature_group_count (10240)`. Sharding just `conv1d.weight` doesn't fix it — the attribute is static in the StableHLO op. See [tt-xla#3508](https://github.com/tenstorrent/tt-xla/issues/3508).
- **Impact:** 48 layers × ~116M params = ~5.6B params fully replicated across all devices. No compute parallelism for DeltaNet attention ops.
- **CCLs (actual):** 0 for linear_attn (replicated), 1 all-reduce per MLP = 48 total

### Dense MLP (×64 layers)
- **Chosen sharding:** `gate_proj, up_proj ("model", None)` column-parallel; `down_proj (None, "model")` row-parallel
- **Why:** Textbook Megatron MLP pair. Matches Llama reference and `general_sharding.md`.
- **Rejected alternatives:** None — this is the standard pattern.
- **CCLs:** 1 all-reduce per layer (10 KB per decode step)

### lm_head
- **Chosen sharding:** `("model", None)` column-parallel
- **Why:** `lm_head.weight` is 1.27B params (248320 × 5120) — the single largest weight, especially dominant in the 4-layer bringup config where it's 31% of the model. Sharding splits the vocab matmul across 4 devices.
- **CCLs:** 1 all-reduce (or all-gather depending on how argmax interacts with sharded logits)

### Embedding
- **Chosen sharding:** Replicated (no annotation)
- **Why:** During decode only 1 token is looked up — no compute benefit from sharding. Consistent with Llama reference.
- **CCLs:** 0

### KV Cache
- **Chosen sharding:** Keys/values `(None, "model", None, None)` — shard along num_kv_heads (dim 1)
- **Why:** Must match attention weight sharding so Q @ K^T operates on local head slices without gathering. 4 KV heads / 4 devices = 1 head per device.
- **CCLs:** 0 (attention is local per head)

## 5. CCL budget

| Component | Collective | Count per fwd | Approx size (decode) |
|---|---|---|---|
| Full Attention `o_proj` | all-reduce | 16 | 16 × 10 KB = 160 KB |
| MLP `down_proj` | all-reduce | 64 | 64 × 10 KB = 640 KB |
| `lm_head` | all-reduce | 1 | ~10 KB |
| DeltaNet `linear_attn` | none (replicated) | 0 | 0 |
| **Total** | **all-reduce** | **81** | **~810 KB** |

Note: During prefill (seq_len=128), each all-reduce is ~1.3 MB instead of 10 KB, so total prefill CCL ≈ 105 MB.

## 6. Bottlenecks & compiler limitations

1. **DeltaNet conv1d feature_group_count mismatch (BLOCKER):** Depthwise `conv1d` (groups=10240) inside `GatedDeltaNet` prevents sharding ALL `linear_attn` projections. When `in_proj_qkv` is sharded column-parallel, its output (10240 channels) is split to 2560 per device. The StableHLO convolution's `feature_group_count` attribute remains 10240 (not partitioned), causing a verifier error. Sharding the `conv1d.weight` alone doesn't help — `feature_group_count` is a static op attribute, not derived from tensor shapes. Tracked in [tt-xla#3508](https://github.com/tenstorrent/tt-xla/issues/3508). **This leaves 48 layers (~5.6B params) fully replicated with zero compute parallelism.**

2. **Embeddings replicated (1.27B params):** Each device holds a full copy. Not a speed concern (single-token lookup) but a memory concern if DRAM is tight (~2.4 GB per device in bf16).

3. **Steady-state decode: ~8.2s/token for 4 layers.** Timing instrumentation shows `fwd=0.001s, sync=8.2s` — all time is TTNN binary execution on device. No host-side bottleneck, no recompilation, no large transfers. The compiled graph itself is slow, likely due to many sequential small ops in the DeltaNet recurrent path. **Next step: run with `TTMLIR_ENABLE_PERF_TRACE=1` to identify which TTNN ops dominate.**

## 7. Sources
- Megatron-LM tensor parallelism: [arxiv 1909.08053](https://arxiv.org/abs/1909.08053), §3
- Llama TP reference: `examples/pytorch/llama.py` (lines 310-317)
- CCL accounting: `references/ccl_cheatsheet.md`
- Shardy sharding: `references/shardy_sharding.md`
- Mesh shapes: `references/mesh_shapes.md` — QB2 = (1, 4)
- Compiler limitations: `references/compiler_support.md`
- DeltaNet conv1d issue: [tt-xla#3508](https://github.com/tenstorrent/tt-xla/issues/3508)

## 8. Sharding coverage summary (actual, after conv1d workaround)

| Category | What | Params | Status |
|---|---|---|---|
| MLP (all 64 layers) | gate/up/down_proj | 17.1B | ✓ Sharded |
| Full Attention (16 layers) | q/k/v/o_proj | 1.68B | ✓ Sharded |
| lm_head | weight | 1.27B | ✓ Sharded |
| KV Cache (16 layers) | keys, values | — | ✓ Sharded |
| DeltaNet linear_attn (48 layers) | all projections + conv1d | 5.57B | ✗ Replicated (compiler limitation) |
| embed_tokens | weight | 1.27B | ✗ Replicated (intentional) |
| Norms, small params | layernorms, dt_bias, A_log | ~2.6M | ✗ Replicated (intentional, tiny) |
| **Total** | | **26.9B** | **74.5% sharded, 25.5% replicated** |
