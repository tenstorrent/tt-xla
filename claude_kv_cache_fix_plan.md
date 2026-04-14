# Fix MLA KV Cache Failures in `_tt_concat_and_cache_mla`

## Context

Running `test_tensor_parallel_generation_llmbox_small` with DeepSeek-V3 fails during engine
initialization with:

```
torch.cat([kv_c, k_pe], dim=-1)
RuntimeError: Number of dimensions must match. Expected 3-D, got 4-D for tensor 1
  kv_c  shape: (1, 32, 512)
  k_pe  shape: (1, 32, 32, 64)
```

Root-cause trace reveals **two distinct bugs**.

---

## Bug 1: Wrong KV cache shape for MLA layers ✅ FIXED

**File**: `integrations/vllm_plugin/vllm_tt/model_runner.py`, ~line 2250

### Problem
`MLAAttentionSpec` is a subclass of `AttentionSpec`. The allocation loop checked
`isinstance(kv_cache_spec, AttentionSpec)`, so MLA layers entered the standard path and called
`TTAttentionBackend.get_kv_cache_shape(...)`, which returns **5D**:
```
(2, num_blocks, num_kv_heads, block_size, head_size)
```
MLA needs **4D** from `TTMLAAttentionBackend.get_kv_cache_shape(...)`:
```
(num_blocks, num_kv_heads, block_size, head_size)
```

Consequences (if left unfixed):
- `_tt_concat_and_cache_mla` reads `kv_cache.shape[2]` as `num_kv_heads=1` instead of `block_size`
- `forward_mqa` receives a 5D cache instead of the expected 4D

### Fix applied
Added `MLAAttentionSpec` guard before the `AttentionSpec` branch, using
`TTMLAAttentionBackend.get_kv_cache_shape` for MLA layers.  Also added `TTMLAAttentionBackend`
to the import from `.attention` in `model_runner.py`.

---

## Bug 2: Wrong `k_pe` shape from rotary-embedding broadcast (NOT YET FIXED)

**Files**: `integrations/vllm_plugin/vllm_tt/attention.py` (two changes needed)

### Problem

The TT model runner passes 2D positions `(batch=1, seq=32)`. In `vllm/model_executor/layers/mla.py`:

1. `kv_lora` has shape `(batch, seq, kv_lora_rank+rope_dim)` = `(1, 32, 576)`
2. After split: `kv_c=(1,32,512)`, `k_pe=(1,32,64)`
3. `k_pe.unsqueeze(1)` gives `(1, 1, 32, 64)` (designed for flat `[T,D]`, gets batched `[B,T,D]`)
4. `DeepseekScalingRotaryEmbedding.forward_native` indexes `cos_sin_cache` with 2D positions,
   producing `cos/sin` of shape `(1, 32, 1, 64)` (4D after repeat+unsqueeze)
5. Broadcasting `(1,1,32,64) * (1,32,1,64)` yields `(1,32,32,64)` — incorrect!

This `(1,32,32,64)` shaped `k_pe` is passed (via `k_pe.squeeze(1)`) as the second argument
to `_tt_concat_and_cache_mla`, which then fails at `torch.cat`.

### Fix A needed: Patch `DeepseekScalingRotaryEmbedding.forward_native` in `attention.py`

Add a monkey-patch near the existing `_tt_concat_and_cache_mla` monkey-patch (line 65):

```python
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)

_orig_deepseek_rope_forward_native = DeepseekScalingRotaryEmbedding.forward_native

def _tt_deepseek_rope_forward_native(self, positions, query, key=None, offsets=None):
    """Handle TT batched (B, T) positions by flattening before cos/sin indexing."""
    if positions.dim() != 2:
        return _orig_deepseek_rope_forward_native(self, positions, query, key, offsets)

    batch, seq = positions.shape
    positions_flat = positions.flatten()          # (T,)

    # key = (B, 1, T, rope_dim) from mla.py unsqueeze(1) on batched input.
    # Reshape to flat (T, 1, rope_dim) so cos (T, 1, rope_dim) broadcasts correctly.
    key_orig_shape = None
    if key is not None and key.dim() == 4:
        key_orig_shape = key.shape                # remember for reshape-back
        # (B, 1, T, D) -> (B, T, 1, D) -> (B*T, 1, D)
        key = key.permute(0, 2, 1, 3).reshape(batch * seq, key.shape[1], key.shape[-1])

    result_q, result_k = _orig_deepseek_rope_forward_native(
        self, positions_flat, query, key, offsets
    )

    # Reshape result_k back: (T, 1, D) -> (B, 1, T, D)
    if key_orig_shape is not None and result_k is not None:
        b, one, t, d = key_orig_shape
        result_k = result_k.reshape(b, t, one, d).permute(0, 2, 1, 3)

    return result_q, result_k

DeepseekScalingRotaryEmbedding.forward_native = _tt_deepseek_rope_forward_native
```

After this patch, `k_pe` is correctly shaped as `(1, 1, 32, 64)` after rotary emb.
`k_pe.squeeze(1)` then gives `(1, 32, 64)` (3D), not 4D.

### Fix B needed: Flatten batched inputs in `_tt_concat_and_cache_mla`

Even with Fix A, `kv_c` and `k_pe` arrive as 3D `(1, 32, 512)` / `(1, 32, 64)` because
the TT model runner uses batched format. Add `.reshape(-1, dim)` at the top:

```python
def _tt_concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
    # Flatten batched [B, T, dim] → flat [T, dim] expected by the rest of the function.
    kv_c = kv_c.reshape(-1, kv_c.shape[-1])   # [T, kv_lora_rank]
    k_pe = k_pe.reshape(-1, k_pe.shape[-1])    # [T, qk_rope_head_dim]
    kv_combined = torch.cat([kv_c, k_pe], dim=-1)
    ...  # rest unchanged
```

---

## Bug 3: `W_UK_T` / `W_UV` left on CPU after `model.to(device)` ✅ FIXED

**File**: `integrations/vllm_plugin/vllm_tt/model_runner.py`, after line 1720

### Problem
vllm's `process_weights_after_loading` creates derived tensors (`W_UK_T`, `W_UV`) as plain
Python attributes (not `nn.Parameter` or buffers). `nn.Module.to(device)` only moves parameters
and buffers, so these stay on CPU while activations are on XLA → device mismatch in `torch.bmm`.

### Fix applied
After `model.to(self.device)`, iterate over all modules and move any plain tensor attributes
still on the wrong device.

---

## Bug 4: Dynamo traces past `kv_cache.numel() == 0` guard ✅ FIXED

**File**: `integrations/vllm_plugin/vllm_tt/attention.py`

### Problem
During profile/dummy run, `kv_cache` is `torch.tensor([])` (1D). Dynamo inlines
`_tt_concat_and_cache_mla` and fails on `kv_cache.shape[2]` (index out of range for 1D tensor).
A `numel() == 0` guard didn't work because `numel()` is symbolic during tracing.

### Fix applied
Use `kv_cache.dim() < 4` guard — `dim()` returns a concrete int during Dynamo tracing.

---

## Files modified

| File | Change |
|------|--------|
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Bug 1: `MLAAttentionSpec` guard + import ✅ |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Bug 3: Move stray CPU tensors to device ✅ |
| `integrations/vllm_plugin/vllm_tt/attention.py` | Bug 2A + 2B: RoPE patch + flatten ✅ |
| `integrations/vllm_plugin/vllm_tt/attention.py` | Bug 4: `dim() < 4` guard ✅ |

## Verification

Re-run the failing test:
```bash
pytest -svv tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py \
    -k "test_tensor_parallel_generation_llmbox_small[True-deepseek-ai/DeepSeek-V3-False-]"
```
