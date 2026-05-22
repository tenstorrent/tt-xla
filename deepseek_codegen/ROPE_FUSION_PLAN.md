# RoPE fusion plan — `rotary_embedding_llama` for the 8 MLA RoPE sites

## Why this is parked

After E42 the per-step decode_1 budget is 46,744 μs. The 8 RoPE sites (4 per layer × 2 layers) contribute approximately:

- ~30 small ops per site (slice / reshape / typecast / multiply / addcmul / concat)
- ~120 μs per site of mostly TM device time
- = **~960 μs / decode step**, of which ~700 μs is `Permute`, ~200 μs is `Slice`, ~50 μs is `Concat`, ~40 μs is `Typecast`

Replacing each site with one `ttnn.experimental.rotary_embedding_llama` call should save ~80-100 μs per site → **~600-800 μs total**, or 1.5-1.7 % of full per-step decode. Real but moderate.

## Why `rotary_embedding_llama` (not the alternatives)

| op | RoPE convention | matches DeepSeek? | layout req | conclusion |
| --- | --- | --- | --- | --- |
| `ttnn.experimental.rotary_embedding` | half-concat | ❌ | TILE; `token_index` is uint32 scalar, not tensor | wrong convention |
| `ttnn.experimental.rotary_embedding_hf` | half-concat | ❌ | TILE; prefill: DRAM-interleaved ok; decode: sharded | wrong convention. Adding pre/post permutes to convert to half-concat ≈ same work as just doing the right kernel |
| `ttnn.experimental.rotary_embedding_llama` | **interleaved-pair** via `trans_mat` | ✓ | decode: HEIGHT_SHARDED L1; prefill: DRAM-interleaved ok but needs `seq_len % TILE_SIZE == 0` | **correct, but sharded decode I/O** |
| `ttnn.experimental.rotary_embedding_llama_fused_qk` | interleaved-pair, Q+K fused | ✓ | decode-only, sharded | overkill, even bigger wiring |

DeepSeek's `modified_model.py` does `torch.stack([y_real, y_imag], dim=-1).flatten(-2)` — interleaved-pair. Our codegen emits the matching memory layout (5-dim reshapes ending in `..., 32, 2]`). The correct semantic match is `rotary_embedding_llama` with `trans_mat`.

Our `seq_len = 1` on decode, so we **must** use `is_decode_mode=True` (the prefill-mode kernel needs `seq_len % 32 == 0` and our 1 isn't). That commits us to **HEIGHT_SHARDED L1 inputs** per the kernel's `RotaryEmbeddingLlamaMultiCoreSharded` factory.

## Reference implementation (the canonical setup)

The pattern lives in `models/tt_transformers/tt/rope.py::RotarySetup` (used by `models/demos/llama3_70b_galaxy/tt/llama_rope.py` and the nightly unit test `tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py`). It does:

1. **`trans_mat`** — `get_rot_transformation_mat_v2(dhead=32)` returns a fixed 32×32 BF16 TILE tensor that the kernel multiplies the input against to perform the half-rotate.
2. **`cos_matrix` / `sin_matrix`** — "doubled" format: `torch.stack([cos, cos], dim=-1).flatten(-2)` then `.unsqueeze(0).unsqueeze(0)` → shape `[1, 1, seq_len, head_dim]` (each unique cos value duplicated so the kernel's pair-multiply lands correctly). In decode mode they end up shaped `[1, 1, batch, head_dim]` where each batch row is the cos/sin for that batch's `cur_pos[b]`.
3. **Input** — shape `[1, batch, n_heads, head_dim]` HEIGHT_SHARDED `(TILE_SIZE, head_dim)` per shard, one shard per batch (= batch cores).

Our setup specifics:
- `batch_per_chip = 32`, `n_heads_local = 16` (per chip after model-axis split), `head_dim = 64` (qk_rope_head_dim)
- Need cos/sin sized for the actual current decode position (`args_1`)

## Required new const_evals

Add three new functions to `consteval__main` (or piggy-back on existing ones if the rotary positions are already present in const_eval form):

```python
def main_const_eval_rope_trans_mat(device=None):
    # 32×32 fixed rotation matrix, BF16 TILE
    from models.tt_transformers.tt.common import get_rot_transformation_mat_v2  # or open-code the matrix
    mat = get_rot_transformation_mat_v2(dhead=32)
    return [ttnn.from_torch(mat, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)]

def main_const_eval_rope_cos_doubled(positions_arg, device=None):
    # Build cos in doubled format for ALL valid kv_tokens positions (kv_tokens=128 in our test)
    # Same for sin in a sibling const_eval
    cos, sin = precompute_freqs(head_dim=64, max_seq_len=...)
    cos_doubled = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)  # [1,1,seq,64]
    ...
```

Note: our existing code has separate cos/sin scalars per RoPE site (slices of `var_*`). For the new kernel call we need them in `[1, 1, seq_len, head_dim]` form. May need to either repack the existing const_evals or compute fresh.

## Sharded I/O wiring per RoPE site

Each of the 8 sites needs the same wrapper:

```python
# Currently: 4 RoPE ops (mul + addcmul ×2 + mul) on DRAM-interleaved input
# After: shard → kernel call → unshard

# 1. Prepare Q_rope shape [32, 1, 16, 64] DRAM_INTERLEAVED BF16 TILE
#    → [1, 32, 16, 64] HEIGHT_SHARDED L1 (32 cores × shard_shape=(TILE_SIZE, 64))
_q_rope_4d = ttnn.reshape(q_rope, [1, 32, 16, 64], memory_config=...)
q_rope_shard_mc = ttnn.create_sharded_memory_config(
    shape=(ttnn.TILE_SIZE, 64),
    core_grid=ttnn.num_cores_to_corerangeset(32, device.compute_with_storage_grid_size(), row_wise=True),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
_q_rope_sharded = ttnn.to_memory_config(_q_rope_4d, q_rope_shard_mc)

# 2. cos/sin/trans_mat from const_eval (also sharded — see RotarySetup for how)
_cos = ce_cache__main["main_const_eval_rope_cos_doubled_layerN"]
_sin = ce_cache__main["main_const_eval_rope_sin_doubled_layerN"]
_trans_mat = ce_cache__main["main_const_eval_rope_trans_mat"]

# 3. Call kernel
_q_roped_sharded = ttnn.experimental.rotary_embedding_llama(
    _q_rope_sharded,
    _cos,
    _sin,
    _trans_mat,
    is_decode_mode=True,
    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    ),
)
ttnn.deallocate(_q_rope_sharded, False)

# 4. Convert back to DRAM_INTERLEAVED for the downstream SDPA-input concat
q_roped = ttnn.to_memory_config(
    _q_roped_sharded,
    ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
)
ttnn.deallocate(_q_roped_sharded, False)
# q_roped now has the post-RoPE Q_rope ready for the SDPA Q-concat
```

## Sites to patch

Identified by their addcmul pairs (post-E42 line numbers):

| layer | role | first addcmul line | second addcmul line |
| --- | --- | --- | --- |
| 0 | Q_rope (q_b nope) | 3955 | 3974 |
| 0 | Q_rope (indexer q_b) | 4275 | 4294 |
| 0 | K_rope (k_pe write) | 4863 | 4882 |
| 0 | K_rope (indexer k) | 4971 | 4990 |
| 1 | Q_rope (q_b nope) | 6055 | 6074 |
| 1 | Q_rope (indexer q_b) | 6370 | 6389 |
| 1 | K_rope (k_pe write) | 6943 | 6962 |
| 1 | K_rope (indexer k) | 7051 | 7071 |

Each site spans ~100-110 lines: input slice/typecast/reshape/slice + multiply + addcmul + multiply + addcmul + concat + reshape + (often) slice/concat/reshape. The patch replaces those ~30 ops with the 4-op wrapper above.

## Step-by-step rollout

1. **Wire `trans_mat` const_eval** — single function, simple math. Verify nothing breaks (no consumer yet).
2. **Wire one cos/sin "doubled" const_eval** — pick the first Q_rope site (layer 0 q_b nope). Source the existing cos/sin from where the original chain pulls them; reshape into `[1, 1, kv_tokens, head_dim]` doubled form.
3. **Replace one site only** (3955 / 3974 pair). Pre-test: keep `is_decode_mode=False` first to see if prefill mode is salvageable by padding seq_len to 32; if not, jump straight to decode mode + sharding.
4. **PCC with 600s budget**. Iterate on shape/dtype/sharding errors. The kernel is friendly to debug — TT_FATAL messages typically say exactly which constraint failed.
5. **Once one site is green**, roll out the other 7 mechanically. Per-site cost-saving estimate: ~80-100 μs. With overhead from new shard/unshard, net realistic gain ~50-70 μs per site × 8 = ~400-560 μs.
6. **Tracy + per-region perf** to confirm attn_0/attn_1 numbers dropped, and that we didn't push cost elsewhere (e.g., the to_memory_config sharding might create new TM in `decode_1_start..layer_0_start`).

## Estimated effort

- Setup + first site green: 4-6 hours
- Remaining 7 sites: 1-2 hours (mechanical)
- Perf + PCC validation: 1 hour
- Total: **~1 working day**

## Reproducer if needed

`tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py` — `run_test_rotary_embedding_llama` with `mode == "decode"` and `batch=32, n_heads=16, n_kv_heads=1, head_dim=64`. The `RotarySetup` class in that test handles all the sharding + cos/sin doubled-format math; that's the API we want to mirror in the const_evals.

## Risks

- The kernel's `compute_kernel_config.math_fidelity=HiFi4 fp32_dest_acc_en=True` numerics may differ slightly from our current `multiply + addcmul + multiply + addcmul` BF16 chain. Expect golden PCC to move by ±0.005 (similar to E41 SDPA's effect). Should still be > 0.9 floor.
- The 32-shard L1 input requires 32 dedicated cores. If those collide with anything in `attn_0`'s sparse-matmul / wo placement, we'd need to verify there's no contention.
- The 8 sites split across both Q (heads=16) and K (heads=1) — the `n_heads` argument to `RotarySetup` differs. Two separate const_eval flavors needed.
