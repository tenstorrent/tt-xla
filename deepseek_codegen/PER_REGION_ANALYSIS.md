# Per-region decode device-time breakdown + full-model extrapolation (after E42)

Measured on tt-xla branch `mvasiljevic/deepseek-router-fuse` @ commit `ab322fc09`. Eleven tracy signposts placed inside `_main()` at the five distributed `rms_norm_pre_all_gather` boundaries + after `argmax`. Each signpost pair gives a scope that `tt-perf-report --start-signpost X --end-signpost Y` can carve out from one full-decode-step capture.

## Signpost layout in `_main`

```
decode_1_start                   ← outer (already existed)
 │
 │  (embedding lookup + cache deref + scaler / position setup, ~22.2 ms)
 │
layer_0_start  = attn_0_start    ← at L0 pre-attn rms_norm_pre_all_gather (1st RMS)
 │
 │  attn_0   ─ L0 MLA: RMS → Q/Kv/Indexer projections → RoPE → 3× paged_update_cache → SDPA → wo → residual
 │             ~3.63 ms
 │
attn_0_end / mlp_0_start         ← at L0 pre-MLP rms_norm_pre_all_gather (2nd RMS)
 │
 │  mlp_0   ─ L0 dense MLP: RMS → stacked (gate+up) matmul → SwiGLU → down matmul → residual
 │            ~0.67 ms
 │
layer_0_end / layer_1_start      ← at L1 pre-attn rms_norm_pre_all_gather (3rd RMS)
   = mlp_0_end = attn_1_start
 │
 │  attn_1   ─ L1 MLA: same shape as attn_0
 │             ~3.65 ms
 │
attn_1_end / moe_start           ← at L1 pre-MoE rms_norm_pre_all_gather (4th RMS)
 │
 │  moe     ─ L1 MoE: RMS → gate matmul → deepseek_grouped_gate → all_to_all_dispatch → sparse_matmul ×2
 │            → SwiGLU → all_to_all_combine → deepseek_moe_post_combine_tilize → reduce_scatter
 │            → all_gather → residual
 │            ~15.54 ms
 │
moe_end / layer_1_end / lm_head_start
 │                               ← at final rms_norm_pre_all_gather (5th RMS)
 │  lm_head ─ final RMS → matmul → reduce_scatter → all_gather → argmax
 │            ~1.05 ms
 │
lm_head_end                      ← right before _main return
 │
decode_1_end                     ← outer (already existed)
```

## Measured per-region device time (decode_1 scope, E42 state)

| Region | Span | μs |
| --- | --- | ---: |
| attn_0 | `layer_0_start` → `attn_0_end` | **3,630** |
| mlp_0 (dense) | `mlp_0_start` → `layer_0_end` | **667** |
| layer_0 (full) | `layer_0_start` → `layer_0_end` | 4,297 |
| attn_1 | `layer_1_start` → `attn_1_end` | **3,646** |
| moe | `moe_start` → `moe_end` | **15,544** |
| layer_1 (full) | `layer_1_start` → `layer_1_end` | 19,189 |
| lm_head | `lm_head_start` → `lm_head_end` | **1,051** |
| Signposted subtotal | | 24,537 |
| Pre-layer-0 setup | `decode_1_start` → `layer_0_start` | ≈ 22,207 |
| **`decode_1` total** | | **46,744** |

Per-region `summary.txt` + `summary.png` artifacts live in `deepseek_codegen/perf_reports/regions/E42_<region>_summary.{txt,png}`.

Key observations:

* `attn_0` and `attn_1` are within **0.4 %** of each other (3,630 vs 3,646 μs) — the MLA attention block is shape-identical across layers, so the per-layer attn cost is essentially a constant.
* `moe` is **23× the cost of `mlp_0`** (15,544 μs vs 667 μs). Almost the whole MoE cost lives in the two `sparse_matmul` ops + the `all_to_all_dispatch`/`combine` pair + the `deepseek_moe_post_combine_tilize` + `reduce_scatter` chain.
* `lm_head` is small (1.05 ms) but contains the 1,266 μs `argmax` of the full 129,280-wide vocab — about 90 % of the LM-head block is that one op.
* The 22.2 ms `pre-layer-0` setup is mostly **one-shot per decode step**: token embedding lookup, current-position scaler, indexer-mask scaffold (FastReduceNC ops, etc.), and the layer-0 pre-attention prep. This cost does NOT multiply with layer count for the full model.

## Full-model extrapolation

DeepSeek V3.2 has **61 transformer layers** (`num_hidden_layers=61`) with `first_k_dense_replace=3`, so **3 dense layers + 58 MoE layers**.

Modeling decode time as

```
full_decode ≈ pre + N_total × attn + N_dense × mlp_dense + N_moe × moe + lm_head
```

with the measured constants:

| symbol | value (μs) |
| --- | ---: |
| `attn` (avg of attn_0, attn_1) | 3,638 |
| `mlp_dense` (= mlp_0) | 667 |
| `moe` | 15,544 |
| `pre` (decode_1_start → layer_0_start) | 22,207 |
| `lm_head` | 1,051 |

→
```
full_decode = 22,207 + 61 × 3,638 + 3 × 667 + 58 × 15,544 + 1,051
            = 22,207 + 221,918  + 2,001       + 901,552    + 1,051
            =                                                1,148,729 μs
            ≈ 1.149 s / decode token
            ≈ 0.87 tokens / second
```

### Where the time goes at full scale

| Component | μs | % of full-model decode |
| --- | ---: | ---: |
| 58× MoE MLP | 901,552 | **78.5 %** |
| 61× MLA attention | 221,918 | **19.3 %** |
| Fixed setup (pre) | 22,207 | 1.9 % |
| LM head | 1,051 | 0.09 % |
| 3× dense MLP | 2,001 | 0.17 % |

**Optimisation leverage at the full-model level is almost entirely in the MoE block.** Halving `moe` would cut full-model decode by ~39 %; halving `attn` would only cut it by ~10 %. The dense-MLP, LM-head, and pre-setup contributions are rounding error against the 61-layer total.

### Sensitivity to layer mix

For reference, varying the dense:MoE split (V3.2 ships as 3:58, but the model card discusses variants):

| Mix | μs / token | tok/s |
| --- | ---: | ---: |
| 3 dense + 58 MoE (V3.2 default) | 1,148,729 | 0.87 |
| 1 dense + 60 MoE | 1,180,367 | 0.85 |
| 0 dense + 61 MoE | 1,196,222 | 0.84 |

The variation between these scenarios is < 5 % — the dense layers are too cheap relative to MoE to noticeably shift the total.

## What this implies for the next round of tuning

After E38 (router fusion) + E41 (MLA SDPA) + E42 (dead-code delete), the per-decode-step number stands at 46,744 μs (-85.4 % vs E0). To move the full-model number significantly:

1. **MoE is the only meaningful lever.** Per-iter, the MoE block is 15,544 μs, of which:
   - `SparseMatmulDeviceOperation` ≈ 5,565 μs (two matmuls)
   - `AllToAllCombineDeviceOperation` ≈ 1,012 μs
   - `AllToAllDispatchDeviceOperation` ≈ 209 μs
   - `MoeExpertTokenRemapDeviceOperation` ≈ 172 μs
   - The remaining ~8,500 μs is layout / reshape / typecast / reduce_scatter / all_gather overhead around the actual MoE compute. The plan to attack that lives in `deepseek_codegen/TM_REDUCTION_PLAN.md`.

2. **Attention is already close to floor.** E41 collapsed the per-layer attn block into one SDPA call + Q/K/V projections. The remaining 3.6 ms is ~30 % weight tilize (auto-inserted at matmul input), ~25 % SDPA + wo matmul, ~15 % all_gather / reduce_scatter, ~30 % small layout/typecast ops. Each of these is hard to cut by half without a structural change (CP wo, paged kvpe cache, sharded I/O on every matmul).

3. **Pre / LM-head / dense-MLP are not worth attacking.** Together they're <2 % of full-model decode.

## How to regenerate this analysis

1. Run tracy on the current `_main`:
   ```bash
   cd deepseek_codegen/graph_0 && bash run -t
   ```
2. Note the timestamped report dir under `generated/profiler/reports/`.
3. For each region, scope `tt-perf-report` to the appropriate signpost pair:
   ```bash
   tt-perf-report ops_perf_results_<ts>.csv \
       --start-signpost layer_0_start \
       --end-signpost   attn_0_end \
       --summary-file   <ts>/attn_0_summary
   ```
   Available signpost pairs:
   `layer_0_start..attn_0_end` (attn_0),
   `mlp_0_start..layer_0_end` (mlp_0),
   `layer_0_start..layer_0_end` (layer_0),
   `layer_1_start..attn_1_end` (attn_1),
   `moe_start..moe_end` (moe),
   `layer_1_start..layer_1_end` (layer_1),
   `lm_head_start..lm_head_end` (lm_head).
