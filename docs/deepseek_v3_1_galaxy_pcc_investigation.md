# DeepSeek-V3.1 Galaxy 4-Layer PCC Investigation

Investigation of the low/regressed PCC on `test_deepseek_v3_1_tp_galaxy_4_layers`
(Galaxy 32-device, mesh `(4,8)` = `("batch","model")`).

## TL;DR

| Configuration | 4L Prefill PCC |
|---|---|
| 464 (rebased) baseline, before fixes | **0.937** |
| 464 + RMSNorm `/N` fix | 0.973 |
| c5f398432 (pre-rebase tt-mlir) + same fixes | **0.998** |
| 464 + RMSNorm `/N` fix + c5f MeanOp decomp | 0.973 (unchanged) |

Two independent issues were found:

1. **RMSNorm `/N` missing normalization** — a real graph-lowering bug, fixed
   here. Brings 1-layer from 0.937 → **0.99998** and dense-only 3-layer to
   **0.9999**.
2. **Remaining ~0.025 gap at 4L** is NOT RMSNorm — it is a tensor-parallel
   **sharding/GSPMD lowering difference** between tt-mlir c5f398432 and 464.
   Tracked as a separate follow-up.

## 1. Root cause: RMSNorm distributed decomposition dropped `/N`

Hidden is sharded over the `"model"` axis (8 shards, `7168/8 = 896` per device).
Distributed RMSNorm needs `E(x^2)` over the full hidden, requiring a cross-device
reduction.

`ttnn.rms_norm_pre_all_gather` (the tt-metal fused kernel) emits the per-device
**`sum(x^2)`** (not the mean). The manual decomposition in
`DistributedRMSNormDecompositionRewritePattern.cpp` mistook this for a per-device
*mean* and only did `mean(over devices)` (`÷ numDevices = ÷8`), forgetting to
divide by the per-device hidden size `N (=896)`:

```
manual E(x^2)  = (Σ_d sum_d) / numDevices        = (Σ sum_d) / 8
correct E(x^2) = (Σ_d sum_d) / H                 = (Σ sum_d) / 7168
ratio = H / numDevices = 896   →   variance 896× too large
            →   inv_rms = 1/sqrt(896×) ≈ 1/30   →   output shrunk ~30×
```

The shrunk RMSNorm output collapses the sublayer contribution in the residual
`h + sublayer(norm(h))`, dropping PCC to 0.937.

**Fix:** after `mean(over devices)`, multiply by `1/inputShape.back()` (`= 1/N`)
to recover `E(x^2) = sum(x^2) / H`.

Unit-test verification (single device, mimicking 8-shard distributed RMSNorm):
```
MANUAL (no /N) vs ref shard0: max|diff| = 3.24044
MANUAL + /N    vs ref shard0: max|diff| = 0.02239   ← fixed
```

Benchmark verification:
- 1L: 0.937 → **0.99998**
- dense-only 3L: **0.999887**
- 4L: 0.937 → 0.973 (MoE layer caps it, see §3)

## 2. Why the fused kernel can't be used directly

`ttnn::fused_rms_minimal` (`experimental/ccl/rms_allgather`) has a hard
constraint:
```
TT_FATAL(a.padded_shape()[-2] == input_height, "Only activations with batch size = 32 are supported");
TT_FATAL(M == input_height, "Minimal version assumes (1,1,TILE_HEIGHT,N) shape");
```
DeepSeek prefill (256 tokens) / decode (16 tokens) have `shape[-2] ≠ 32`, so
`isEligibleForFusedKernel` returns false and lowering falls back to the manual
decomposition (where the `/N` bug lived).

## 3. Remaining 4L gap is sharding, NOT RMSNorm

To isolate, c5f398432 (pre-rebase tt-mlir, which gave 4L=0.998) was rebuilt and
its 4L graph compared op-by-op with 464:

- **MoE block ops are byte-for-byte identical** between c5f and 464:
  `sparse_matmul` 24=24, `scatter` 68=68, `reduce_scatter` 192=192,
  `topk`/`softmax`/`silu`/`mesh_partition` all equal → **no MoE regression**.
- The only graph difference is RMSNorm lowering (c5f uses pure `MeanOp`,
  464 uses `rms_norm_pre_all_gather` fused kernel) plus layout ops.

Critically, **replacing 464's RMSNorm decomp with c5f's exact `MeanOp` fallback
still gives 0.973** (not 0.998). So RMSNorm lowering is not the 0.025 gap.

The real difference shows up in `to_layout` shapes:

| | 464 | c5f |
|---|---|---|
| | `576x896`, `1536x896`, `4608x896` | `576x7168`, `1536x7168`, `18432x7168` |
| | `7168x2048` | `7168x16384` (=2048×8) |

464 shards weights/activations per-device (`896 = 7168/8`, `4608 = 18432/4`) and
does partial matmul + reduce; c5f operates on larger (full-hidden) tensors. The
**bf16 accumulation along the sharded reduce path** accounts for the ~0.025 PCC
difference. This is a tt-mlir GSPMD/sharding-propagation change between c5f398432
and 464 (the loader's sharding spec is identical), and is left as a separate
follow-up.

## Files changed

- `third_party/tt-mlir` — `DistributedRMSNormDecompositionRewritePattern.cpp`:
  add `/N` (per-device hidden) normalization after the cross-device mean.
- `third_party/tt_forge_models` — DeepSeek loader: expert sharding
  `("model","batch")` → `("batch","model")` (matches expert_mapping; the wrong
  order was a separate pre-existing bug).
