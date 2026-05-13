# DiT 1-Layer Reshape/Permute Audit Report

**Source:** `tests/torch/models/wan14b/test_wan_dit.py::test_wan_dit_480p_sharded`
**Log:** `dit1layer-perf.log` (TTIR: 2027 lines, TTNN: 2381 lines)
**Perf summary:** `perf-dit14.txt`

## Headline numbers (from perf-dit14.txt stacked report)

| Op class | Device μs | % | Count |
|---|---:|---:|---:|
| ReshapeViewDeviceOperation | 1,263,602 | 17.99% | 197 |
| BinaryNgDeviceOperation | 1,204,497 | 17.14% | 328 |
| SDPAOperation | 855,751 | 12.18% | 12 |
| **PermuteDeviceOperation** | **817,145** | **11.63%** | **82** |
| Conv3dDeviceOperation | 545,286 | 7.76% | 6 |
| MatmulDeviceOperation | 368,786 | 5.25% | 84 |
| TilizeWithValPaddingDeviceOperation | 304,471 | 4.33% | 154 |
| TypecastDeviceOperation | 294,939 | 4.20% | 211 |
| TransposeDeviceOperation | 111,625 | 1.59% | 48 |

Reshape + Permute + Transpose = ~31.2% of device time.

## Findings ranked by impact

### Tier 1 — Massive savings (>1s in full 40-layer model)

#### F1. RoPE strided-write pattern → `aten__index` embedding lookup
**Cost:** ~340 ms / layer (per Agent 3), 4 sites (Q×2, K×2) per attn block.
**Root cause:** `monkey_patch.py:170-173`
```python
out[..., 0::2] = x1*cos - x2*sin
out[..., 1::2] = x1*sin + x2*cos
```
The strided `__setitem__` lowers to `aten__index` → `embedding(W, idx)` + `where`. tt-mlir's existing `RoPEFusingPattern.cpp` (half-rotation form) does not match this interleaved form, so the embedding+permute chain materializes. Each Q/K site:
- `permute(3,0,1,2)` `1x32760x10x64 → 64x1x32760x10` — ~6.2ms
- `reshape → 64x327600` — ~15.4ms
- `embedding` — ~0.7ms
- `tilize` — ~12ms
- `reshape → 128x1x32760x10` — ~11.1ms
- `permute(1,2,3,0) → 1x32760x10x128` — ~26ms
- plus `where`/`ternary` masking — ~6.9ms

**Two viable fixes (mutually exclusive):**

- **(F1a, source rewrite, ~340ms/layer)** Rewrite `apply_rotary_emb` to use the half-rotation form so tt-mlir's existing fuser matches and emits `ttnn.rotary_embedding`. Requires reshaping pairs to contiguous halves and `concat([-second, first], dim=-1)` rotation form.

- **(F1b, compiler workaround, ~200-300ms/layer)** Add `TTIRCommutePermuteThroughEmbedding` to `lib/Dialect/TTIR/Transforms/EraseInverseOps/`. The two surrounding permutes `[3,0,1,2]` and `[1,2,3,0]` are inverses — commuting them past the embedding allows EraseInverseOps to cancel both. Lower yield because the `aten__index→embedding` lowering itself remains.

**Recommendation:** Implement F1a first (simpler, no compiler rebuild, larger savings).

### Tier 2 — Significant savings (~30-200ms)

#### F2. Patchify conv3d output ping-pong
**Cost:** ~33.6 ms (Agent 1 finding).
**Verbatim TTNN** (dit1layer.ttnn.mlir:582-590):
```
%7 = ttnn.conv3d(...) → 1x21x30x52x5120 (NDHWC native)
%8 = ttnn.permute(%7) <{(0,4,1,2,3)}> → 1x5120x21x30x52   # 14.2ms
%9 = ttnn.reshape(%8) → 1x5120x32760                       # 15.0ms (COPY: strides don't match)
%10 = ttnn.to_layout(%9, tile)                             # 4.1ms (forced retile)
%11 = ttnn.permute(%10) <{(0,2,1)}> → 1x32760x5120         # 4.4ms
```
**Root cause:** `monkey_patch.py:62`
```python
hidden_states = hidden_states.flatten(2).transpose(1, 2)
```
This is `flatten(2).transpose(1,2)` of NCDHW. Since TTNN conv3d returns NDHWC natively, the compiler had to insert the `(0,4,1,2,3)` permute to match torch semantics — then the user's code immediately undoes it.

**Fix (F2):** Replace with
```python
hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
    1, -1, hidden_states.shape[1]
)
```
The `(0,2,3,4,1)` algebraically inverts the compiler's `(0,4,1,2,3)`, both cancel via EraseInverseOps.

#### F3. AdaLN modulation hoist + fusion
**Cost:** ~4.2 ms × 40 layers = 168 ms for the `(1+scale)·norm + shift` arithmetic; plus ~5 ms × 40 = 200 ms for chunk/typecast hoisted out (Agent 5).
**Patterns (dit1layer.ttnn.mlir:599-654, 1031-1076):**
```
typecast(bf16→f32) → layer_norm(no affine) → typecast(f32→bf16) → reshape
 → mul(normed, 1+scale_msa) → add(_, shift_msa)
```

**Two fixes:**
- **(F3a, source)** Move the chunk + typecast + `(1+x)` precomputation from `patched_block_forward` to `patched_model_forward` (pre-loop). Saves the per-block chunk/cast cost.
- **(F3b, compiler)** Add `LayerNormAffineFusing` pattern: match `typecast→layer_norm→typecast→reshape→mul(W)→add(B)` with `W=(D,)` and `B=(D,)`, fold into `ttnn.layer_norm` with `operandSegmentSizes=[1,1,1]`. tt-mlir currently lacks any layer-norm fusion file.

#### F4. SDPA-input typecast wrap
**Cost:** ~6.5 ms / layer × 40 = 260 ms (Agent 4).
**Verbatim:**
```
%140 = ttnn.typecast(%96 bf16→f32)
%141 = ttnn.permute(%140) <{(0,2,1,3)}>     # runs in FP32 = 4.4ms vs ~2.2ms BF16
%147 = ttnn.typecast(%141 f32→bf16)
```
Same pattern for Q, K, V → 3 sites × self-attn × 2 attn blocks (cross-attn K/V too).
**Fix:** Remove the fp32 wrap. Keep permute on bf16. Compiler workaround to delete typecast pairs across permute (which is type-preserving anyway). Could be a tt-mlir pattern matching `typecast(t1, t2) → permute → typecast(t2, t1)` where t1=t2.

### Tier 3 — Moderate savings (~50-100ms)

#### F5. Fused QKV projection (`attn.fuse_projections()`)
Diffusers' `WanAttention` supports `fuse_projections=True`. Today Q/K/V matmuls are 3 separate calls on the same input. Fusing makes them one wider matmul + 3 slices. Tt-mlir's existing `SplitQueryKeyValueAndSplitHeadsFusing` pattern (`/root/tt-xla/third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.cpp`) is designed for this case but won't fire because the RMSNorm + RoPE chain interposes between slice and head-permute.

#### F6. ReduceScatter rank-padding reshapes
3 sites × 18.6 ms = 56 ms (Agent 6/8).
**Pattern:** `matmul → reshape(32760×5120 → 1×1×32760×5120) → reduce_scatter → reshape back to 2D`. Pure metadata but materializes because the ttnn layout encoding differs by rank.
**Fix (compiler):** Either let `reduce_scatter` accept rank<4, or treat leading-1 dim add as a true view in the layout encoder.

### Tier 4 — Negligible / already optimized

- **Cross-attention** (Agent 7): mostly view-only reshapes, ~500μs/layer total.
- **Condition embedder** (Agent 10): ~2ms once per forward.
- **Mesh sharding CCL reshapes** (Agent 8): <200μs total. The real CCL cost (396ms = 12%) is the ops themselves; could be improved by `AG(RS(x)) → AR(x)` fusion or sharding-aware residual.
- **NLPConcatHeads** (output side): already fused, ~500μs/call × 24 = ~12ms (essentially free).
- **FFN** (Agent 6/FFN): structurally well-fused — `ttnn.linear` with `activation = "gelu"` set. Verify activation is actually fused at runtime (could be 1.85ms/layer extra if not).
- **Unpatchify 8D permute**: small tensor (32760×64 ≈ 4MB), only ~1.7ms — not worth touching.

## Implementation plan (priority order)

1. **F1a (RoPE rewrite)** — `monkey_patch.py`, expected huge perf win, easy to verify.
2. **F2 (patchify rewrite)** — `monkey_patch.py`, 2-line edit, ~30ms saved.
3. **F3a (AdaLN hoist)** — `monkey_patch.py`, structural refactor.
4. **F5 (fuse_projections)** — small `monkey_patch.py` addition.
5. **F4 (typecast wrap removal)** — needs tt-mlir investigation of why typecast wrap was added.
6. **F3b (LayerNormAffineFusing)** — new tt-mlir pattern.
7. **F1b (PermuteCommuteEmbedding)** — fallback if F1a doesn't fully fire fusion.
8. **F6 (ReduceScatter reshape elimination)** — tt-mlir layout encoding work.

Each is verifiable independently by comparing:
- (a) presence/count of target ops in the TTNN IR (`grep`)
- (b) `warm_times` average from the test

## Verified results (May 2026)

Measured on `test_wan_dit_480p_sharded` (MAX_BLOCKS=1, BH 4-chip):

| Run | Warm time (ms) | Δ vs baseline | PCC |
|---|---:|---:|---|
| baseline (adaln bf16 only) | 1255 | — | 0.99947 |
| + F2 patchify NDHWC | 1221 | -34 ms | 0.99947 |
| + F2 + F1a RoPE half-rotation | 1031 | -224 ms / -17.9 % | 0.99946 |
| + F2 + F1a + tt-mlir SDPA-fold walk-through-Reshape | **1019** | **-236 ms / -18.8 %** | 0.99947 |
| + F2 + F5 fuse_qkv | 1313 | +58 ms (regressed) | 0.99947 |
| + F2 + naive RoPE stack form | 1785 | +530 ms (regressed) | 0.99946 |

### F2 patchify NDHWC — VERIFIED (-34 ms)
IR diff: `ttnn.permute<0,4,1,2,3>` (14ms) + `ttnn.reshape` (15ms) + `ttnn.permute<0,2,1>` (4ms) collapsed to a single direct `ttnn.reshape` on the NDHWC conv3d output (line 8 of `dit1layer.ttnn.mlir` → line 8 of new TTNN: one reshape, no big permutes).

### F1a RoPE half-rotation — VERIFIED (-190 ms incremental)
IR diff: `aten__index` 20→0, `ttnn.where` 10→0, `ttnn.embedding` 4→0, `ttnn.permute` 35→22 (the big `(3,0,1,2)` and `(1,2,3,0)` permutes around the embedding chain are gone). The tt-mlir `RoPEFusingPattern` did **not** fire (still no `ttnn.rotary_embedding`), but the half-rotation rewrite produces a cheaper op sequence anyway — all 4D, only small permutes on dim-2 axis (size 2).

### tt-mlir `ScaledDotProductAttentionFoldScaleRewritePattern` extended — VERIFIED (-12 ms incremental)
**Branch:** `ppadjin/sdpa-scale-fold-walk-through-reshape` in `third_party/tt-mlir/src/tt-mlir/`.
The fold-scale walk-through op list was `{TypecastOp, PermuteOp}`. Wan's lowering puts the multiply BEHIND a reshape (`multiply(scale) → reshape(BSHD) → permute(0,2,1,3) → typecast → SDPA`), so the walk failed and the multiply stayed. Adding `ReshapeOp` to the walk-through list fires the fold for both Q and K, baking `0.0883883387` into SDPA's `scale` attribute and removing two `ttnn.multiply` ops per attention block. PCC stable; warm time 1031 ms → 1019 ms.

### F5 fuse_projections — REGRESSED (do not use as-is)
Combining Q/K/V into one wider matmul triggers the SLOW matmul kernel path (32760×5120×3840 at ~8 % DRAM BW per perf-dit14.txt). Sharding spec also doesn't cover the new `to_qkv.weight`. Needs joint sharding spec + matmul program-config tuning.

### Deferred (not perf-verifiable at MAX_BLOCKS=1)

- **F3a AdaLN chunk hoist** — moves per-block chunk/typecast/(1+x) out of the block loop. At MAX_BLOCKS=1 the chunk runs once either way. Expected ~5 ms × 40 layers = 200 ms saved in full 40-layer runs.

### Compiler-side patterns to land next

The remaining big wins require tt-mlir changes:

- **F1b PermuteCommuteEmbedding** (Agent 9) — generic cancellation of the inverse permute pair around `embedding`. Even with F1a applied, this pattern would help any other model that hits the same `aten__index → embedding → permute pair` idiom.
- **F3b LayerNormAffineFusing** — fold `layer_norm(no affine) → (1+scale)*x → +shift` into `ttnn.layer_norm` with weight/bias operands (`operandSegmentSizes=[1,1,1]`). Saves ~4 ms × 40 layers = ~168 ms.
- **F6 ReduceScatter rank-padding** — `(N,M) → (1,1,N,M) → reduce_scatter → (N,K)` round-trip is pure metadata but materializes due to layout encoding. ~56 ms saved if the layout encoder treats leading-1 dim addition as a view.
- **F4 typecast wrap removal** — the bf16→f32→permute→f32→bf16 idiom around SDPA-input permutes wastes ~260 ms total. Pattern: fold `typecast(t1,t2) → permute → typecast(t2,t1)` to a single bf16 permute.
