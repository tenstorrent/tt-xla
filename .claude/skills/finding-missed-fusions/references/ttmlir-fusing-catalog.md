# tt-mlir Fusing Catalog

Source of truth for **which fusions tt-mlir already implements at the MLIR level**. Use this to decide, for each missed-fusion candidate, whether it needs a brand-new pattern (`no_pass`), an existing pattern that did not fire (`pass_exists_not_fired`), or an existing pattern that is compiled/configured off (`gated_off`).

Paths are relative to the checked-out tt-mlir source at `third_party/tt-mlir/src/tt-mlir/`. The pinned tt-mlir commit lives in `third_party/CMakeLists.txt` (`set(TT_MLIR_VERSION "...")`) — re-derive this catalog if that commit changes substantially.

There are **two MLIR fusing passes**, one per dialect level:

| Pass (pipeline flag) | Dialect level | Source file | When it runs |
|----------------------|---------------|-------------|--------------|
| `ttir-fusing` | TTIR (framework-shaped ops, pre-backend) | `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` | After StableHLO→TTIR conversion |
| `ttnn-fusing` | TTNN (backend ops, post-lowering, HW-aware) | `lib/Dialect/TTNN/Transforms/TTNNFusing.cpp` | After TTIR→TTNN lowering |

Rule of thumb: framework-shape fusions (matmul+bias→linear, layer/rms norm, activations, RoPE, topk, softmax) live at **TTIR**; hardware-op fusions that depend on TTNN ops/layouts/op-model constraints (conv/matmul + activation, SDPA, NLP head ops, split-QKV) live at **TTNN**.

---

## TTIR-level patterns (`ttir-fusing`)

All registered in `TTIRFusing.cpp` `runOnOperation()`. "Anchor op" = the op type the `OpRewritePattern` matches on (the rewrite's entry point).

| Pattern class | Fuses to (`fused_op`) | Anchor op | Gating | Notes |
|---------------|----------------------|-----------|--------|-------|
| `MatmulWithBiasFusionPattern` | `ttir.linear` | `ttir.add` | always on | matmul + bias add → linear with bias operand |
| `ConvAddBias<Conv2dOp \| ConvTranspose2dOp \| Conv3dOp>` | conv with bias operand | `ttir.add` | always on | folds bias add into conv (sums with existing bias if present) |
| `ConvTagWeights<Conv2dOp \| Conv3dOp>` | (weight tagging, prep) | conv op | always on | not a fusion per se; tags constant weights |
| `ReductionWithReshapePattern<Sum \| Mean \| Max \| Min \| Prod \| ReduceAnd \| ReduceOr \| ArgMax>` | keep-dim reduction | the reduction op | always on | reduction + reshape → reduction with `keep_dim` |
| `SoftmaxFusionPattern` | `ttir.softmax` | `ttir.div` | always on | `div(exp(x), sum(exp(x)))` → softmax |
| `NumericStableSoftmaxFusionPattern` | `ttir.softmax` | `ttir.div` | always on | numerically-stable (max-subtract) softmax variant |
| `RMSNormFusionPattern` | `ttir.rms_norm` | `ttir.multiply` | always on | RMSNorm decomposition → rms_norm |
| `GeluFusionPattern` | `ttir.gelu` | (gelu decomp) | always on | erf/tanh gelu decomposition |
| `Relu6FusionPattern` | `ttir.relu6` | min/max clamp | always on | clamp(0,6) → relu6 |
| `SiluFusionPattern` | `ttir.silu` | `ttir.multiply` | always on | `x * sigmoid(x)` → silu |
| `HardsigmoidFusionPattern` | `ttir.hardsigmoid` | clamp pattern | always on | |
| `MishFusingPattern` | `ttir.mish` | mish decomp | always on | |
| `ReluFusionPattern` | `ttir.relu` | `ttir.maximum` | always on | `maximum(x, 0)` → relu |
| `ScaledSumToMeanPattern` | `ttir.mean` | `ttir.multiply` | always on | `sum(x) * (1/N)` → mean |
| `SpatialMeanOptimizationPattern` | optimized `ttir.mean` | `ttir.mean` | always on | rank-4 spatial (HxW) mean optimization |
| `ConcatenateHeadsUpdatePattern` | concatenate-heads form | `ttir.reshape` | always on | attention head concat reshape |
| `RepVGGConvSumFusionPattern` | fused conv sum | sum/add | always on | RepVGG conv-branch sum |
| `SharedLHSMatmulFusion<MatmulOp \| LinearOp>` | shared-LHS matmul | matmul/linear | always on | merges matmuls that share an LHS operand |
| `ReshapeBroadcastReshapeToRepeatPattern` | `ttir.repeat` | `ttir.reshape` | always on | reshape→broadcast→reshape → repeat |
| `fusing::RoPERotateHalfFusingPattern` | `ttir.rotary_embedding` (RoPE) | rotate-half motif | always on | rotate_half RoPE |
| `fusing::RoPEComplexRotationFusingPattern` | RoPE | complex-rotation motif | always on | complex-multiply RoPE variant |
| `fusing::TopKFusingPattern` | `ttir.topk` | topk motif | always on | (header `Transforms/Fusing/TopKFusingPattern.h`) |
| `ConvWithMultiply<Conv2dOp \| ConvTranspose2dOp>` | conv with scale | `ttir.multiply` | **gated**: `conv2dWithMultiplyEnabled` (default **false**) | fold scale into conv |
| `BatchNormDecomposition` | decomposed BN | batch_norm | **gated**: `conv2dWithMultiplyEnabled` (default **false**) | |
| `PermuteMatmulFusionPattern<MatmulOp \| LinearOp>` | matmul w/ transpose attr | `ttir.permute` | **gated**: `permuteMatmulEnabled` (default **false**) | fold permute into matmul transpose_a/b |

RoPE / TopK pattern bodies live in `lib/Dialect/TTIR/Transforms/Fusing/{RoPEFusingPattern,TopKFusingPattern}.cpp` with headers under `include/ttmlir/Dialect/TTIR/Transforms/Fusing/`.

Pass options (from `include/ttmlir/Dialect/TTIR/Transforms/Passes.td`):
- `ttnn-enable-conv2d-with-multiply-pattern` (`conv2dWithMultiplyEnabled`, default false)
- `enable-permute-matmul-fusion` (`permuteMatmulEnabled`, default false)

---

## TTNN-level patterns (`ttnn-fusing`)

Registered in `TTNNFusing.cpp` `runOnOperation()`.

| Pattern class | Fuses to (`fused_op`) | Anchor op | Gating | Notes |
|---------------|----------------------|-----------|--------|-------|
| `TTNNConv2dWithActivation<ReluOp \| Relu6Op \| SiluOp \| SigmoidOp>` | `ttnn.conv2d` w/ activation in `Conv2dConfig` | `ttnn.conv2d` | always on | folds following activation (and optional reshape) into conv config |
| `TTNNMatmulAndLinearWithActivation<{Matmul,Linear} × {Sigmoid,Silu,Gelu}>` | `ttnn.matmul`/`ttnn.linear` w/ `activation` attr | matmul/linear | always on | folds following activation into the op's `activation` attribute |
| `fusing::SDPAFusing` | `ttnn.scaled_dot_product_attention` / `...attention_decode` | `ttnn.matmul` (the AV matmul) | **gated**: `TTMLIR_ENABLE_OPMODEL` build + `enableOpConstraints` | matmul→(scale)→softmax→matmul → SDPA; validates via op-model. Source: `Transforms/Fusing/SDPAFusingPattern.cpp` |
| `NLPConcatHeadsDecodeFusing` | `ttnn.nlp_concat_heads_decode` | `ttnn.reshape` | **gated**: `TTMLIR_ENABLE_OPMODEL` + `enableOpConstraints` | decode-phase permute+reshape head concat (seq_len==1, tile-aligned) |
| `fusing::SplitQueryKeyValueAndSplitHeadsFusing<MatmulOp \| LinearOp>` | `ttnn.split_query_key_value_and_split_heads` | matmul/linear | **gated**: `TTMLIR_ENABLE_OPMODEL` + `enableOpConstraints` | fused QKV projection + head split. Source: `Transforms/Fusing/SplitQKVFusingPatterns.cpp` |
| `fusing::NLPCreateQKVHeadsDecodeFusing` | `ttnn.nlp_create_qkv_heads_decode` | (QKV head create motif) | **gated**: `TTMLIR_ENABLE_OPMODEL` + `enableOpConstraints` | decode-phase QKV head creation |
| `fusing::RoPERotateHalfFusing` / `RoPEExpandedFusing` / `RoPEDecodeFusing` | RoPE ttnn op | RoPE motifs | **gated**: `TTMLIR_ENABLE_OPMODEL` + `enableOpConstraints` + `enableRoPEFusion` (default **false**; RoPE normally done at TTIR) | Source: `Transforms/Fusing/RoPEFusingPattern.cpp` |
| `TypecastOp::getCanonicalizationPatterns` | (cleanup) | `ttnn.typecast` | always on | folds consecutive typecasts so other patterns match cleanly |

Pass options (from `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`):
- `enable-op-constraints` (`enableOpConstraints`, default false) — required for the op-model-validated patterns above; only effective in a `TTMLIR_ENABLE_OPMODEL` build.
- `max-fallback-attempts` (`maxFallbackAttempts`, default 10000)
- `enable-rope-fusion` (`enableRoPEFusion`, default false)

The op-model-gated TTNN patterns are wrapped in `#ifdef TTMLIR_ENABLE_OPMODEL ... if (enableOpConstraints) { ... }`. If a candidate matches one of these but is not firing, the most likely cause is `gated_off` (build/pass not configured with op constraints), not a missing pattern.

---

## Composite path (not `ttir-fusing`)

Some high-level ops reach TTIR as **StableHLO composites** (`tenstorrent.<op>`), wrapped on the tt-xla side and legalized by `lib/Conversion/StableHLOToTTIR/StableHLOLegalizeCompositePass.cpp` (and `StableHLOToTTIRPatterns.cpp`), **not** by `ttir-fusing`. Known composite names include `tenstorrent.gelu`, `tenstorrent.rms_norm`, `tenstorrent.layer_norm`, `tenstorrent.topk`, `tenstorrent.scaled_dot_product_attention`.

Implication for detection: if the IR shows these ops already collapsed (or shows the `tenstorrent.*` composite), the fusion is already handled via the composite path — do **not** report it. A `layer_norm` that appears as separate `mean/sub/mul/rsqrt/add` with no composite marker and no `ttir.layer_norm` is still a real candidate, but its owner is the composite/Torch-FX path rather than `ttir-fusing` (note this in the report; it is out of scope for the MLIR enable skill).

---

## How to refresh this catalog

```bash
cd third_party/tt-mlir/src/tt-mlir
# TTIR pattern registrations:
rg -n "patterns\.add<" lib/Dialect/TTIR/Transforms/TTIRFusing.cpp
# TTNN pattern registrations:
rg -n "patterns\.add<" lib/Dialect/TTNN/Transforms/TTNNFusing.cpp
# Pass options / gating flags:
rg -n "def TTIRFusing|def TTNNFusing" -A40 include/ttmlir/Dialect/TTIR/Transforms/Passes.td include/ttmlir/Dialect/TTNN/Transforms/Passes.td
```
