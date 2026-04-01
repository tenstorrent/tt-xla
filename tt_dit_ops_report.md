# tt_dit TTNN Ops Evaluation for Z-Image Transformer

## Summary

Evaluated all TTNN ops from the `tt_dit` library (tt-metal/models/tt_dit/) for potential
performance improvements in the Z-Image transformer single-chip TTNN implementation.

**Baseline**: ~15,338ms per iteration (warm), PCC: 0.998898
**Best result**: ~15,114ms per iteration (warm), PCC: 0.998941 (**-1.5% faster**)

## Environment

- Device: Tenstorrent Blackhole (single chip)
- Core grid: 13x10
- Model: Z-Image DiT transformer (2 noise_refiner + 2 context_refiner + 30 main layers)
- Sequence: 3648 unified (3616 image + 32 caption), hidden_dim=3840, 30 heads

## Results Table

| Op | Status | Perf Impact | PCC | Notes |
|----|--------|-------------|-----|-------|
| `ttnn.experimental.minimal_matmul` + `get_matmul_config` | Tested | **+0.5% slower** (15,405ms) | 0.999380 | No optimized configs for Z-Image shapes on 13x10 grid. Default 8x8x8 blocking. Lost fused SiLU (SILU not supported as parametrized activation in minimal_matmul). |
| `ttnn.experimental.rotary_embedding_llama` | Not tested | N/A | N/A | Incompatible cos/sin format. Z-Image uses 3-axis positional encoding with half-dim [1,seq,1,64,1]. Op requires interleaved full-dim [1,1,seq,128]. High integration cost, low benefit (RoPE is small fraction of compute). |
| `ttnn.experimental.nlp_create_qkv_heads` | Tested (V only) | **-1.4% faster** (15,122ms) | 0.998898 | Replaces typecast(F32) + reshape + permute + typecast(BF16) for V path with single fused reshape+head-split. Only applicable to V (Q/K need norm/RoPE between reshape and permute). |
| `ttnn.addcmul` | Tested | **-0.1% additional** | 0.998941 | Fuses multiply(gate, normed) + add(residual) into single op. Applied to 68 locations (34 blocks x 2 paths). Small but cumulative savings. |
| `ttnn.experimental.dit_minimal_matmul_addcmul_fused` | Not applicable | N/A | N/A | Computes `residual + matmul(x,W) * gate`. Z-Image has rms_norm between output projection and gating: `residual + norm(matmul) * gate`. Intermediate norm breaks fusion. Designed for WAN architecture. |
| `ttnn.experimental.minimal_matmul_split` | Not tested | N/A | N/A | Designed for fused QKV projection + split. Z-Image applies per-head norm between projection and head split, preventing fusion. Same L1 overflow issues as minimal_matmul. |
| `ttnn.transformer.joint_scaled_dot_product_attention` | Not applicable | N/A | N/A | Designed for joint spatial+prompt attention. Z-Image concatenates sequences before main layers (unified SDPA). Noise/context refiners run independently on different blocks. |
| `ttnn.experimental.rotary_embedding_llama` (trans_mat) | Not tested | N/A | N/A | Requires 32x32 transformation matrix for pair-wise rotation. Z-Image RoPE uses separate real/imag slicing. Would need complete cos/sin tensor reformatting. |
| CCL ops (`all_gather_async`, `reduce_scatter_minimal_async`, etc.) | Skipped | N/A | N/A | Multi-device collective communication ops. Not applicable to single-chip inference. |
| `ttnn.experimental.conv3d` | Skipped | N/A | N/A | 3D convolution for video/VAE models. Not used in transformer. |

## Combined Best Configuration

Applied together: `nlp_create_qkv_heads` (V path) + `ttnn.addcmul` (gated residuals)

| Run | Baseline | Optimized | Delta |
|-----|----------|-----------|-------|
| Iteration 0 (cold) | 20,797ms | 20,041ms | -3.6% |
| Iteration 1 (warm) | 15,336ms | 15,116ms | -1.4% |
| Iteration 2 (warm) | 15,341ms | 15,112ms | -1.5% |
| PCC | 0.998898 | 0.998941 | +0.004% |

## Detailed Analysis

### Why minimal_matmul didn't help

1. **No optimized blocking configs**: The `get_matmul_config` lookup tables contain configs for
   WAN/Mochi/SD3.5 model shapes, not Z-Image. All Z-Image shapes (3616x3840x3840,
   3648x3840x10240, etc.) fall back to default 8x8x8 blocking.
2. **L1 overflow on small matmuls**: The x_embedder (3616x64x3840) and cap_embedder (32x2560x3840)
   matmuls overflow L1 with default blocking due to small K dimensions.
3. **Lost fused SiLU**: `minimal_matmul` doesn't support SILU as a fused activation
   (`UnaryOpType::SILU` is not a parametrized type), requiring separate silu() call.
4. **Net effect**: The overhead of unfused SiLU + non-optimized blocking configs negated any
   benefit from the op itself.

### Why rotary_embedding_llama didn't work

Z-Image uses a 3-axis positional encoding system where cos/sin are computed from 3 separate
embedding tables (axis 0: 16 dims, axis 1: 24 dims, axis 2: 24 dims = 64 total per cos/sin).
The current RoPE applies in a "split real/imag" format: reshape [128] → [64, 2], slice, multiply.
`rotary_embedding_llama` requires interleaved full-dim [128] with duplicated values [c0,c0,c1,c1,...].
Converting between formats adds complexity for minimal compute savings (RoPE is ~1% of total).

### Why dit_minimal_matmul_addcmul_fused didn't apply

Z-Image's TransformerBlock has a **post-projection RMSNorm** that WAN doesn't:
```
Z-Image:  residual + norm(matmul(x, W)) * gate    [norm breaks fusion]
WAN:      residual + matmul(x, W) * gate           [fully fusable]
```

### What worked: nlp_create_qkv_heads + addcmul

**nlp_create_qkv_heads** (V path): Eliminated unnecessary F32 round-trip (the V typecast to F32
was carried over from codegen but serves no purpose) and fused reshape+permute into single op.
Saves 3 ops per attention call x 34 blocks = 102 ops eliminated.

**addcmul**: Fused `multiply(gate, normed) + add(residual)` into single kernel.
Saves 1 op + 1 intermediate tensor per gate application x 68 locations = 68 ops eliminated.

## Recommendations

1. **Apply the combined nlp_create_qkv_heads + addcmul optimizations** — they're safe, maintain
   PCC, and provide a consistent ~1.5% improvement.

2. **For further optimization**, consider:
   - Adding Z-Image-specific shapes to `get_matmul_config` lookup tables for 13x10 grid
   - Profiling with Tracy to identify the actual bottleneck ops
   - Exploring sharded memory configs (L1 instead of DRAM) for hot-path tensors
   - Fused QKV projection (concatenate to_q/to_k/to_v weights) to reduce matmul dispatch overhead

3. **The tt_dit library ops are designed for multi-chip scaling** — most of the performance-critical
   ops (CCL, ring attention, distributed norms) are for parallelizing across devices. The
   single-chip ops (minimal_matmul, addcmul) provide modest benefits without model-specific tuning.
