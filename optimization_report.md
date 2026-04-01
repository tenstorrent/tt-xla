# Z-Image Transformer Optimization Report

## Summary

Applied a series of targeted optimizations to the Z-Image transformer TTNN implementation,
achieving **-18.6% latency reduction** with **improved numerical accuracy**.

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Warm latency | 15,336 ms | 12,480 ms | **-18.6%** |
| FPS | 0.07 | 0.08 | +14% |
| PCC vs PyTorch | 0.998898 | 0.999483 | **+0.059%** |

Device: Tenstorrent Blackhole p150 (single chip, 13x10 core grid).

## Changes Applied

### 1. BFLOAT16 RoPE — eliminated F32 typecast round-trip (-16.0%)

**The biggest single win.** The original codegen inserted F32 typecasts around RoPE arithmetic
that were unnecessary.

**Before** (per Q/K, per block):
```
rms_norm → typecast(FLOAT32) → RoPE math (F32) → permute → typecast(BFLOAT16)
```

**After:**
```
rms_norm → RoPE math (BF16) → permute
```

This eliminates:
- 4 typecast ops per attention (2 to F32, 2 back to BF16) × 34 blocks = **136 typecast ops removed**
- All RoPE multiply/add/subtract ops operate on BF16 tensors (half the memory bandwidth vs F32)
- The rope_cos/rope_sin tables are converted to BF16 once at init time

**Why it's safe:** The RoPE computation is simple element-wise multiply/add on pre-normalized
values (after QK-norm with RMS values near 1.0). BF16 has sufficient dynamic range for these
operations. PCC actually improved, confirming no accuracy regression.

**Why it's so impactful:** Each typecast is a full data movement operation that reads and writes
the entire tensor. For the main layers, Q and K tensors are [1, 3648, 30, 128] = ~53MB each.
4 typecasts × 53MB × 30 main blocks alone = ~6.4GB of unnecessary data movement per forward pass.

**Files changed:** `model_ttnn.py` — `_apply_rope()` dtype arguments, removed typecasts in
`Attention.__call__()`, added BF16 conversion for rope tables in `ZImageTransformerTTNN.__init__()`.

### 2. Compute kernel config tuning (-1.6%, +PCC)

Changed matmul compute config from the codegen default to a config validated by the tt_dit
library and our own sweep of 13 configurations.

**Before:**
```python
COMPUTE_CONFIG = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi4, fp32_dest_acc_en=True
)
```

**After:**
```python
COMPUTE_CONFIG = WormholeComputeKernelConfig(
    math_fidelity=MathFidelity.HiFi2,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

Two changes from baseline:
- **HiFi2** (was HiFi4): Reduces multiply-accumulate precision from 4 iterations to 2. Safe
  because `fp32_dest_acc_en=True` keeps the running accumulator in FP32, preventing error
  accumulation across the K=3840 reduction dimension (120 tiles).
- **packer_l1_acc=True** (was False): Keeps packer intermediate data in L1 instead of DRAM.
  Reduces DRAM bandwidth pressure and improves accuracy (L1 has no quantization on intermediate
  values). This is a free improvement — faster AND more accurate.

The norm config (COMPUTE_CONFIG_NORM) was already optimal and unchanged: HiFi4 with fp32_acc
and packer_l1_acc. Normalization operations (mean, variance, rsqrt) are sensitive to precision
and benefit from HiFi4.

**Files changed:** `model_ttnn.py` — `COMPUTE_CONFIG` constant (line 33-36).

### 3. nlp_create_qkv_heads for V path (-1.4%)

Replaced the V projection's post-processing chain with a single fused op from the tt_dit library.

**Before:**
```python
v = matmul(x, to_v)                    # [seq, 3840] BF16
v = typecast(v, FLOAT32)               # unnecessary F32 conversion
v = reshape(v, [1, seq, 30, 128])      # reshape to heads
v = permute(v, [0, 2, 1, 3])           # transpose to heads-first
# ... later: v = typecast(v, BFLOAT16)  # convert back
```

**After:**
```python
v = matmul(x, to_v)                              # [seq, 3840] BF16
v = reshape(v, [1, 1, seq, 3840])                # prep for head split
v, _, _ = nlp_create_qkv_heads(v, num_heads=30)  # [1, 30, seq, 128] BF16
```

Eliminates: 1 unnecessary F32 typecast, 1 reshape, 1 permute, 1 BF16 typecast.
Replaces with: 1 reshape + 1 fused head-split op.

Not applicable to Q/K because they need per-head RMSNorm + RoPE between reshape and permute.

**Files changed:** `model_ttnn.py` — `Attention.__call__()` V projection section.

### 4. ttnn.addcmul for gated residuals (-0.1%)

Fused the gate-multiply + residual-add pattern in modulated transformer blocks.

**Before:**
```python
gated = ttnn.multiply(gate, normed)    # intermediate tensor allocated
x = ttnn.add(x, gated)                 # second op + dealloc
```

**After:**
```python
x = ttnn.addcmul(x, normed, gate)      # x + normed * gate in one kernel
```

Applied at both attention and FFN gated residual paths in all modulated blocks:
32 blocks with modulation × 2 paths = **64 fused locations**. Small per-op savings that
compound across the 64 call sites.

**Files changed:** `model_ttnn.py` — `TransformerBlock.__call__()` modulated path, two locations.

### 5. SDPA compute config (+PCC, speed neutral)

Passed the optimized compute kernel config to `scaled_dot_product_attention`, which previously
used the TTNN default.

```python
attn_out = ttnn.transformer.scaled_dot_product_attention(
    q, k, v, ...,
    compute_kernel_config=COMPUTE_CONFIG,  # added
)
```

The HiFi2 + packer_l1_acc config improves SDPA accuracy without measurable speed impact.

**Files changed:** `model_ttnn.py` — `Attention.__call__()` SDPA call.

## What Didn't Work

### Fused QKV projection (no benefit)

Concatenated Q/K/V weights into one `[11520, 3840]` matrix for a single matmul, then split
the output with 3 slices. Result: 12,493ms vs 12,480ms — within noise.

The dispatch overhead savings from 3→1 kernel launches are negligible at these tensor sizes
(the hardware saturates all cores with individual [3648, 3840] × [3840, 3840] matmuls), and
the 3 slice operations to split the fused output add their own overhead.

Reverted to keep cleaner code.

### tt_dit library ops (from earlier investigation)

Several tt_dit ops were evaluated but found inapplicable or unhelpful for single-chip Z-Image:
- **minimal_matmul**: No optimized blocking configs for Z-Image shapes; +0.5% slower
- **rotary_embedding_llama**: Incompatible cos/sin format (3-axis vs interleaved)
- **dit_minimal_matmul_addcmul_fused**: Z-Image has rms_norm between matmul and gate
- **joint_scaled_dot_product_attention**: Sequence already concatenated

See `tt_dit_ops_report.md` for full details.

## Cumulative Impact Breakdown

Starting from the 15,336ms baseline and applying optimizations incrementally:

```
15,336 ms  baseline (HiFi4, no packer_l1, F32 RoPE)
15,092 ms  + compute config (HiFi2, packer_l1)         -244 ms  (-1.6%)
14,869 ms  + nlp_create_qkv_heads + addcmul             -223 ms  (-1.5%)
14,864 ms  + SDPA compute config                          -5 ms  (-0.0%)
12,480 ms  + BF16 RoPE                                -2,384 ms  (-16.0%)
───────────────────────────────────────────────────────────────────────
           Total                                      -2,856 ms  (-18.6%)
```

The BF16 RoPE change dominates — it accounts for 83% of the total improvement. The compute
config and op fusion changes contribute the remaining 17%.
