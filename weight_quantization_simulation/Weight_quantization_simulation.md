# TT BFP8/BFP4 Weight Quantization Simulation (CPU / PyTorch)

This document explains the purpose, scope, and implementation approach of the `quantize_model.py` effort.
It is intended to live in the repo as design/context so it can be extended in the future.

---

## 1. Goal

We want to **simulate Tenstorrent Block Floating Point (BFP)** weight quantization on the host (CPU) while still working in standard PyTorch.

Tenstorrent devices can store weights in:
- **BFP8_B (bfp8)**: 8-bit shared exponent per block of 16 values, and per-value (sign + 7 mantissa bits)
- **BFP4_B (bfp4)**: 8-bit shared exponent per block of 16 values, and per-value (sign + 3 mantissa bits)

However, on device the compute pipeline uses **16-bit math**, so BFP weights are **unpacked into BF16** (or BF16-like scalars) before multiplications.
Therefore, for host simulation we do:

> **BF16/FP32 weights → simulate BFP quantization → reconstruct unpacked BF16-like values**
> (the reconstructed tensor is stored on host as `torch.bfloat16`)

This allows end-to-end accuracy testing in PyTorch without needing TT hardware memory formats.

---

## 2. Scope: Quantize only matmul / linear weights

This effort is intentionally narrow:
- We only quantize **weights used for matmul-like operations**:
  - `torch.nn.Linear.weight`
  - (optionally) `transformers.pytorch_utils.Conv1D.weight` (used in some HF GPT architectures)
- We do **not** quantize:
  - embeddings (`nn.Embedding`)
  - norm weights (LayerNorm / RMSNorm)
  - biases
  - any other parameters or buffers

This matches the typical “post training weight quantization” workflow for LLM inference where the main memory pressure is in matmul weights.

---

## 3. High-level flow

1. **Load a model** (preferably HuggingFace `AutoModelForCausalLM`)
2. **Identify exactly which parameters correspond to matmul/linear weights**
3. For each such tensor:
   - Convert to float32 (matching TT conversion rules)
   - Apply BFP conversion:
     - partition into blocks of 16 consecutive scalars
     - compute shared exponent per block as the maximum exponent among the 16 values
     - align mantissas to that shared exponent
     - reduce mantissa precision (BFP8 or BFP4)
     - flush denormals to zero
   - **Option A reconstruction**:
     - rebuild float32 values from sign + shared exponent + reduced mantissa bits
   - cast reconstructed values to BF16 and store back
4. Save a new checkpoint or a HF model directory

---

## 4. Block rule (shared exponent grouping)

Tenstorrent’s BFP formats share an exponent across **16 consecutive values** (a “row” inside a 16×16 face).

For this simulation we need a deterministic rule for generic PyTorch tensors. The current policy is:

- Choose an axis (default: **last dimension**) for contiguous grouping
- Partition that axis into **chunks of 16**
- Each chunk of 16 values shares one exponent = max exponent in the chunk

### Remainder handling
If the chosen axis length is not a multiple of 16:
- The simulation **pads with zeros** up to the next multiple of 16,
- performs quantization,
- then unpads back to the original shape.

Padding with zeros is chosen because zeros do not increase the shared exponent and thus typically do not worsen quantization.

---

## 5. Conversion behavior (what we emulate)

The simulator follows the tt-metal algorithm structure:

### 5.1 Input conversion
- All inputs are converted to float32 before any bit-level work.

### 5.2 Float32 decomposition
Each value is interpreted as IEEE754 float32 and decomposed into:
- sign bit
- 8-bit exponent
- 23-bit mantissa

### 5.3 Denormal / zero handling
- If `exp == 0`, the value is treated as **zero** (flush denormals to 0).
- This matches tt-metal’s "zero or denormal -> 0" rule.

### 5.4 Shared exponent selection
- For each block of 16, shared exponent is:
  - the maximum exponent across those 16 values
  - invalid (denormal/zero) values are treated as exponent 0 so they won’t dominate the max

### 5.5 Mantissa alignment (relative denormalization)
- Mantissa is extended with the hidden bit (24-bit `1.mantissa`)
- If a value’s exponent is smaller than the shared exponent:
  - its mantissa is right-shifted by the exponent difference

This is the key place where **outliers** can cause precision loss:
- One large outlier increases the shared exponent,
- forcing many smaller values to shift right and lose bits.

### 5.6 Mantissa reduction (BFP8 vs BFP4)
After alignment, mantissa is reduced:

- **BFP8** keeps **7 bits total** (including hidden bit)
- **BFP4** keeps **3 bits total** (including hidden bit)

The simulator supports:
- **RNE rounding**: round-to-nearest, ties-to-even (default)
- **Truncation**: simple bit truncation (optional)

### 5.7 Option A reconstruction (bit-faithful unpack)
This is the most important choice for accuracy and faithfulness.

Instead of reconstructing numerically via scaling formulas, we rebuild the IEEE float32 bit-pattern:

- exponent field is set to the **shared exponent**
- mantissa field is set using the reduced mantissa bits
  - the hidden bit is removed (since IEEE stores only fractional bits)
  - fractional bits are placed into the top of the 23-bit mantissa field
- sign bit is applied
- if reduced mantissa becomes 0, the sign is cleared and output becomes 0

Finally, reconstructed float32 values are cast to BF16 to mimic the device "unpack to bf16 before math".

---

## 6. Why this is “close to TT” but not identical to full packing

The simulation is designed to match the *numerical* behavior of TT’s shared exponent + mantissa reduction.

However, it does **not** implement:
- TT’s tile layout (32×32 tiles, 16×16 faces)
- exponent packing into dwords
- mantissa packing into dwords

Those details matter for memory layout and binary equivalence, but they do **not** change the reconstructed numeric values if the shared exponent grouping and mantissa rules are the same.

If we later need bit-exact validation against TT packed tensors, we can extend this effort to implement full tile packing.

---

## 7. How linear weights are selected

Instead of heuristic name matching (e.g., `q_proj`, `k_proj`, etc.), we identify quantizable weights by walking the module graph:

- if a module is `nn.Linear`, we select `.weight`
- if a module is HF `Conv1D`, we select `.weight`

This approach is:
- robust to naming changes
- robust to tied weights (same parameter referenced in multiple places)
- safe (does not accidentally quantize embeddings/norms)

---

## 8. Outputs and saving

The “quantized” output checkpoint is still a normal PyTorch/HF checkpoint:
- weights are stored as BF16 tensors
- but their values are already degraded as if they were stored in BFP8/BFP4 and then unpacked

This allows:
- running standard HF evaluation scripts
- doing apples-to-apples comparisons of accuracy before/after BFP simulation

---

## 9. Expected next extensions

Likely follow-up work in this repo:
1. Add name-based fallback filtering if HF module graph is unavailable
2. Add per-tensor metrics:
   - block exponent histograms
   - outlier influence scores
   - error distributions (p50/p90/p99 |err|)
3. Extend grouping policy to match TT tiling more closely for 2D weights
4. Add correctness tests vs a scalar reference implementation
5. Integrate into an evaluation harness (perplexity, task accuracy, etc.)

---

## 10. Summary

This effort provides a practical, TT-faithful simulation of:
- shared exponent per 16 values
- mantissa reduction to BFP8/BFP4
- denormal flush to zero
- rounding behavior
- bit-level reconstruction back to float32, then BF16

And applies it **only** to linear/matmul weights, enabling post-training quantization experiments and end-to-end accuracy testing in CPU PyTorch.
