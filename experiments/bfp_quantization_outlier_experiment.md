# BFP Quantization Outlier Experiment

## Goal

Investigate how outlier values affect accuracy in block floating point (BFP) quantization formats (bfp8 and bfp4) on Tenstorrent hardware using `ttnn`.

## Background: Block Floating Point Formats

Both `bfloat8_b` and `bfloat4_b` are **block floating point** formats:
- Tensors are split into consecutive blocks of **16 floats**
- All 16 floats in a block **share a single 8-bit exponent**
- Each float has its own sign bit and mantissa (8 bits for bfp8, 4 bits for bfp4)

This shared exponent creates a fundamental tradeoff: when one value in a block has a much larger magnitude than the others, the shared exponent must accommodate the outlier. This pushes smaller values toward zero (they lose precision because their mantissa bits can't represent the gap between the shared exponent's scale and their actual value).

## Experiment Design

1. Create a 32x32 `bfloat16` PyTorch tensor with values drawn from uniform distribution [0, 1)
2. Inject outliers at fixed positions: in each row `i`, multiply `tensor[i, 8]` and `tensor[i, 24]` by a configurable `outlier_magnitude`. This ensures each block of 16 consecutive floats contains exactly one outlier
3. Transfer the tensor to the TT device via `ttnn.from_torch`
4. Use `ttnn.typecast` to create `bfloat8_b` and `bfloat4_b` versions
5. Transfer quantized tensors back to PyTorch via `ttnn.to_torch`
6. Compare: compute average absolute difference between the original tensor and each quantized version
7. Separately report error for outlier positions vs non-outlier positions

## Configurable Parameters

- `--outlier-magnitude`: Multiplier for outlier elements (default: 5.0)
- `--seed`: Random seed for reproducibility (default: 42)

## Expected Outcome

- bfp4 should show more error than bfp8 (fewer mantissa bits)
- Larger outlier magnitudes should increase error for non-outlier elements in the same block
- Outlier elements themselves should be relatively well-preserved (the shared exponent is driven by them)

## How to Run

```bash
python experiments/bfp_quantization_outlier_experiment.py --outlier-magnitude 5
python experiments/bfp_quantization_outlier_experiment.py --outlier-magnitude 20
```

## Files

- `experiments/bfp_quantization_outlier_experiment.md` - This document
- `experiments/bfp_quantization_outlier_experiment.py` - The experiment script
