# DeepSeek V4 Flash Model Bringup Guide

This document captures tips and instructions for bringing up the DeepSeek V4 Flash model on TT-XLA.

## Model Overview

DeepSeek V4 Flash uses several novel attention mechanisms:
- **CSA (Compressed Sparse Attention)**: `compress_ratio=4` with Indexer for top-k selection
- **HSA (Heavily Compressed Attention)**: `compress_ratio=128` without Indexer
- **Sliding Window Attention**: `compress_ratio=0` (baseline)

The `compress_ratios` tuple in `ModelArgs` defines which attention type each layer uses:
```python
compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
```

## Files

- `model.py` - Original DeepSeek V4 Flash model from HuggingFace (requires CUDA kernels)
- `kernel.py` - Original CUDA kernels using tilelang (requires CUDA)
- `kernel_stubs.py` - Stub implementations for CUDA kernels (no-ops for BF16 testing)
- `modified_model.py` - Adapted model with CUDA dependencies removed for TT-XLA testing
- `test_basic.py` - Basic module tests (ParallelEmbedding, Linear)
- `test_compressor.py` - Compressor module tests (uses kernel_stubs.py)
- `test_ds4_attention.py` - Attention block tests (CSA, HSA)

## Bringup Tips

### 1. Initialize Weights Before Testing

The original model uses `torch.empty()` for parameters, which creates **uninitialized memory** containing garbage values (NaN, inf, denormals).

**Problem**: CPU and TT hardware interpret uninitialized memory differently, causing mismatches like CPU showing `nan` where TT shows `inf`.

**Solution**: Always initialize weights before testing:

```python
import torch.nn as nn

def init_weights(module):
    """Initialize weights with random values to avoid NaN/inf from uninitialized memory."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
    # Initialize Compressor's APE (Absolute Positional Encoding) parameter
    if hasattr(module, 'ape') and module.ape is not None:
        nn.init.normal_(module.ape, mean=0.0, std=0.02)

# Apply to model
model.apply(init_weights)
```

### 2. Use BF16 for Testing

The original model supports FP8 and FP4 quantization via custom CUDA kernels. For TT-XLA testing, use BF16:

```python
# When creating Linear layers, explicitly specify bf16
linear = Linear(in_features, out_features, dtype=torch.bfloat16)

# Or modify ModelArgs
args = ModelArgs(dtype="bf16", scale_dtype="fp32")
```

**Important**: Some modules like `Compressor` have internal float32 layers (`wkv`, `wgate`) and handle dtype conversion in their forward method. Do NOT call `.to(bfloat16)` on these modules - it will cause dtype mismatch errors. The module handles conversion internally:

```python
# WRONG - will cause dtype mismatch
compressor = Compressor(args, ...)
compressor = compressor.to(torch.bfloat16)  # Don't do this!

# CORRECT - let the module handle dtypes
compressor = Compressor(args, ...)
compressor.apply(init_weights)
compressor.eval()
# Input can still be bfloat16 - forward() converts to float32 internally
```

### 3. CUDA Kernel Dependencies

The original `model.py` imports from `kernel.py`:
- `act_quant` - FP8 activation quantization (in-place quant+dequant for BF16 simulation)
- `fp4_act_quant` - FP4 activation quantization
- `fp8_gemm` - FP8 matrix multiplication
- `fp4_gemm` - FP4 matrix multiplication
- `sparse_attn` - Sparse attention kernel
- `hc_split_sinkhorn` - Hyper-Connections Sinkhorn normalization

**Two approaches for handling CUDA dependencies:**

#### Option A: Use kernel_stubs.py (Recommended for testing individual modules)

Install stubs BEFORE importing from model.py:

```python
# At the top of your test file, BEFORE any model imports:
import kernel_stubs
kernel_stubs.install()

# Now model.py will use stub implementations
from model import Compressor, ...
```

The stubs make CUDA kernel functions no-ops:
- `act_quant` with `inplace=True` does nothing (tensor unchanged)
- `act_quant` with `inplace=False` returns `(x, None)`
- Same pattern for `fp4_act_quant`

This allows testing the core computation logic without FP8/FP4 quantization simulation.

#### Option B: Use modified_model.py (For full attention testing)

For TT-XLA testing of complete attention blocks, `modified_model.py` provides:
- Stub implementations built directly into the model
- `scipy.linalg.hadamard` instead of `fast_hadamard_transform`
- Pure PyTorch sparse attention implementation

### 4. Hadamard Transform

The original model uses `fast_hadamard_transform` (CUDA-only). The modified model uses `scipy.linalg.hadamard`:

```python
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    import scipy.linalg
    hidden_size = x.size(-1)
    hadamard = torch.tensor(
        scipy.linalg.hadamard(hidden_size),
        dtype=torch.bfloat16,
        device=x.device
    ) * (hidden_size ** -0.5)
    return F.linear(x, hadamard)
```

### 5. Test Incrementally

Start with simple modules and build up:

1. **ParallelEmbedding** - Simple lookup, should have PCC ≈ 0.99
2. **Linear** - Basic matmul, should have PCC ≈ 0.99
3. **RMSNorm** - Normalization layer
4. **Compressor** - KV cache compression
5. **Indexer** - Top-k selection (CSA only)
6. **Attention** - Full attention block

### 6. Dimension Naming Convention

- `_mini` suffix: Reduced dimensions for fast testing
- No suffix: Full DeepSeek V4 dimensions

Example:
```python
def test_linear_mini(...)     # in=256, out=512
def test_linear(...)          # in=4096, out=1024
```

## Running Tests

```bash
# Run all basic tests (no CUDA dependencies)
pytest -svv ds4/test_basic.py

# Run compressor tests (uses kernel_stubs.py)
pytest -svv ds4/test_compressor.py

# Run specific test
pytest -svv ds4/test_basic.py::test_parallel_embedding_mini

# Run with specific batch size
pytest -svv ds4/test_basic.py::test_linear_mini[1-16]
```

## Model Dimensions (Full Size)

From `ModelArgs` defaults:
- `vocab_size`: 129280
- `dim`: 4096
- `n_heads`: 64
- `head_dim`: 512
- `q_lora_rank`: 1024
- `o_lora_rank`: 1024
- `rope_head_dim`: 64
- `window_size`: 128
- `index_n_heads`: 64
- `index_head_dim`: 128
- `index_topk`: 512

Note from previous bringup: - we want to do the same thing.
# This model is modified from the original deepseek_v3_2_exp model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
#    - fast_hadamard_transform requires a CUDA enviroment and fails to install
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
#    - the original implementation (kernel.py) relies on custom tilelang kernels not supported on TT
# 3. Avoid torch.view_as_complex/view_as_real operations

## References

- Model source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference
- DeepSeek V4 paper: (link to paper when available)
