# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Const-eval transformations for the Z-Image text encoder (Qwen3) TTNN model.

There are 5 const-eval patterns:
  0: Causal attention mask [1,1,512,512] BF16 (lower-triangular bool)
  1: Zero tensor [1,1,512,512] F32 (attend value for where())
  2: Move embedding weights to device DRAM
  3: Negative infinity tensor [1,1,512,512] F32 (mask-out value for where())
  4: RoPE cos/sin from inv_freq [64] -> [1,1,512,128] BF16
"""

import ttnn

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

SEQ_LEN = 512
HEAD_DIM = 128
HALF_HEAD = HEAD_DIM // 2  # 64
NEG_INF_VALUE = -3.3895313892515355e38  # ~ -finfo(f32).max


# ---------------------------------------------------------------------------
# Pattern 0: Causal mask [1,1,512,512] BF16
# ---------------------------------------------------------------------------

def build_causal_mask(device):
    """Build a [1,1,512,512] BF16 causal boolean mask (lower-triangular).

    row_i >= col_j => 1.0, else 0.0

    Codegen: main_const_eval_0 (lines 7-593)
    """
    # Position indices [0..511]
    indices_data = list(range(SEQ_LEN))
    indices = ttnn.Tensor(
        indices_data,
        [SEQ_LEN],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        DRAM,
    )

    # Zero matrix [512, 512]
    zero_mat = ttnn.full(
        shape=ttnn.Shape([SEQ_LEN, SEQ_LEN]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )

    # row indices [1,1,1,512] and col indices [1,1,512,1]
    row = ttnn.reshape(indices, [1, 1, 1, SEQ_LEN], memory_config=DRAM)
    col = ttnn.reshape(indices, [1, 1, SEQ_LEN, 1], memory_config=DRAM)
    ttnn.deallocate(indices, False)

    # row - col: broadcast to [1,1,512,512]
    neg_col = ttnn.neg(col, memory_config=DRAM)
    ttnn.deallocate(col, False)
    row_minus_col = ttnn.add(row, neg_col, memory_config=DRAM)
    ttnn.deallocate(row, False)
    ttnn.deallocate(neg_col, False)

    # Compare: (row - col) <= 0 => causal mask (i >= j)
    zero_4d = ttnn.reshape(zero_mat, [1, 1, SEQ_LEN, SEQ_LEN], memory_config=DRAM)
    ttnn.deallocate(zero_mat, False)
    causal_mask = ttnn.le(
        row_minus_col, zero_4d,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=DRAM,
    )
    ttnn.deallocate(row_minus_col, False)
    ttnn.deallocate(zero_4d, False)

    return causal_mask


# ---------------------------------------------------------------------------
# Pattern 1: Zeros mask [1,1,512,512] F32
# ---------------------------------------------------------------------------

def build_zeros_mask(device):
    """Build a [1,1,512,512] F32 all-zeros tensor.

    Codegen: main_const_eval_1 (lines 596-611)
    """
    scalar = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    result = ttnn.repeat(
        scalar, ttnn.Shape([1, 1, SEQ_LEN, SEQ_LEN]), memory_config=DRAM
    )
    ttnn.deallocate(scalar, False)
    return result


# ---------------------------------------------------------------------------
# Pattern 2: Move embedding table to device
# ---------------------------------------------------------------------------

def move_embed_to_device(embed_weight, device):
    """Move the embedding weight tensor to device DRAM.

    Codegen: main_const_eval_2 (lines 614-624)
    Input: host ROW_MAJOR BF16 tensor [151936, 2560]
    """
    return ttnn.to_device(embed_weight, device=device, memory_config=DRAM)


# ---------------------------------------------------------------------------
# Pattern 3: Negative infinity mask [1,1,512,512] F32
# ---------------------------------------------------------------------------

def build_neg_inf_mask(device):
    """Build a [1,1,512,512] F32 all negative-infinity tensor.

    Codegen: main_const_eval_3 (lines 627-642)
    """
    scalar = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=NEG_INF_VALUE,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    result = ttnn.repeat(
        scalar, ttnn.Shape([1, 1, SEQ_LEN, SEQ_LEN]), memory_config=DRAM
    )
    ttnn.deallocate(scalar, False)
    return result


# ---------------------------------------------------------------------------
# Pattern 4: RoPE cos/sin precomputation
# ---------------------------------------------------------------------------

def build_rope_cos_sin(inv_freq, device):
    """Precompute RoPE cos/sin tables for all 512 positions, head_dim=128.

    Codegen: main_const_eval_4 (lines 645-1264)

    Input: inv_freq [64] host ROW_MAJOR FLOAT32
    Returns: (rope_cos, rope_sin) each [1,1,512,128] BF16

    Steps:
      1. Move inv_freq to device, tile it
      2. Create position indices [0..511] as float
      3. Outer product: positions [1,64,1] @ inv_freq reshaped [1,1,512] -> [1,64,512]
         Actually: inv_freq [1,64,1] @ positions [1,1,512] -> [1,64,512]
      4. Permute -> [1,512,64]
      5. Reshape -> [1,1,512,64]
      6. Concat [above, above] dim=3 -> [1,1,512,128]
      7. cos/sin -> typecast BF16
    """
    # Move inv_freq to device and tile
    inv_freq_dev = ttnn.to_device(inv_freq, device=device, memory_config=DRAM)
    inv_freq_tiled = ttnn.to_layout(
        inv_freq_dev, ttnn.Layout.TILE, None, memory_config=None
    )

    # Position indices as float [1, 1, 512]
    pos_data = list(range(SEQ_LEN))
    positions = ttnn.Tensor(
        [float(x) for x in pos_data],
        [1, 1, SEQ_LEN],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        DRAM,
    )

    # Reshape inv_freq [64] -> [1, 64, 1] for outer product
    inv_freq_3d = ttnn.reshape(
        inv_freq_tiled, [1, HALF_HEAD, 1], memory_config=DRAM
    )
    ttnn.deallocate(inv_freq_tiled, False)

    # Outer product: [1, 64, 1] @ [1, 1, 512] -> [1, 64, 512]
    freqs = ttnn.matmul(
        inv_freq_3d, positions,
        memory_config=DRAM,
    )
    ttnn.deallocate(inv_freq_3d, False)
    ttnn.deallocate(positions, False)

    # Permute [1, 64, 512] -> [1, 512, 64]
    freqs = ttnn.permute(freqs, [0, 2, 1], memory_config=DRAM, pad_value=0.0)

    # Reshape to [1, 1, 512, 64]
    freqs = ttnn.reshape(
        freqs, [1, 1, SEQ_LEN, HALF_HEAD], memory_config=DRAM
    )

    # Concat to full head_dim: [1, 1, 512, 128]
    freqs_full = ttnn.concat([freqs, freqs], 3, memory_config=DRAM)
    ttnn.deallocate(freqs, False)

    # Compute cos and sin
    rope_cos_f32 = ttnn.cos(freqs_full, memory_config=DRAM)
    rope_sin_f32 = ttnn.sin(freqs_full, memory_config=DRAM)
    ttnn.deallocate(freqs_full, False)

    # Cast to BF16
    rope_cos = ttnn.typecast(rope_cos_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
    rope_sin = ttnn.typecast(rope_sin_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
    ttnn.deallocate(rope_cos_f32, False)
    ttnn.deallocate(rope_sin_f32, False)

    return rope_cos, rope_sin


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_const_evals(params, device):
    """Run all const-eval transformations and return derived tensors.

    Args:
        params: dict mapping parameter name -> ttnn tensor (from load_params_from_pytorch).
                Must contain "embed_tokens.weight" and "rotary_emb.inv_freq" on host.
        device: ttnn device handle.

    Returns:
        dict with keys:
          "causal_mask":    [1,1,512,512] BF16 lower-triangular mask
          "zeros_mask":     [1,1,512,512] F32 all zeros
          "embed_table":    embedding weight on device
          "neg_inf_mask":   [1,1,512,512] F32 all -inf
          "rope_cos":       [1,1,512,128] BF16
          "rope_sin":       [1,1,512,128] BF16
    """
    result = {}

    result["causal_mask"] = build_causal_mask(device)
    result["zeros_mask"] = build_zeros_mask(device)
    result["embed_table"] = move_embed_to_device(params["embed_tokens.weight"], device)
    result["neg_inf_mask"] = build_neg_inf_mask(device)

    rope_cos, rope_sin = build_rope_cos_sin(params["rotary_emb.inv_freq"], device)
    result["rope_cos"] = rope_cos
    result["rope_sin"] = rope_sin

    return result
