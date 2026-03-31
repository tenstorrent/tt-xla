# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Deduplicated const-eval transformations for the Z-Image transformer TTNN model.

Each function prepares a weight tensor for use on device. The codegen main.py
has 85+ const_eval functions that fall into a small number of patterns. This
module deduplicates them into reusable helpers plus a single entry point
`run_const_evals(weights, device)` that returns a dict of prepared tensors.
"""

import ttnn

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


# ---------------------------------------------------------------------------
# Pattern 0: 2D weight matrix  (to_device -> TILE -> permute([1,0]) -> FLOAT32)
# Used for: all weight matrices going through ttnn.linear (adaLN, embedder, etc.)
# ---------------------------------------------------------------------------

def prepare_weight_matrix(tensor, device):
    """Transpose and cast a 2D weight matrix for TTNN linear with transpose_b=False.

    Codegen pattern: to_device -> to_layout(TILE) -> permute([1,0]) -> typecast(FLOAT32)
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.permute(t, [1, 0], memory_config=DRAM, pad_value=0.0)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern 1: 1D bias  (to_device -> TILE -> FLOAT32)
# Used for: biases of adaLN, MLP, embedders
# ---------------------------------------------------------------------------

def prepare_bias(tensor, device):
    """Cast a 1D tensor to FLOAT32 on device.

    Codegen pattern: to_device -> to_layout(TILE) -> typecast(FLOAT32)
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern 2: Attention mask construction
# (to_device -> TILE -> reshape [1,1,1,seq] -> where(0, -inf) -> repeat [1,1,seq,1])
# Used for: x_attn_mask (3616), cap_attn_mask (32), unified_attn_mask (3648)
# ---------------------------------------------------------------------------

def prepare_attn_mask(tensor, device, seq_len):
    """Build a broadcastable attention mask from a boolean-like 1D mask.

    Codegen pattern: to_device -> to_layout(TILE) -> reshape [1,1,1,seq] ->
    full(-inf) -> full(0) -> where(mask, 0, -inf) -> repeat [1,1,seq,1]
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)

    neg_inf = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    zero = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )

    t = ttnn.reshape(t, [1, 1, 1, seq_len], memory_config=DRAM)
    t = ttnn.where(t, zero, neg_inf, memory_config=DRAM)
    ttnn.deallocate(zero, False)
    ttnn.deallocate(neg_inf, False)
    t = ttnn.repeat(t, ttnn.Shape([1, 1, seq_len, 1]), memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern 3: Scalar ones  (full([1,1], 1.0))
# Used for: adding 1.0 to adaLN scale output (scale = 1 + linear(adaln_input))
# ---------------------------------------------------------------------------

def prepare_scalar_one(device):
    """Create a [1,1] tensor filled with 1.0 in BFLOAT16.

    Codegen: ttnn.full(shape=[1,1], fill_value=1.0, dtype=BFLOAT16, layout=TILE)
    """
    return ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )


# ---------------------------------------------------------------------------
# Pattern 4: Buffer reshape  (to_device -> TILE -> reshape)
# Used for: cap_pad_mask [32] -> [32,1]
# ---------------------------------------------------------------------------

def prepare_reshape_buffer(tensor, device, target_shape):
    """Reshape a buffer to a target shape on device.

    Codegen pattern: to_device -> to_layout(TILE) -> reshape(target_shape)
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, target_shape, memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern 5: Bias broadcast  (to_device -> TILE -> reshape -> typecast -> repeat)
# Used for: x_embedder.bias [3840] -> [1,3840] -> repeat [3616,1]
#           final_layer.linear.bias [64] -> [1,64] -> repeat [3648,1]
#           cap_embedder.1.bias -> broadcast
# ---------------------------------------------------------------------------

def prepare_bias_broadcast(tensor, device, reshape_to, repeat_shape):
    """Reshape a bias and broadcast via repeat.

    Codegen pattern: to_device -> to_layout(TILE) -> reshape -> typecast(FLOAT32) -> repeat
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, reshape_to, memory_config=DRAM)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    t = ttnn.repeat(t, ttnn.Shape(repeat_shape), memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern 6: Freqs reshape + typecast  (to_device -> TILE -> reshape -> typecast)
# Used for: t_embedder.freqs [128] -> [1, 128] -> FLOAT32
# ---------------------------------------------------------------------------

def prepare_freqs(tensor, device):
    """Prepare the timestep frequency table: reshape to [1, 128] and cast to FLOAT32.

    Codegen: main_const_eval_20 on input[8] (t_embedder.freqs)
    """
    t = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.reshape(t, [1, 128], memory_config=DRAM)
    t = ttnn.typecast(t, ttnn.DataType.FLOAT32, memory_config=DRAM)
    return t


# ---------------------------------------------------------------------------
# Pattern: LayerNorm epsilon scalar
# ---------------------------------------------------------------------------

def prepare_layernorm_eps(device):
    """Create a [1,1,1] scalar with LayerNorm epsilon = 1e-6.

    Codegen: main_const_eval_21 -> full([1,1,1], 9.9999999747524271e-07, FLOAT32)
    """
    return ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=9.9999999747524271e-07,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )


# ---------------------------------------------------------------------------
# Pattern 7: RoPE precomputation  (main_const_eval_22)
#
# Takes 8 inputs:
#   cap_pos_ids   [32, 3]   (si32)
#   sin_2         [512, 24]  (bf16)
#   sin_1         [512, 24]  (bf16)
#   sin_0         [1536, 16] (bf16)
#   cos_2         [512, 24]  (bf16)
#   cos_1         [512, 24]  (bf16)
#   cos_0         [1536, 16] (bf16)
#   image_pos_ids [3616, 3]  (si32)
#
# Returns 6 tensors:
#   image_rope_cos  [1, 3616, 1, 64, 1] FLOAT32
#   image_rope_sin  [1, 3616, 1, 64, 1] FLOAT32
#   cap_rope_cos    [1, 32, 1, 64, 1]   FLOAT32
#   cap_rope_sin    [1, 32, 1, 64, 1]   FLOAT32
#   unified_rope_cos [1, 3648, 1, 64, 1] FLOAT32
#   unified_rope_sin [1, 3648, 1, 64, 1] FLOAT32
# ---------------------------------------------------------------------------

def _prepare_pos_indices(pos_ids_layout, axis_col, seq_len, table_size, fill_zero, fill_table_size, device):
    """Extract one axis column from position IDs, handle negative wrap, prepare for embedding.

    For each axis column in pos_ids [seq, 3]:
      - slice column -> reshape [seq]
      - gt(0, ids) to find negatives -> where(neg, ids + table_size, ids)
      - reshape [seq, 1] -> typecast UINT32 -> to_layout ROW_MAJOR

    Returns a ROW_MAJOR UINT32 index tensor ready for ttnn.embedding.
    """
    col = ttnn.slice(
        pos_ids_layout, [0, axis_col], [seq_len, axis_col + 1], [1, 1],
        memory_config=DRAM,
    )
    col = ttnn.reshape(col, [seq_len], memory_config=DRAM)

    is_neg = ttnn.gt(fill_zero, col, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    wrapped = ttnn.add(col, fill_table_size, dtype=ttnn.DataType.INT32, memory_config=DRAM)

    is_neg_f = ttnn.typecast(is_neg, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(is_neg, False)
    wrapped_f = ttnn.typecast(wrapped, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(wrapped, False)
    col_f = ttnn.typecast(col, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(col, False)

    result = ttnn.where(is_neg_f, wrapped_f, col_f, memory_config=DRAM)
    ttnn.deallocate(col_f, False)
    ttnn.deallocate(wrapped_f, False)
    ttnn.deallocate(is_neg_f, False)

    result = ttnn.typecast(result, ttnn.DataType.INT32, memory_config=DRAM)
    result = ttnn.reshape(result, [seq_len, 1], memory_config=DRAM)
    result = ttnn.typecast(result, ttnn.DataType.UINT32, memory_config=DRAM)
    result = ttnn.to_layout(result, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    return result


def _build_rope_for_seq(
    pos_ids_tiled, seq_len,
    cos_0_dev, sin_0_dev, cos_1_dev, sin_1_dev, cos_2_dev, sin_2_dev,
    fill_zero, fill_1536, fill_512, device,
):
    """Build cos/sin rope embeddings for one sequence (image or caption).

    Returns (cos_half, sin_half, rope_full) where:
      cos_half: [1, seq, 1, 64, 1] FLOAT32
      sin_half: [1, seq, 1, 64, 1] FLOAT32
      rope_full: [1, seq, 128] BFLOAT16 (kept for unified concat)
    """
    # Extract index tensors for each axis
    idx_0 = _prepare_pos_indices(pos_ids_tiled, 0, seq_len, 1536, fill_zero, fill_1536, device)
    idx_1 = _prepare_pos_indices(pos_ids_tiled, 1, seq_len, 512, fill_zero, fill_512, device)
    idx_2 = _prepare_pos_indices(pos_ids_tiled, 2, seq_len, 512, fill_zero, fill_512, device)

    # Embedding lookups
    emb_cos_0 = ttnn.embedding(idx_0, cos_0_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    emb_cos_1 = ttnn.embedding(idx_1, cos_1_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    emb_cos_2 = ttnn.embedding(idx_2, cos_2_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    emb_sin_0 = ttnn.embedding(idx_0, sin_0_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    emb_sin_1 = ttnn.embedding(idx_1, sin_1_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
    emb_sin_2 = ttnn.embedding(idx_2, sin_2_dev, padding_idx=None, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

    ttnn.deallocate(idx_0, False)
    ttnn.deallocate(idx_1, False)
    ttnn.deallocate(idx_2, False)

    # Reshape to [1, seq, dim_half] for concatenation
    emb_cos_0 = ttnn.reshape(emb_cos_0, [1, seq_len, 16], memory_config=DRAM)
    emb_cos_1 = ttnn.reshape(emb_cos_1, [1, seq_len, 24], memory_config=DRAM)
    emb_cos_2 = ttnn.reshape(emb_cos_2, [1, seq_len, 24], memory_config=DRAM)
    emb_sin_0 = ttnn.reshape(emb_sin_0, [1, seq_len, 16], memory_config=DRAM)
    emb_sin_1 = ttnn.reshape(emb_sin_1, [1, seq_len, 24], memory_config=DRAM)
    emb_sin_2 = ttnn.reshape(emb_sin_2, [1, seq_len, 24], memory_config=DRAM)

    # Concatenate: [cos_0, cos_1, cos_2, sin_0, sin_1, sin_2] along dim 2
    # Result: [1, seq, 128] where [:64] = cos half, [64:] = sin half
    rope_full = ttnn.concat(
        [emb_cos_0, emb_cos_1, emb_cos_2, emb_sin_0, emb_sin_1, emb_sin_2],
        2, memory_config=DRAM,
    )
    for t in [emb_cos_0, emb_cos_1, emb_cos_2, emb_sin_0, emb_sin_1, emb_sin_2]:
        ttnn.deallocate(t, False)

    # Split into cos [0:64] and sin [64:128], reshape for broadcasting
    # rope_full layout: [cos_0, cos_1, cos_2, sin_0, sin_1, sin_2]
    #   indices [0:64]   = cos half (16 + 24 + 24 = 64)
    #   indices [64:128] = sin half (16 + 24 + 24 = 64)
    cos_half = ttnn.slice(rope_full, [0, 0, 0], [1, seq_len, 64], [1, 1, 1], memory_config=DRAM)
    cos_half = ttnn.reshape(cos_half, [1, seq_len, 1, 64, 1], memory_config=DRAM)
    cos_half = ttnn.typecast(cos_half, ttnn.DataType.FLOAT32, memory_config=DRAM)

    sin_half = ttnn.slice(rope_full, [0, 0, 64], [1, seq_len, 128], [1, 1, 1], memory_config=DRAM)
    sin_half = ttnn.reshape(sin_half, [1, seq_len, 1, 64, 1], memory_config=DRAM)
    sin_half = ttnn.typecast(sin_half, ttnn.DataType.FLOAT32, memory_config=DRAM)

    return cos_half, sin_half, rope_full


def prepare_rope_embeddings(
    cap_pos_ids, sin_2, sin_1, sin_0, cos_2, cos_1, cos_0, image_pos_ids, device
):
    """Precompute all RoPE cos/sin embeddings for image, caption, and unified sequences.

    This is the deduplicated version of main_const_eval_22 (lines 658-1776 of codegen).

    The function performs embedding lookups on the cos/sin tables for each of the 3 RoPE
    axes, concatenates them to form [seq, 128] cos and sin vectors, then splits into
    first-half (sin) [0:64] and second-half (cos) [64:128], reshapes to [1, seq, 1, 64, 1]
    for broadcasting during the RoPE application.

    Returns:
        Tuple of 6 tensors:
          (image_rope_sin, image_rope_cos, cap_rope_sin, cap_rope_cos,
           unified_rope_sin, unified_rope_cos)
        Each shaped [1, seq, 1, 64, 1] in FLOAT32.
    """
    # Put lookup tables on device
    cos_0_dev = ttnn.to_device(cos_0, device=device, memory_config=DRAM)
    sin_0_dev = ttnn.to_device(sin_0, device=device, memory_config=DRAM)
    cos_1_dev = ttnn.to_device(cos_1, device=device, memory_config=DRAM)
    sin_1_dev = ttnn.to_device(sin_1, device=device, memory_config=DRAM)
    cos_2_dev = ttnn.to_device(cos_2, device=device, memory_config=DRAM)
    sin_2_dev = ttnn.to_device(sin_2, device=device, memory_config=DRAM)

    # Put position IDs on device and tile them
    img_ids = ttnn.to_device(image_pos_ids, device=device, memory_config=DRAM)
    img_ids = ttnn.to_layout(img_ids, ttnn.Layout.TILE, None, memory_config=None)
    cap_ids = ttnn.to_device(cap_pos_ids, device=device, memory_config=DRAM)
    cap_ids = ttnn.to_layout(cap_ids, ttnn.Layout.TILE, None, memory_config=None)

    # Shared scalar fill constants
    fill_zero = ttnn.full(
        shape=ttnn.Shape([1]), fill_value=0,
        dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE,
        device=device, memory_config=DRAM,
    )
    fill_512 = ttnn.full(
        shape=ttnn.Shape([1]), fill_value=512,
        dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE,
        device=device, memory_config=DRAM,
    )
    fill_1536 = ttnn.full(
        shape=ttnn.Shape([1]), fill_value=1536,
        dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE,
        device=device, memory_config=DRAM,
    )

    # Image RoPE: seq=3616
    img_cos, img_sin, img_rope_full = _build_rope_for_seq(
        img_ids, 3616,
        cos_0_dev, sin_0_dev, cos_1_dev, sin_1_dev, cos_2_dev, sin_2_dev,
        fill_zero, fill_1536, fill_512, device,
    )
    ttnn.deallocate(img_ids, False)

    # Caption RoPE: seq=32
    cap_cos, cap_sin, cap_rope_full = _build_rope_for_seq(
        cap_ids, 32,
        cos_0_dev, sin_0_dev, cos_1_dev, sin_1_dev, cos_2_dev, sin_2_dev,
        fill_zero, fill_1536, fill_512, device,
    )
    ttnn.deallocate(cap_ids, False)

    # Cleanup shared resources
    ttnn.deallocate(fill_zero, False)
    ttnn.deallocate(fill_512, False)
    ttnn.deallocate(fill_1536, False)
    ttnn.deallocate(cos_0_dev, False)
    ttnn.deallocate(sin_0_dev, False)
    ttnn.deallocate(cos_1_dev, False)
    ttnn.deallocate(sin_1_dev, False)
    ttnn.deallocate(cos_2_dev, False)
    ttnn.deallocate(sin_2_dev, False)

    # Unified RoPE: concat image + caption rope_full along seq dim
    unified_full = ttnn.concat([img_rope_full, cap_rope_full], 1, memory_config=DRAM)
    ttnn.deallocate(img_rope_full, False)
    ttnn.deallocate(cap_rope_full, False)

    # Split unified into cos/sin halves (same layout as rope_full: [0:64]=cos, [64:128]=sin)
    unified_cos = ttnn.slice(unified_full, [0, 0, 0], [1, 3648, 64], [1, 1, 1], memory_config=DRAM)
    unified_cos = ttnn.reshape(unified_cos, [1, 3648, 1, 64, 1], memory_config=DRAM)
    unified_cos = ttnn.typecast(unified_cos, ttnn.DataType.FLOAT32, memory_config=DRAM)

    unified_sin = ttnn.slice(unified_full, [0, 0, 64], [1, 3648, 128], [1, 1, 1], memory_config=DRAM)
    ttnn.deallocate(unified_full, False)
    unified_sin = ttnn.reshape(unified_sin, [1, 3648, 1, 64, 1], memory_config=DRAM)
    unified_sin = ttnn.typecast(unified_sin, ttnn.DataType.FLOAT32, memory_config=DRAM)

    return img_cos, img_sin, cap_cos, cap_sin, unified_cos, unified_sin


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_const_evals(weights, device):
    """Apply the correct const-eval transformation to each weight in the state dict.

    Args:
        weights: dict mapping weight name -> ttnn tensor (on host, ROW_MAJOR).
        device: ttnn device handle.

    Returns:
        dict mapping weight name -> prepared ttnn tensor (on device, ready for compute).
        Also includes special keys for generated constants:
          "scalar_one", "layernorm_eps",
          "x_attn_mask", "cap_attn_mask", "unified_attn_mask",
          "image_rope_sin", "image_rope_cos",
          "cap_rope_sin", "cap_rope_cos",
          "unified_rope_sin", "unified_rope_cos",
          "x_embedder.bias_broadcast", "final_layer.linear.bias_broadcast",
          "cap_pad_mask_reshaped", "freqs_prepared"
    """
    result = {}

    # --- Weight matrices used with ttnn.linear (Pattern 0): permute + typecast ---
    # These get pre-transposed so ttnn.linear can use transpose_b=False.
    linear_weights = [
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "final_layer.adaLN_modulation.1.weight",
        "x_embedder.weight",
        "cap_embedder.1.weight",
    ]
    for prefix in ["noise_refiner", "layers"]:
        count = 2 if prefix == "noise_refiner" else 30
        for i in range(count):
            linear_weights.append(f"{prefix}.{i}.adaLN_modulation.0.weight")

    for name in linear_weights:
        if name in weights:
            result[name] = prepare_weight_matrix(weights[name], device)

    # --- 1D biases (Pattern 1): typecast only ---
    bias_names = [
        "t_embedder.mlp.0.bias",
        "t_embedder.mlp.2.bias",
        "final_layer.adaLN_modulation.1.bias",
        "cap_embedder.1.bias",
    ]
    for prefix in ["noise_refiner", "layers"]:
        count = 2 if prefix == "noise_refiner" else 30
        for i in range(count):
            bias_names.append(f"{prefix}.{i}.adaLN_modulation.0.bias")

    for name in bias_names:
        if name in weights:
            result[name] = prepare_bias(weights[name], device)

    # --- Attention masks (Pattern 2) ---
    if "x_attn_mask" in weights:
        result["x_attn_mask"] = prepare_attn_mask(weights["x_attn_mask"], device, 3616)
    if "cap_attn_mask" in weights:
        result["cap_attn_mask"] = prepare_attn_mask(weights["cap_attn_mask"], device, 32)
    if "unified_attn_mask" in weights:
        result["unified_attn_mask"] = prepare_attn_mask(weights["unified_attn_mask"], device, 3648)

    # --- Scalar ones (Pattern 3) ---
    result["scalar_one"] = prepare_scalar_one(device)

    # --- Pad mask reshapes (Pattern 4) ---
    if "cap_pad_mask" in weights:
        result["cap_pad_mask"] = prepare_reshape_buffer(weights["cap_pad_mask"], device, [32, 1])
    if "image_pad_mask" in weights:
        result["image_pad_mask"] = prepare_reshape_buffer(weights["image_pad_mask"], device, [3616, 1])

    # --- Bias broadcasts (Pattern 5) ---
    if "x_embedder.bias" in weights:
        result["x_embedder.bias_broadcast"] = prepare_bias_broadcast(
            weights["x_embedder.bias"], device, [1, 3840], [3616, 1]
        )
    if "final_layer.linear.bias" in weights:
        result["final_layer.linear.bias_broadcast"] = prepare_bias_broadcast(
            weights["final_layer.linear.bias"], device, [1, 64], [3648, 1]
        )

    # --- Freqs (Pattern 6) ---
    if "t_embedder.freqs" in weights:
        result["freqs_prepared"] = prepare_freqs(weights["t_embedder.freqs"], device)

    # --- LayerNorm epsilon ---
    result["layernorm_eps"] = prepare_layernorm_eps(device)

    # --- Pad token broadcast for cap ---
    if "cap_pad_token" in weights:
        result["cap_pad_token_broadcast"] = prepare_bias_broadcast(
            weights["cap_pad_token"], device, [1, 3840], [32, 1]
        )

    # --- RoPE embeddings (Pattern 7) ---
    rope_inputs_present = all(
        name in weights for name in [
            "cap_pos_ids", "rope_embedder.sin_2", "rope_embedder.sin_1",
            "rope_embedder.sin_0", "rope_embedder.cos_2", "rope_embedder.cos_1",
            "rope_embedder.cos_0", "image_pos_ids",
        ]
    )
    if rope_inputs_present:
        (
            result["image_rope_cos"],
            result["image_rope_sin"],
            result["cap_rope_cos"],
            result["cap_rope_sin"],
            result["unified_rope_cos"],
            result["unified_rope_sin"],
        ) = prepare_rope_embeddings(
            cap_pos_ids=weights["cap_pos_ids"],
            sin_2=weights["rope_embedder.sin_2"],
            sin_1=weights["rope_embedder.sin_1"],
            sin_0=weights["rope_embedder.sin_0"],
            cos_2=weights["rope_embedder.cos_2"],
            cos_1=weights["rope_embedder.cos_1"],
            cos_0=weights["rope_embedder.cos_0"],
            image_pos_ids=weights["image_pos_ids"],
            device=device,
        )

    return result
