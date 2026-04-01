# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter loading for Z-Image Transformer TTNN model.

Converts a PyTorch state_dict into TTNN tensors with the correct
layout, dtype, and device placement for each parameter.
"""

import ttnn

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)


# ---------------------------------------------------------------------------
# Parameter config: (layout, dtype, on_device) per parameter name.
#
# - TILE/BFLOAT16/True:  on device, used directly in matmul or rms_norm
# - ROW_MAJOR/BFLOAT16/False: on host, processed by consteval before use
# - ROW_MAJOR/INT32/False: position ID buffers (host, for RoPE lookup)
# ---------------------------------------------------------------------------

def _build_param_config():
    """Build the full parameter config dict programmatically."""
    cfg = {}

    # --- Buffers and special tensors ---
    for name in ("cap_attn_mask", "cap_pad_mask", "image_pad_mask",
                 "x_attn_mask", "unified_attn_mask"):
        cfg[name] = ("ROW_MAJOR", "BFLOAT16", False)
    for name in ("cap_pos_ids", "image_pos_ids"):
        cfg[name] = ("ROW_MAJOR", "INT32", False)
    for name in ("cap_pad_token", "x_pad_token"):
        cfg[name] = ("TILE", "BFLOAT16", True)
    for axis in range(3):
        cfg[f"rope_embedder.cos_{axis}"] = ("ROW_MAJOR", "BFLOAT16", False)
        cfg[f"rope_embedder.sin_{axis}"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- Timestep embedder ---
    cfg["t_embedder.freqs"] = ("ROW_MAJOR", "BFLOAT16", False)
    for layer in (0, 2):
        cfg[f"t_embedder.mlp.{layer}.weight"] = ("ROW_MAJOR", "BFLOAT16", False)
        cfg[f"t_embedder.mlp.{layer}.bias"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- Patch embedder ---
    cfg["x_embedder.weight"] = ("ROW_MAJOR", "BFLOAT16", False)
    cfg["x_embedder.bias"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- Caption embedder ---
    cfg["cap_embedder.0.weight"] = ("TILE", "BFLOAT16", True)  # RMSNorm param
    cfg["cap_embedder.1.weight"] = ("ROW_MAJOR", "BFLOAT16", False)  # linear, consteval
    cfg["cap_embedder.1.bias"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- Final layer ---
    cfg["final_layer.linear.weight"] = ("TILE", "BFLOAT16", True)
    cfg["final_layer.linear.bias"] = ("ROW_MAJOR", "BFLOAT16", False)
    cfg["final_layer.adaLN_modulation.1.weight"] = ("ROW_MAJOR", "BFLOAT16", False)
    cfg["final_layer.adaLN_modulation.1.bias"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- Transformer blocks (noise_refiner, context_refiner, layers) ---
    block_specs = [
        ("noise_refiner", 2, True),
        ("context_refiner", 2, False),
        ("layers", 30, True),
    ]
    for prefix, count, has_adaln in block_specs:
        for i in range(count):
            p = f"{prefix}.{i}"
            # Attention params — on device for matmul with transpose_b=True (BF8 weights)
            for suffix in ("to_q", "to_k", "to_v", "to_out"):
                cfg[f"{p}.attention.{suffix}.weight"] = ("TILE", "BFLOAT8_B", True)
            # QK-norm params — on device for rms_norm (must stay BF16)
            for suffix in ("norm_q", "norm_k"):
                cfg[f"{p}.attention.{suffix}.weight"] = ("TILE", "BFLOAT16", True)
            # Block norms — on device for rms_norm (must stay BF16)
            for suffix in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"):
                cfg[f"{p}.{suffix}.weight"] = ("TILE", "BFLOAT16", True)
            # FFN params — on device for matmul with transpose_b=True (BF8 weights)
            for suffix in ("w1", "w2", "w3"):
                cfg[f"{p}.feed_forward.{suffix}.weight"] = ("TILE", "BFLOAT8_B", True)
            # adaLN modulation — on host for consteval (transpose + cast)
            if has_adaln:
                cfg[f"{p}.adaLN_modulation.0.weight"] = ("ROW_MAJOR", "BFLOAT16", False)
                cfg[f"{p}.adaLN_modulation.0.bias"] = ("ROW_MAJOR", "BFLOAT16", False)

    return cfg


PARAM_CONFIG = _build_param_config()


def load_params_from_pytorch(state_dict, device):
    """Convert a PyTorch state_dict to TTNN tensors.

    Each parameter's (layout, dtype, on_device) is looked up in PARAM_CONFIG.
    Parameters on device use TILE layout; host params stay ROW_MAJOR for consteval.

    Args:
        state_dict: dict from PyTorch model.state_dict()
        device: ttnn device handle

    Returns:
        dict mapping name -> ttnn tensor (mix of host and device tensors)
    """
    dtype_map = {
        "BFLOAT16": ttnn.DataType.BFLOAT16,
        "BFLOAT8_B": ttnn.DataType.BFLOAT8_B,
        "BFLOAT4_B": ttnn.DataType.BFLOAT4_B,
        "FLOAT32": ttnn.DataType.FLOAT32,
        "INT32": ttnn.DataType.INT32,
    }
    params = {}
    for name, tensor in state_dict.items():
        layout_str, dtype_str, on_device = PARAM_CONFIG[name]
        t = ttnn.from_torch(tensor)
        t = ttnn.to_dtype(t, dtype_map[dtype_str])
        if layout_str == "TILE":
            t = ttnn.to_layout(t, ttnn.Layout.TILE)
        if on_device:
            t = ttnn.to_device(t, device=device, memory_config=DRAM)
        params[name] = t
    return params
