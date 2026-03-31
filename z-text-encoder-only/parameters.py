# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter loading config and function for the Z-Image text encoder (Qwen3).

Converts a PyTorch state_dict into TTNN tensors with the correct
layout, dtype, and device placement for each parameter.

Architecture: Qwen3 decoder (text encoder mode)
  - embed_tokens: [151936, 2560] embedding lookup
  - rotary_emb.inv_freq: [64] RoPE inverse frequencies
  - 35 decoder layers (0-34), each with:
      input_layernorm, self_attn (q/k/v/o_proj, q_norm, k_norm),
      post_attention_layernorm, mlp (gate/up/down_proj)
"""

import ttnn

DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

NUM_LAYERS = 35


# ---------------------------------------------------------------------------
# Parameter config: (layout, dtype, on_device) per parameter name.
#
# - TILE/BFLOAT16/True:     on device, used directly in matmul or rms_norm
# - ROW_MAJOR/BFLOAT16/False: on host, processed by consteval before use
# - ROW_MAJOR/FLOAT32/False:  on host, processed by consteval (inv_freq)
# ---------------------------------------------------------------------------

def _build_param_config():
    """Build the full parameter config dict programmatically."""
    cfg = {}

    # --- Embedding table (host, processed by consteval) ---
    cfg["embed_tokens.weight"] = ("ROW_MAJOR", "BFLOAT16", False)

    # --- RoPE inverse frequencies (host, processed by consteval) ---
    cfg["rotary_emb.inv_freq"] = ("ROW_MAJOR", "FLOAT32", False)

    # --- Decoder layers ---
    for i in range(NUM_LAYERS):
        p = f"layers.{i}"

        # LayerNorm weights (on device for rms_norm)
        cfg[f"{p}.input_layernorm.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.post_attention_layernorm.weight"] = ("TILE", "BFLOAT16", True)

        # Attention projection weights (on device for matmul with transpose_b=True)
        cfg[f"{p}.self_attn.q_proj.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.self_attn.k_proj.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.self_attn.v_proj.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.self_attn.o_proj.weight"] = ("TILE", "BFLOAT16", True)

        # QK-norm weights (on device for rms_norm)
        cfg[f"{p}.self_attn.q_norm.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.self_attn.k_norm.weight"] = ("TILE", "BFLOAT16", True)

        # MLP weights (on device for matmul with transpose_b=True)
        cfg[f"{p}.mlp.gate_proj.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.mlp.up_proj.weight"] = ("TILE", "BFLOAT16", True)
        cfg[f"{p}.mlp.down_proj.weight"] = ("TILE", "BFLOAT16", True)

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
        "FLOAT32": ttnn.DataType.FLOAT32,
        "INT32": ttnn.DataType.INT32,
    }
    params = {}
    for name, tensor in state_dict.items():
        if name not in PARAM_CONFIG:
            print(f"  [skip] {name} not in PARAM_CONFIG")
            continue
        layout_str, dtype_str, on_device = PARAM_CONFIG[name]
        t = ttnn.from_torch(tensor)
        t = ttnn.to_dtype(t, dtype_map[dtype_str])
        if layout_str == "TILE":
            t = ttnn.to_layout(t, ttnn.Layout.TILE)
        if on_device:
            t = ttnn.to_device(t, device=device, memory_config=DRAM)
        params[name] = t
    return params
