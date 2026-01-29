#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Load TTNN model weights from PyTorch model instead of tensorbin files.

This allows running the TTNN model without the large tensors/ directory by
downloading weights directly from HuggingFace.
"""

import json
import os

import torch
import ttnn
from model_pt import CLIPVisionEncoderAndResamplerPT


def load_weights_from_pytorch(
    state_dict, device, config_path="tensor_load_config.json"
):
    """
    Load weights from PyTorch model and convert to TTNN format.

    Args:
        state_dict: PyTorch state_dict
        config_path: Path to tensor_load_config.json

    Returns:
        Dictionary mapping weight names to TTNN tensors
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Convert each weight to TTNN format
    weights = {}
    special_count = 0
    converted_count = 0

    for i in range(len(config)):
        key = f"arg{i}"
        cfg = config[key]
        weight_name = cfg.get("weight_name", "")
        layout_str = cfg.get("layout", "TILE")
        dtype_str = cfg.get("dtype", "BFLOAT16")
        on_device = cfg.get("on_device", False)

        # Handle special entries
        if weight_name == "__INPUT__":
            # This is a placeholder for runtime input
            weights[weight_name] = None
            special_count += 1
            continue

        if weight_name == "__POSITION_IDS__":
            # Generate position IDs tensor
            pos_ids = _create_position_ids()
            weights[weight_name] = pos_ids
            special_count += 1
            continue

        # Get PyTorch tensor
        if weight_name not in state_dict:
            raise ValueError(f"Weight '{weight_name}' not found in PyTorch model")

        pt_tensor = state_dict[weight_name]

        # Convert to TTNN
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        # Apply layout
        layout = ttnn.Layout.TILE if layout_str == "TILE" else ttnn.Layout.ROW_MAJOR
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)

        # Apply dtype
        if dtype_str == "BFLOAT16":
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
        elif dtype_str == "INT32":
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.INT32)

        # Move to device if needed
        if on_device and device is not None:
            mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            )
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, mem_config)

        weights[weight_name] = ttnn_tensor
        converted_count += 1

    print(f"Converted {converted_count} weights, {special_count} special entries")
    return weights


def _create_position_ids():
    """
    Create position IDs tensor for CLIP vision model.

    The position IDs are [0, 1, 2, ..., 256] for 257 positions
    (256 patches + 1 CLS token).
    """
    # CLIP ViT-H has 257 positions (16x16 patches + CLS token)
    num_positions = 257
    pos_ids = torch.arange(num_positions, dtype=torch.int32).unsqueeze(0)

    # Convert to TTNN
    ttnn_tensor = ttnn.from_torch(pos_ids)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
    ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.INT32)

    return ttnn_tensor


def compare_with_tensorbin(config_path="tensor_load_config.json"):
    """
    Compare weights loaded from PyTorch vs tensorbin files.

    Returns PCC for each weight tensor.
    """
    import utils

    # Load from tensorbin
    print("Loading weights from tensorbin files...")
    from weights_loader import load_inputs_for__main

    tensorbin_weights = load_inputs_for__main()

    # Load from PyTorch
    print("\nLoading weights from PyTorch model...")
    device = utils.DeviceGetter.get_device((1, 1))
    pytorch_weights = load_weights_from_pytorch(config_path, device)

    # Compare
    print("\nComparing weights...")
    results = []

    with open(config_path) as f:
        config = json.load(f)

    for i in range(len(tensorbin_weights)):
        weight_name = config[f"arg{i}"].get("weight_name", "")
        tb_w = tensorbin_weights[i]
        pt_w = pytorch_weights.get(weight_name)

        if tb_w is None or pt_w is None:
            results.append((weight_name, "SKIP", None))
            continue

        # Convert both to torch for comparison
        tb_torch = ttnn.to_torch(tb_w)
        pt_torch = ttnn.to_torch(pt_w)

        # Calculate PCC
        try:
            pcc = utils.calculate_pcc(tb_torch, pt_torch)
            status = "OK" if pcc > 0.999 else "WARN" if pcc > 0.99 else "FAIL"
            results.append((weight_name, status, pcc))
        except Exception as e:
            results.append((weight_name, "ERROR", str(e)))

    # Print summary
    ok_count = sum(1 for _, s, _ in results if s == "OK")
    warn_count = sum(1 for _, s, _ in results if s == "WARN")
    fail_count = sum(1 for _, s, _ in results if s == "FAIL")
    skip_count = sum(1 for _, s, _ in results if s == "SKIP")
    error_count = sum(1 for _, s, _ in results if s == "ERROR")

    print(
        f"\nResults: {ok_count} OK, {warn_count} WARN, {fail_count} FAIL, {skip_count} SKIP, {error_count} ERROR"
    )

    # Show failures and warnings
    for weight_name, status, pcc in results:
        if status in ("FAIL", "WARN", "ERROR"):
            print(f"  {weight_name}: {status} (PCC={pcc})")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_with_tensorbin()
    else:
        # Just test loading
        import utils

        device = utils.DeviceGetter.get_device((1, 1))
        weights = load_weights_from_pytorch(device=device)
        print(f"\nLoaded {len(weights)} weights")
        print(f"Non-None weights: {sum(1 for w in weights.values() if w is not None)}")
