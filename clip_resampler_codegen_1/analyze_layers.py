#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Analyze model_ttnn.py to extract weight and cer mappings per encoder layer."""

import re
from collections import defaultdict


def parse_forward():
    """Parse forward() method to extract layer call sequences."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    # Extract forward method
    forward_match = re.search(
        r"def forward\(self, pixel_values\):(.*?)(?=\n    def |\nclass |\Z)",
        content,
        re.DOTALL,
    )
    if not forward_match:
        print("Could not find forward method")
        return

    forward_body = forward_match.group(1)

    # Find all method calls with their arguments
    # Pattern: self.MethodName_N_0(...) or var = self.MethodName_N_0(...)
    call_pattern = re.compile(
        r"(?:(\w+(?:,\s*\w+)*)\s*=\s*)?self\.(\w+_\d+_0)\((.*?)\)", re.DOTALL
    )

    calls = []
    for match in call_pattern.finditer(forward_body):
        return_vars = match.group(1)
        method_name = match.group(2)
        args_str = match.group(3)

        # Parse arguments
        args = []
        # Handle multiline arguments
        args_clean = re.sub(r"\s+", " ", args_str.strip())
        # Split by comma, but respect brackets
        depth = 0
        current_arg = ""
        for char in args_clean:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            current_arg += char
        if current_arg.strip():
            args.append(current_arg.strip())

        calls.append({"return_vars": return_vars, "method": method_name, "args": args})

    return calls


def analyze_encoder_layers(calls):
    """Group calls by encoder layer."""
    # Layer 0 pattern: CLIPEncoderLayer_2, CLIPAttention_3, CLIPEncoderLayer_4, CLIPMLP_5, CLIPEncoderLayer_6
    # Layer N (N>=1) pattern: CLIPAttention_{4N+3}, CLIPEncoderLayer_{4N+4}, CLIPMLP_{4N+5}, CLIPEncoderLayer_{4N+6}

    layers = defaultdict(dict)

    for call in calls:
        method = call["method"]
        args = call["args"]

        # Extract method type and index
        match = re.match(
            r"(CLIPEncoderLayer|CLIPAttention|CLIPMLP|LayerNorm|CLIPVisionEmbeddings|Linear|IPAdapter|Attention)_(\d+)_0",
            method,
        )
        if not match:
            continue

        method_type = match.group(1)
        method_idx = int(match.group(2))

        # Skip non-encoder methods
        if method_type in ["CLIPVisionEmbeddings", "LayerNorm"] and method_idx <= 1:
            continue
        if method_type in ["Linear", "IPAdapter", "Attention"] and method_idx >= 127:
            continue

        # Determine layer index
        if method_idx == 2:  # First layer_norm1 (layer 0 only)
            layer_idx = 0
            op_type = "layer_norm1"
        elif method_idx == 3:  # First attention (layer 0)
            layer_idx = 0
            op_type = "attention"
        elif method_idx == 4:  # First layer_norm2 (layer 0)
            layer_idx = 0
            op_type = "layer_norm2_add"
        elif method_idx == 5:  # First MLP (layer 0)
            layer_idx = 0
            op_type = "mlp"
        elif method_idx == 6:  # First layer_norm1_next (layer 0)
            layer_idx = 0
            op_type = "layer_norm1_next"
        elif method_idx >= 7 and method_idx <= 126:
            # Layers 1-31: 4 methods per layer starting at index 7
            # Method 7,8,9,10 -> layer 1
            # Method 11,12,13,14 -> layer 2
            # etc.
            adjusted_idx = method_idx - 7
            layer_idx = (adjusted_idx // 4) + 1
            op_within_layer = adjusted_idx % 4
            op_types = ["attention", "layer_norm2_add", "mlp", "layer_norm1_next"]
            op_type = op_types[op_within_layer]
        else:
            continue

        # Extract weights and cer keys from arguments
        weights = []
        cer_keys = []

        for arg in args:
            weight_match = re.search(r"self\.weights\[(\d+)\]", arg)
            if weight_match:
                weights.append(int(weight_match.group(1)))

            cer_match = re.search(r'self\.cer\["([^"]+)"\]', arg)
            if cer_match:
                cer_keys.append(cer_match.group(1))

        if layer_idx not in layers:
            layers[layer_idx] = {"weights": [], "cer_keys": [], "ops": {}}

        layers[layer_idx]["weights"].extend(weights)
        layers[layer_idx]["cer_keys"].extend(cer_keys)
        layers[layer_idx]["ops"][op_type] = {
            "method": method,
            "weights": weights,
            "cer_keys": cer_keys,
            "args": args,
        }

    return dict(layers)


def print_layer_analysis(layers):
    """Print analysis of each layer."""
    print("=" * 80)
    print("ENCODER LAYER ANALYSIS")
    print("=" * 80)

    for layer_idx in sorted(layers.keys()):
        if layer_idx > 31:
            continue
        layer = layers[layer_idx]
        print(f"\n{'='*40}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*40}")
        print(f"Total weights: {sorted(set(layer['weights']))}")
        print(f"Total cer keys: {sorted(set(layer['cer_keys']))}")

        for op_type, op_info in layer["ops"].items():
            print(f"\n  {op_type}:")
            print(f"    Method: {op_info['method']}")
            print(f"    Weights: {op_info['weights']}")
            print(f"    CER keys: {op_info['cer_keys']}")


def generate_weight_mapping(layers):
    """Generate weight mapping code."""
    print("\n" + "=" * 80)
    print("WEIGHT MAPPING CODE")
    print("=" * 80)

    print(
        """
def _get_layer_weights(self, layer_idx):
    \"\"\"Get weight dictionary for a specific encoder layer.\"\"\"
    # Weight indices per layer (descending pattern)
    layer_weight_map = {"""
    )

    for layer_idx in sorted(layers.keys()):
        if layer_idx > 31:
            continue
        layer = layers[layer_idx]
        weights = sorted(set(layer["weights"]), reverse=True)
        print(f"        {layer_idx}: {weights},")

    print(
        """    }
    return layer_weight_map.get(layer_idx, [])
"""
    )

    print(
        """
def _get_layer_cer(self, layer_idx):
    \"\"\"Get CER keys for a specific encoder layer.\"\"\"
    layer_cer_map = {"""
    )

    for layer_idx in sorted(layers.keys()):
        if layer_idx > 31:
            continue
        layer = layers[layer_idx]
        cer_keys = sorted(set(layer["cer_keys"]))
        print(f"        {layer_idx}: {cer_keys},")

    print(
        """    }
    return layer_cer_map.get(layer_idx, [])
"""
    )


if __name__ == "__main__":
    print("Parsing forward() method...")
    calls = parse_forward()
    print(f"Found {len(calls)} method calls")

    print("\nAnalyzing encoder layers...")
    layers = analyze_encoder_layers(calls)
    print(f"Found {len([k for k in layers.keys() if k <= 31])} encoder layers")

    print_layer_analysis(layers)
    generate_weight_mapping(layers)
