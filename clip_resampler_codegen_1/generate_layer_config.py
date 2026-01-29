#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate layer configuration by analyzing method bodies.

For each method, we analyze which input_N parameter is used in which operation,
then map the actual arguments from the forward() call to semantic names.
"""

import re


def analyze_attention_method(method_body):
    """
    Analyze an attention method body to determine input mappings.

    Returns dict mapping input_N to semantic name.
    """
    mappings = {}

    # Find hidden_states (used in first reshape)
    reshape_match = re.search(r"ttnn\.reshape\(\s*input_(\d+)", method_body)
    if reshape_match:
        mappings[int(reshape_match.group(1))] = "hidden_states"

    # Find qkv_weight (used in first matmul, second operand)
    # Pattern: ttnn.matmul(ttnn_reshape_..., input_N, ...)
    matmul_matches = list(
        re.finditer(r"ttnn\.matmul\(\s*\w+,\s*input_(\d+)", method_body)
    )
    if len(matmul_matches) >= 1:
        mappings[int(matmul_matches[0].group(1))] = "qkv_weight"
    if len(matmul_matches) >= 2:
        mappings[int(matmul_matches[1].group(1))] = "out_proj_weight"

    # Find biases (used in ttnn.add after matmul)
    # First add after first matmul = qkv_bias
    # Second add (after SDPA) = out_proj_bias
    add_matches = list(re.finditer(r"ttnn\.add\(\s*\w+,\s*input_(\d+)", method_body))
    if len(add_matches) >= 1:
        mappings[int(add_matches[0].group(1))] = "qkv_bias"
    if len(add_matches) >= 2:
        mappings[int(add_matches[1].group(1))] = "out_proj_bias"

    return mappings


def analyze_mlp_method(method_body):
    """
    Analyze an MLP method body to determine input mappings.
    """
    mappings = {}

    # Find hidden_states (used in first reshape)
    reshape_match = re.search(r"ttnn\.reshape\(\s*input_(\d+)", method_body)
    if reshape_match:
        mappings[int(reshape_match.group(1))] = "hidden_states"

    # Find fc1_weight and fc2_weight (used in matmuls)
    matmul_matches = list(
        re.finditer(r"ttnn\.matmul\(\s*\w+,\s*input_(\d+)", method_body)
    )
    if len(matmul_matches) >= 1:
        mappings[int(matmul_matches[0].group(1))] = "fc1_weight"
    if len(matmul_matches) >= 2:
        mappings[int(matmul_matches[1].group(1))] = "fc2_weight"

    # Find biases (fc1_bias after first matmul, fc2_bias after second)
    add_matches = list(re.finditer(r"ttnn\.add\(\s*\w+,\s*input_(\d+)", method_body))
    if len(add_matches) >= 1:
        mappings[int(add_matches[0].group(1))] = "fc1_bias"
    if len(add_matches) >= 2:
        mappings[int(add_matches[1].group(1))] = "fc2_bias"

    return mappings


def analyze_layernorm2_method(method_body):
    """
    Analyze CLIPEncoderLayer method that does residual + layer_norm2.
    """
    mappings = {}

    # Find the two inputs to ttnn.add (residual and attn_output)
    add_match = re.search(r"ttnn\.add\(\s*input_(\d+),\s*input_(\d+)", method_body)
    if add_match:
        # One of these is residual, one is attn_output - determined by usage
        # The layer_norm uses the add result, so we need to find weight/bias
        pass

    # Find layer_norm weight and bias
    ln_match = re.search(
        r"ttnn\.layer_norm\([^)]*weight=input_(\d+)[^)]*bias=input_(\d+)", method_body
    )
    if ln_match:
        mappings[int(ln_match.group(1))] = "layer_norm2_weight"
        mappings[int(ln_match.group(2))] = "layer_norm2_bias"

    return mappings


def parse_method_definitions():
    """Parse all method definitions and extract input mappings."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    # Find all method definitions
    method_pattern = re.compile(
        r"def (CLIPAttention|CLIPMLP|CLIPEncoderLayer)_(\d+)_0\(self[^)]*\):(.*?)(?=\n    def |\nclass |\Z)",
        re.DOTALL,
    )

    results = {}

    for match in method_pattern.finditer(content):
        method_type = match.group(1)
        method_idx = int(match.group(2))
        method_body = match.group(3)

        if method_type == "CLIPAttention":
            mappings = analyze_attention_method(method_body)
            results[f"attn_{method_idx}"] = mappings
        elif method_type == "CLIPMLP":
            mappings = analyze_mlp_method(method_body)
            results[f"mlp_{method_idx}"] = mappings
        elif method_type == "CLIPEncoderLayer":
            mappings = analyze_layernorm2_method(method_body)
            results[f"ln_{method_idx}"] = mappings

    return results


def parse_forward_calls():
    """Parse forward() method calls to get actual arguments."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    forward_match = re.search(
        r"def forward\(self, pixel_values\):(.*?)(?=\n    def |\nclass |\Z)",
        content,
        re.DOTALL,
    )
    forward_body = forward_match.group(1)

    # Pattern for method calls with arguments
    call_pattern = re.compile(
        r"self\.(CLIPAttention|CLIPMLP|CLIPEncoderLayer)_(\d+)_0\(\s*([^)]+)\)",
        re.DOTALL,
    )

    calls = {}
    for match in call_pattern.finditer(forward_body):
        method_type = match.group(1)
        method_idx = int(match.group(2))
        args_str = match.group(3)

        # Parse arguments
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            weight_match = re.search(r"self\.weights\[(\d+)\]", arg)
            cer_match = re.search(r'self\.cer\["([^"]+)"\]', arg)
            if weight_match:
                args.append(("weight", int(weight_match.group(1))))
            elif cer_match:
                args.append(("cer", cer_match.group(1)))
            else:
                args.append(("var", arg.strip()))

        key = f'{method_type.lower().replace("clipencoderlayer", "ln").replace("clipattention", "attn").replace("clipmlp", "mlp")}_{method_idx}'
        calls[key] = args

    return calls


def generate_layer_config():
    """Generate configuration for each layer."""
    method_mappings = parse_method_definitions()
    forward_calls = parse_forward_calls()

    print("# Layer configurations")
    print("# For each layer, maps semantic names to weight indices or CER keys")
    print()
    print("LAYER_CONFIGS = {")

    # Determine layer index from method index
    def get_layer_idx(method_idx, method_type):
        if method_type == "attn":
            if method_idx == 3:
                return 0
            elif method_idx >= 7:
                return ((method_idx - 7) // 4) + 1
        elif method_type == "mlp":
            if method_idx == 5:
                return 0
            elif method_idx >= 9:
                return ((method_idx - 9) // 4) + 1
        return -1

    # Process attention methods
    for key, mappings in sorted(method_mappings.items()):
        if not key.startswith("attn_"):
            continue
        method_idx = int(key.split("_")[1])
        layer_idx = get_layer_idx(method_idx, "attn")
        if layer_idx < 0:
            continue

        call_args = forward_calls.get(key, [])
        if not call_args:
            continue

        print(f"    # Layer {layer_idx} attention (method {method_idx})")
        print(f"    # Input mappings: {mappings}")

        config = {}
        for input_idx, semantic_name in mappings.items():
            if input_idx < len(call_args):
                arg_type, arg_val = call_args[input_idx]
                if arg_type == "weight":
                    config[semantic_name] = ("weight", arg_val)
                elif arg_type == "cer":
                    config[semantic_name] = ("cer", arg_val)

        for name, (src, val) in config.items():
            if name != "hidden_states":
                print(f"    # {name}: {src}[{val}]")

    print("}")


if __name__ == "__main__":
    generate_layer_config()
