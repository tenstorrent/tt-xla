#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract all weight and CER mappings from forward() method calls."""

import json
import re


def parse_forward_calls():
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    # Extract forward method
    forward_match = re.search(
        r"def forward\(self, pixel_values\):(.*?)(?=\n    def |\nclass |\Z)",
        content,
        re.DOTALL,
    )
    forward_body = forward_match.group(1)

    # Find all attention calls
    attn_pattern = re.compile(
        r"CLIPAttention_(\d+)_0_0 = self\.CLIPAttention_\d+_0\(\s*" r"([^)]+)\)",
        re.DOTALL,
    )

    # Find all MLP calls
    mlp_pattern = re.compile(
        r"CLIPMLP_(\d+)_0_0 = self\.CLIPMLP_\d+_0\(\s*" r"([^)]+)\)", re.DOTALL
    )

    # Find all layer_norm2_add calls (CLIPEncoderLayer that returns 2 values)
    ln2_pattern = re.compile(
        r"v_\d+, v_\d+ = self\.CLIPEncoderLayer_(\d+)_0\(\s*" r"([^)]+)\)", re.DOTALL
    )

    # Find layer 0's layer_norm1 call
    ln1_layer0_pattern = re.compile(
        r"CLIPEncoderLayer_2_0_0 = self\.CLIPEncoderLayer_2_0\(\s*" r"([^)]+)\)",
        re.DOTALL,
    )

    def parse_args(args_str):
        """Parse comma-separated arguments, handling newlines."""
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if arg:
                # Extract weight index or cer key
                weight_match = re.search(r"self\.weights\[(\d+)\]", arg)
                cer_match = re.search(r'self\.cer\["([^"]+)"\]', arg)
                if weight_match:
                    args.append(("weight", int(weight_match.group(1))))
                elif cer_match:
                    args.append(("cer", cer_match.group(1)))
                else:
                    # Variable (hidden_states)
                    args.append(("var", arg.strip()))
        return args

    # Determine method index to layer mapping
    def get_layer_idx(method_idx):
        if method_idx == 2:
            return 0  # layer_norm1 for layer 0
        elif method_idx == 3:
            return 0  # attention for layer 0
        elif method_idx == 4:
            return 0  # layer_norm2 for layer 0
        elif method_idx == 5:
            return 0  # mlp for layer 0
        elif method_idx == 6:
            return 0  # layer_norm1_next for layer 0
        elif method_idx >= 7 and method_idx <= 126:
            return ((method_idx - 7) // 4) + 1
        return -1

    # Process attention calls
    print("# Attention mappings (layer_idx -> {semantic_name: (type, value)})")
    print("ATTENTION_MAPPINGS = {")
    for match in attn_pattern.finditer(forward_body):
        method_idx = int(match.group(1))
        layer_idx = get_layer_idx(method_idx)
        args = parse_args(match.group(2))

        # Now we need to know what each argument position maps to
        # By analyzing the method bodies:
        # - The hidden_states arg position varies
        # - But we know: qkv_weight is used in first matmul, qkv_bias in first add
        #               out_proj_weight in second matmul, out_proj_bias in second add

        # Find which arg is the variable (hidden_states)
        hidden_idx = next(i for i, (t, v) in enumerate(args) if t == "var")

        # The others are weights/cer
        non_hidden = [(i, t, v) for i, (t, v) in enumerate(args) if t != "var"]

        print(f"    {layer_idx}: {{")
        print(
            f"        # Method: CLIPAttention_{method_idx}_0, hidden_states at arg {hidden_idx}"
        )
        for i, (t, v) in enumerate(args):
            if t == "weight":
                print(f"        # arg{i}: weights[{v}]")
            elif t == "cer":
                print(f"        # arg{i}: cer['{v}']")
            else:
                print(f"        # arg{i}: {v} (hidden_states)")
        print(f"    }},")

    print("}")

    # Process MLP calls
    print("\n# MLP mappings")
    print("MLP_MAPPINGS = {")
    for match in mlp_pattern.finditer(forward_body):
        method_idx = int(match.group(1))
        layer_idx = get_layer_idx(method_idx)
        args = parse_args(match.group(2))

        hidden_idx = next(i for i, (t, v) in enumerate(args) if t == "var")

        print(f"    {layer_idx}: {{")
        print(
            f"        # Method: CLIPMLP_{method_idx}_0, hidden_states at arg {hidden_idx}"
        )
        for i, (t, v) in enumerate(args):
            if t == "weight":
                print(f"        # arg{i}: weights[{v}]")
            elif t == "cer":
                print(f"        # arg{i}: cer['{v}']")
            else:
                print(f"        # arg{i}: {v} (hidden_states)")
        print(f"    }},")

    print("}")


if __name__ == "__main__":
    parse_forward_calls()
