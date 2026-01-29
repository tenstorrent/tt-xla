#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate 31 CLIPEncoderLayerTTNN classes, one per layer.

Each class has hardcoded weight/cer argument mappings extracted from the original code.
"""

import re


def parse_method_bodies():
    """Parse all method bodies from model_ttnn.py."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    # Extract method definitions - looking for indented def
    method_pattern = re.compile(
        r"^    def (CLIPAttention|CLIPMLP|CLIPEncoderLayer)_(\d+)_0\(self, ([^)]*)\):(.*?)(?=\n    def |\nclass |\Z)",
        re.MULTILINE | re.DOTALL,
    )

    methods = {}
    for match in method_pattern.finditer(content):
        method_type = match.group(1)
        method_idx = int(match.group(2))
        params = match.group(3).strip()
        body = match.group(4)

        methods[f"{method_type}_{method_idx}_0"] = {
            "type": method_type,
            "idx": method_idx,
            "params": [p.strip() for p in params.split(",")],
            "body": body,
        }

    return methods


def parse_forward_calls():
    """Parse forward() method to get call arguments."""
    with open("model_ttnn.py", "r") as f:
        content = f.read()

    forward_match = re.search(
        r"def forward\(self, pixel_values\):(.*?)(?=\n    def )", content, re.DOTALL
    )
    forward_body = forward_match.group(1)

    # Pattern for method calls
    call_pattern = re.compile(
        r"(?:(\w+(?:,\s*\w+)*)\s*=\s*)?self\.(CLIPAttention|CLIPMLP|CLIPEncoderLayer)_(\d+)_0\(\s*([^)]+)\)",
        re.DOTALL,
    )

    calls = []
    for match in call_pattern.finditer(forward_body):
        return_vars = match.group(1) or ""
        method_type = match.group(2)
        method_idx = int(match.group(3))
        args_str = match.group(4)

        # Parse arguments
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if arg:
                args.append(arg)

        calls.append(
            {
                "return_vars": return_vars,
                "method_type": method_type,
                "method_idx": method_idx,
                "method_name": f"{method_type}_{method_idx}_0",
                "args": args,
            }
        )

    return calls


def get_layer_structure():
    """
    Determine which methods belong to which layer.

    Layer 0 is special:
    - CLIPEncoderLayer_2_0: layer_norm1
    - CLIPAttention_3_0: attention
    - CLIPEncoderLayer_4_0: residual + layer_norm2
    - CLIPMLP_5_0: mlp
    - CLIPEncoderLayer_6_0: residual + layer_norm1_next

    Layers 1-29:
    - CLIPAttention_{4N+3}_0: attention
    - CLIPEncoderLayer_{4N+4}_0: residual + layer_norm2
    - CLIPMLP_{4N+5}_0: mlp
    - CLIPEncoderLayer_{4N+6}_0: residual + layer_norm1_next

    Layer 30 (last):
    - CLIPAttention_123_0: attention
    - CLIPEncoderLayer_124_0: residual + layer_norm2
    - CLIPMLP_125_0: mlp
    - CLIPEncoderLayer_126_0: final residual (no layer_norm)
    """
    layers = {}

    # Layer 0
    layers[0] = {
        "layer_norm1": "CLIPEncoderLayer_2_0",
        "attention": "CLIPAttention_3_0",
        "layer_norm2_add": "CLIPEncoderLayer_4_0",
        "mlp": "CLIPMLP_5_0",
        "layer_norm1_next": "CLIPEncoderLayer_6_0",
    }

    # Layers 1-29
    for n in range(1, 30):
        base = 4 * n + 3
        layers[n] = {
            "attention": f"CLIPAttention_{base}_0",
            "layer_norm2_add": f"CLIPEncoderLayer_{base + 1}_0",
            "mlp": f"CLIPMLP_{base + 2}_0",
            "layer_norm1_next": f"CLIPEncoderLayer_{base + 3}_0",
        }

    # Layer 30 (last)
    layers[30] = {
        "attention": "CLIPAttention_123_0",
        "layer_norm2_add": "CLIPEncoderLayer_124_0",
        "mlp": "CLIPMLP_125_0",
        "final_residual": "CLIPEncoderLayer_126_0",
    }

    return layers


def normalize_return_order(body, op_name):
    """
    Normalize the return statement order for methods that return two values.

    For _layer_norm2_add:
        Always return (layer_norm_output, add_result) = (mlp_input, residual)

    For _layer_norm1_next:
        Always return (add_result, layer_norm_output) = (new_residual, normalized)
    """
    if op_name == "layer_norm2_add":
        # Find the return statement and ensure it returns (layer_norm, add)
        # The layer_norm variable contains "layer_norm" in its name
        # The add variable contains "add" in its name
        return_match = re.search(r"return\s+(\w+),\s*(\w+)", body)
        if return_match:
            var1, var2 = return_match.group(1), return_match.group(2)
            # Check if order needs swapping
            # layer_norm should be first (mlp_input), add should be second (residual)
            if "add" in var1.lower() and "layer_norm" in var2.lower():
                # Swap the order
                old_return = f"return {var1}, {var2}"
                new_return = f"return {var2}, {var1}"
                body = body.replace(old_return, new_return)
    elif op_name == "layer_norm1_next":
        # Find the return statement and ensure it returns (add, layer_norm)
        # The add variable is new_residual, layer_norm is normalized_for_next
        return_match = re.search(r"return\s+(\w+),\s*(\w+)", body)
        if return_match:
            var1, var2 = return_match.group(1), return_match.group(2)
            # Check if order needs swapping
            # add should be first (new_residual), layer_norm should be second (normalized)
            if "layer_norm" in var1.lower() and "add" in var2.lower():
                # Swap the order
                old_return = f"return {var1}, {var2}"
                new_return = f"return {var2}, {var1}"
                body = body.replace(old_return, new_return)
    return body


def transform_method_body(body, params, call_args, op_name):
    """
    Transform a method body by replacing input_N parameters with actual values.

    Args:
        body: Original method body
        params: List of parameter names ['input_0', 'input_1', ...]
        call_args: List of actual arguments from forward() call
        op_name: Operation name for determining output variable mapping

    Returns:
        Transformed method body
    """
    # Create replacement mapping
    replacements = {}
    for i, param in enumerate(params):
        if i < len(call_args):
            arg = call_args[i]
            # Transform self.weights[N] to self.wN
            weight_match = re.search(r"self\.weights\[(\d+)\]", arg)
            cer_match = re.search(r'self\.cer\["([^"]+)"\]', arg)

            if weight_match:
                replacements[param] = f"self.w{weight_match.group(1)}"
            elif cer_match:
                key = cer_match.group(1)
                safe_key = key.replace("utils_constEvalFuncWrapper_", "cer_")
                replacements[param] = f"self.{safe_key}"
            else:
                # Variable - map based on operation type
                if op_name == "layer_norm1":
                    replacements[param] = "hidden_states"
                elif op_name == "attention":
                    replacements[param] = "hidden_states"
                elif op_name == "layer_norm2_add":
                    # CLIPEncoderLayer_N_0 for residual + layernorm2:
                    # Typically: input_0=bias, input_1=weight, input_2=residual, input_3=attn_output
                    # But may vary. Check the argument name for patterns:
                    # - LayerNorm_1_0_0 or v_NNN (no _0 suffix) = residual
                    # - CLIPAttention_N_0_0 = attn_output
                    if "LayerNorm" in arg or ("v_" in arg and "_0_0" not in arg):
                        replacements[param] = "residual"
                    elif "CLIPAttention" in arg or "v_" in arg:
                        replacements[param] = "attn_output"
                    else:
                        replacements[param] = arg
                elif op_name == "mlp":
                    replacements[param] = "hidden_states"
                elif op_name in ["layer_norm1_next", "final_residual"]:
                    # input typically: mlp_output and residual
                    # - CLIPMLP_N_0_0 = mlp_output
                    # - v_NNN (residual) = residual
                    if "CLIPMLP" in arg:
                        replacements[param] = "mlp_output"
                    elif "v_" in arg:
                        replacements[param] = "residual"
                    else:
                        replacements[param] = arg
                else:
                    replacements[param] = arg

    # Apply replacements - need to be careful about partial matches
    # Sort by length (longest first) to avoid partial replacement issues
    for param in sorted(replacements.keys(), key=len, reverse=True):
        replacement = replacements[param]
        # Use word boundary matching to avoid partial replacements
        body = re.sub(rf"\b{param}\b", replacement, body)

    return body


def generate_layer_classes():
    """Generate the layer classes."""
    methods = parse_method_bodies()
    calls = parse_forward_calls()
    layer_structure = get_layer_structure()

    # Build a lookup from method name to call info
    call_lookup = {}
    for call in calls:
        call_lookup[call["method_name"]] = call

    output = []
    output.append("# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC")
    output.append("#")
    output.append("# SPDX-License-Identifier: Apache-2.0")
    output.append('"""Generated CLIP encoder layer classes."""')
    output.append("")
    output.append("import ttnn")
    output.append("from models.common.lightweightmodule import LightweightModule")
    output.append("")

    for layer_idx in range(31):
        layer_methods = layer_structure[layer_idx]
        is_first = layer_idx == 0
        is_last = layer_idx == 30

        output.append(f"")
        output.append(f"class CLIPEncoderLayerTTNN_{layer_idx}(LightweightModule):")
        output.append(f'    """CLIP Encoder Layer {layer_idx}."""')
        output.append(f"")

        # Generate __init__
        output.append(f"    def __init__(self, weights, cer):")
        output.append(f'        """Store layer weights and cer values."""')

        # Collect all weights and cer keys used by this layer's methods
        weight_indices = set()
        cer_keys = set()

        for op_name, method_name in layer_methods.items():
            call_info = call_lookup.get(method_name)
            if call_info:
                for arg in call_info["args"]:
                    weight_match = re.search(r"self\.weights\[(\d+)\]", arg)
                    cer_match = re.search(r'self\.cer\["([^"]+)"\]', arg)
                    if weight_match:
                        weight_indices.add(int(weight_match.group(1)))
                    elif cer_match:
                        cer_keys.add(cer_match.group(1))

        # Store weights
        for idx in sorted(weight_indices, reverse=True):
            output.append(f"        self.w{idx} = weights[{idx}]")

        # Store cer values
        for key in sorted(cer_keys):
            safe_key = key.replace("utils_constEvalFuncWrapper_", "cer_")
            output.append(f'        self.{safe_key} = cer["{key}"]')

        output.append(f"")

        # Generate forward method
        output.append(f"    def forward(self, hidden_states, residual):")
        output.append(f'        """Forward pass."""')

        # Layer 0 has extra layer_norm1
        if is_first:
            output.append(f"        # layer_norm1")
            output.append(f"        hidden_states = self._layer_norm1(hidden_states)")

        # Attention
        output.append(f"        # attention")
        output.append(f"        attn_output = self._attention(hidden_states)")

        # Residual + layer_norm2
        output.append(f"        # residual + layer_norm2")
        output.append(
            f"        mlp_input, residual = self._layer_norm2_add(residual, attn_output)"
        )

        # MLP
        output.append(f"        # mlp")
        output.append(f"        mlp_output = self._mlp(mlp_input)")

        # Final residual or layer_norm1_next
        if is_last:
            output.append(f"        # final residual")
            output.append(
                f"        output = self._final_residual(residual, mlp_output)"
            )
            output.append(f"        return output, None")
        else:
            output.append(f"        # residual + layer_norm1_next")
            output.append(
                f"        new_residual, normalized = self._layer_norm1_next(residual, mlp_output)"
            )
            output.append(f"        return new_residual, normalized")

        output.append(f"")

        # Generate helper methods by extracting from original methods
        for op_name, method_name in layer_methods.items():
            method_info = methods.get(method_name)
            call_info = call_lookup.get(method_name)

            if not method_info or not call_info:
                output.append(f"    # WARNING: Could not find {method_name}")
                continue

            # Get transformed body
            transformed_body = transform_method_body(
                method_info["body"], method_info["params"], call_info["args"], op_name
            )

            # Normalize return order for methods that return two values
            transformed_body = normalize_return_order(transformed_body, op_name)

            # Determine method signature based on operation
            if op_name == "layer_norm1":
                output.append(f"    def _layer_norm1(self, hidden_states):")
            elif op_name == "attention":
                output.append(f"    def _attention(self, hidden_states):")
            elif op_name == "layer_norm2_add":
                output.append(f"    def _layer_norm2_add(self, residual, attn_output):")
            elif op_name == "mlp":
                output.append(f"    def _mlp(self, hidden_states):")
            elif op_name == "layer_norm1_next":
                output.append(f"    def _layer_norm1_next(self, residual, mlp_output):")
            elif op_name == "final_residual":
                output.append(f"    def _final_residual(self, residual, mlp_output):")
            else:
                continue

            # Add the transformed body (with proper indentation)
            for line in transformed_body.split("\n"):
                # Skip empty lines at start
                if not output[-1].endswith(":") and not line.strip():
                    continue
                # Add 4 more spaces for method body indentation
                if line.strip():
                    output.append(f"    {line}")
                else:
                    output.append("")

            output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    code = generate_layer_classes()

    # Write to file
    output_file = "clip_encoder_layers_generated.py"
    with open(output_file, "w") as f:
        f.write(code)

    print(f"Generated {len(code)} characters to {output_file}")

    # Verify syntax
    import py_compile

    try:
        py_compile.compile(output_file, doraise=True)
        print("Syntax OK!")
    except py_compile.PyCompileError as e:
        print(f"Syntax error: {e}")
