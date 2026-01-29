#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Analyze generated layer classes to find structurally identical layers.

Uses AST parsing to compare layers while ignoring variable names.
"""

import ast
import hashlib
from collections import defaultdict


class ASTNormalizer(ast.NodeTransformer):
    """Normalize AST by replacing all names with placeholders."""

    def __init__(self):
        self.name_counter = 0
        self.name_map = {}

    def visit_Name(self, node):
        # Keep built-in names and module references
        if node.id in ("self", "ttnn", "None", "True", "False", "utils"):
            return node

        # Map variable names to normalized placeholders
        if node.id not in self.name_map:
            self.name_map[node.id] = f"_var_{self.name_counter}"
            self.name_counter += 1

        return ast.Name(id=self.name_map[node.id], ctx=node.ctx)

    def visit_Attribute(self, node):
        # Normalize self.wN and self.cer_N to generic placeholders
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr.startswith("w") and node.attr[1:].isdigit():
                return ast.Attribute(value=node.value, attr="_weight_", ctx=node.ctx)
            elif node.attr.startswith("cer_"):
                return ast.Attribute(value=node.value, attr="_cer_", ctx=node.ctx)

        self.generic_visit(node)
        return node


def normalize_ast(tree):
    """Normalize an AST tree for comparison."""
    normalizer = ASTNormalizer()
    return normalizer.visit(tree)


def get_method_hash(method_def):
    """Get a hash of a method definition, ignoring variable names."""
    # Create a copy to avoid modifying original
    method_copy = ast.parse(ast.unparse(method_def)).body[0]

    # Normalize the AST
    normalized = normalize_ast(method_copy)

    # Convert to string and hash
    code_str = ast.unparse(normalized)
    return hashlib.md5(code_str.encode()).hexdigest(), code_str


def extract_layer_methods(source_code):
    """Extract methods from each layer class."""
    tree = ast.parse(source_code)

    layers = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith(
            "CLIPEncoderLayerTTNN_"
        ):
            layer_idx = int(node.name.split("_")[-1])
            layers[layer_idx] = {}

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    if method_name != "__init__":
                        layers[layer_idx][method_name] = item

    return layers


def analyze_similarity():
    """Analyze similarity between layer methods."""
    with open("clip_encoder_layers_generated.py", "r") as f:
        source = f.read()

    layers = extract_layer_methods(source)

    print(f"Found {len(layers)} layer classes\n")

    # Group methods by their normalized hash
    method_groups = defaultdict(lambda: defaultdict(list))

    for layer_idx, methods in sorted(layers.items()):
        for method_name, method_def in methods.items():
            try:
                method_hash, normalized_code = get_method_hash(method_def)
                method_groups[method_name][method_hash].append(
                    (layer_idx, normalized_code)
                )
            except Exception as e:
                print(f"Error processing layer {layer_idx} method {method_name}: {e}")

    # Report findings
    print("=" * 80)
    print("METHOD SIMILARITY ANALYSIS")
    print("=" * 80)

    for method_name in sorted(method_groups.keys()):
        hash_groups = method_groups[method_name]
        print(f"\n{method_name}:")
        print(
            f"  Total layers with this method: {sum(len(layers) for layers in hash_groups.values())}"
        )
        print(f"  Unique structures: {len(hash_groups)}")

        if len(hash_groups) == 1:
            print(f"  ✓ ALL LAYERS ARE IDENTICAL for this method!")
            layers_list = list(hash_groups.values())[0]
            print(f"    Layers: {[l[0] for l in layers_list]}")
        else:
            print(f"  Groups:")
            for i, (hash_val, layers_list) in enumerate(
                sorted(hash_groups.items(), key=lambda x: -len(x[1]))
            ):
                layer_indices = [l[0] for l in layers_list]
                print(f"    Group {i+1} ({len(layers_list)} layers): {layer_indices}")

    # Check if we can reduce to fewer unique implementations
    print("\n" + "=" * 80)
    print("POTENTIAL DEDUPLICATION")
    print("=" * 80)

    # Check forward methods specifically
    forward_groups = method_groups.get("forward", {})
    if len(forward_groups) <= 3:
        print(f"\nforward() has only {len(forward_groups)} unique structure(s)")
        for hash_val, layers_list in forward_groups.items():
            layer_indices = [l[0] for l in layers_list]
            print(f"  Layers {layer_indices}:")
            # Print first few lines of normalized code
            code = layers_list[0][1]
            for line in code.split("\n")[:10]:
                print(f"    {line}")
            print("    ...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_identical = True
    for method_name, hash_groups in method_groups.items():
        if len(hash_groups) > 1:
            all_identical = False
            print(f"  {method_name}: {len(hash_groups)} unique structures")

    if all_identical:
        print("  All methods are identical across layers!")
        print("  → Can use a single parameterized class instead of 31 classes")


if __name__ == "__main__":
    analyze_similarity()
