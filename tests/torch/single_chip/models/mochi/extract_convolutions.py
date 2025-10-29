#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to extract ttir.convolution operations from TTIR files.
Usage: python extract_convolutions.py <path_to_ttir_file>
"""

import json
import re
import sys
from typing import Any, Dict, List


def extract_array_values(array_str: str) -> List:
    """Extract values from array notation like 'array<i64: 1, 1, 1>'"""
    match = re.search(r"array<[^:]+:\s*([^>]+)>", array_str)
    if match:
        values = match.group(1).strip()
        # Handle different types (i64, i1, etc.)
        return [v.strip() for v in values.split(",")]
    return []


def extract_tensor_shape(tensor_str: str) -> Dict[str, str]:
    """Extract tensor shape and dtype from notation like 'tensor<1x768x9x62x108xbf16>'"""
    match = re.search(r"tensor<([^>]+)>", tensor_str)
    if match:
        full_shape = match.group(1)
        # Split shape and dtype (last part after 'x' that's not a digit)
        parts = full_shape.split("x")
        if len(parts) > 0:
            # Last part is dtype if it contains letters
            dtype = parts[-1] if any(c.isalpha() for c in parts[-1]) else None
            shape = parts[:-1] if dtype else parts
            return {"shape": "x".join(shape), "dtype": dtype, "full": full_shape}
    return {"shape": "", "dtype": "", "full": tensor_str}


def parse_convolution_layout(layout_str: str) -> Dict[str, Any]:
    """Parse the convolution_layout attribute"""
    layout = {}

    # Extract each field from the layout
    fields = [
        "input_batch",
        "input_feature",
        "input_spatial_dimensions",
        "kernel_output_feature",
        "kernel_input_feature",
        "kernel_spatial_dimensions",
        "output_batch",
        "output_feature",
        "output_spatial_dimensions",
    ]

    for field in fields:
        pattern = rf"{field}\s*=\s*(\S+?)(?:,|\s|>)"
        match = re.search(pattern, layout_str)
        if match:
            layout[field] = match.group(1).strip()

    return layout


def extract_location_definitions(content: str) -> Dict[str, str]:
    """Extract location definitions from the TTIR file"""
    loc_map = {}

    # Pattern to match: #loc123 = loc("actual location string")
    loc_def_pattern = r'(#loc\d+)\s*=\s*loc\("([^"]+)"\)'

    for match in re.finditer(loc_def_pattern, content):
        loc_id = match.group(1)
        loc_string = match.group(2)
        loc_map[loc_id] = loc_string

    return loc_map


def get_convolution_signature(conv: Dict[str, Any]) -> str:
    """Generate a unique signature for a convolution based on its properties"""
    signature_parts = []

    # Tensor shapes
    signature_parts.append(f"input:{conv['tensor_types']['input']['full']}")
    signature_parts.append(f"weight:{conv['tensor_types']['weight']['full']}")
    signature_parts.append(f"output:{conv['tensor_types']['output']['full']}")

    # Attributes
    attrs = conv["attributes"]
    signature_parts.append(f"bgc:{attrs.get('batch_group_count')}")
    signature_parts.append(f"fgc:{attrs.get('feature_group_count')}")
    signature_parts.append(f"id:{','.join(attrs.get('input_dilation', []))}")
    signature_parts.append(f"pad:{','.join(attrs.get('padding', []))}")
    signature_parts.append(f"wd:{','.join(attrs.get('weight_dilation', []))}")
    signature_parts.append(f"wr:{','.join(attrs.get('window_reversal', []))}")
    signature_parts.append(f"ws:{','.join(attrs.get('window_strides', []))}")

    # Convolution layout
    if "convolution_layout" in attrs:
        layout = attrs["convolution_layout"]
        signature_parts.append(f"layout:{json.dumps(layout, sort_keys=True)}")

    return "|".join(signature_parts)


def deduplicate_convolutions(
    convolutions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only unique convolutions and track all occurrences"""
    seen_signatures = {}
    unique_convolutions = []

    for conv in convolutions:
        signature = get_convolution_signature(conv)

        if signature not in seen_signatures:
            # First occurrence - add to unique list
            conv["occurrences"] = [
                {
                    "result": conv["result"],
                    "location_ref": conv["location_ref"],
                    "location": conv["location"],
                    "operands": conv["operands"],
                }
            ]
            seen_signatures[signature] = len(unique_convolutions)
            unique_convolutions.append(conv)
        else:
            # Duplicate - add to occurrences list
            idx = seen_signatures[signature]
            unique_convolutions[idx]["occurrences"].append(
                {
                    "result": conv["result"],
                    "location_ref": conv["location_ref"],
                    "location": conv["location"],
                    "operands": conv["operands"],
                }
            )

    return unique_convolutions


def extract_convolutions(file_path: str) -> List[Dict[str, Any]]:
    """Extract all convolution operations from the TTIR file"""

    with open(file_path, "r") as f:
        content = f.read()

    # First, extract all location definitions
    loc_map = extract_location_definitions(content)

    # Find all convolution operations (may span multiple lines)
    # Pattern to match from result variable through the end of the operation including loc
    conv_pattern = r'(%\d+)\s*=\s*"ttir\.convolution"\(([^)]+)\)\s*<\{([^}]+)\}>\s*:\s*\(([^)]+)\)\s*->\s*(\S+)\s*loc\(([^)]+)\)'

    convolutions = []

    for match in re.finditer(conv_pattern, content, re.DOTALL):
        result_var = match.group(1)
        operands = match.group(2)
        attributes = match.group(3)
        input_types = match.group(4)
        output_type = match.group(5)
        location_ref = match.group(6)

        # Resolve location reference
        location_string = loc_map.get(location_ref, location_ref)

        # Parse operands
        operand_list = [op.strip() for op in operands.split(",")]

        # Parse input tensor types
        input_type_list = [t.strip() for t in input_types.split(",")]

        # Extract individual attributes
        conv_data = {
            "result": result_var,
            "location_ref": location_ref,
            "location": location_string,
            "operands": {
                "input": operand_list[0] if len(operand_list) > 0 else None,
                "weight": operand_list[1] if len(operand_list) > 1 else None,
                "output": operand_list[2] if len(operand_list) > 2 else None,
            },
            "attributes": {},
            "tensor_types": {
                "input": (
                    extract_tensor_shape(input_type_list[0])
                    if len(input_type_list) > 0
                    else None
                ),
                "weight": (
                    extract_tensor_shape(input_type_list[1])
                    if len(input_type_list) > 1
                    else None
                ),
                "output": extract_tensor_shape(output_type),
            },
        }

        # Extract batch_group_count
        match_bgc = re.search(r"batch_group_count\s*=\s*(\d+)", attributes)
        if match_bgc:
            conv_data["attributes"]["batch_group_count"] = int(match_bgc.group(1))

        # Extract feature_group_count
        match_fgc = re.search(r"feature_group_count\s*=\s*(\d+)", attributes)
        if match_fgc:
            conv_data["attributes"]["feature_group_count"] = int(match_fgc.group(1))

        # Extract convolution_layout
        layout_match = re.search(
            r"convolution_layout\s*=\s*#ttir<convolution_layout\s+([^>]+)>", attributes
        )
        if layout_match:
            conv_data["attributes"]["convolution_layout"] = parse_convolution_layout(
                layout_match.group(1)
            )

        # Extract input_dilation
        match_id = re.search(r"input_dilation\s*=\s*(array<[^>]+>)", attributes)
        if match_id:
            conv_data["attributes"]["input_dilation"] = extract_array_values(
                match_id.group(1)
            )

        # Extract padding
        match_pad = re.search(r"padding\s*=\s*(array<[^>]+>)", attributes)
        if match_pad:
            conv_data["attributes"]["padding"] = extract_array_values(
                match_pad.group(1)
            )

        # Extract weight_dilation
        match_wd = re.search(r"weight_dilation\s*=\s*(array<[^>]+>)", attributes)
        if match_wd:
            conv_data["attributes"]["weight_dilation"] = extract_array_values(
                match_wd.group(1)
            )

        # Extract window_reversal
        match_wr = re.search(r"window_reversal\s*=\s*(array<[^>]+>)", attributes)
        if match_wr:
            conv_data["attributes"]["window_reversal"] = extract_array_values(
                match_wr.group(1)
            )

        # Extract window_strides
        match_ws = re.search(r"window_strides\s*=\s*(array<[^>]+>)", attributes)
        if match_ws:
            conv_data["attributes"]["window_strides"] = extract_array_values(
                match_ws.group(1)
            )

        convolutions.append(conv_data)

    return convolutions


def print_convolution_summary(
    conv: Dict[str, Any], index: int, show_occurrences: bool = False
):
    """Print a human-readable summary of a convolution operation"""
    print(f"\n{'='*80}")
    if "occurrences" in conv:
        print(
            f"Convolution #{index + 1}: {conv['occurrences'][0]['result']} (+ {len(conv['occurrences']) - 1} more occurrences)"
        )
    else:
        print(f"Convolution #{index + 1}: {conv['result']}")
    print(f"{'='*80}")

    # Show first occurrence operands
    if "occurrences" in conv:
        print("\nOperands (first occurrence):")
        print(f"  Input:  {conv['occurrences'][0]['operands']['input']}")
        print(f"  Weight: {conv['occurrences'][0]['operands']['weight']}")
        print(f"  Output: {conv['occurrences'][0]['operands']['output']}")
    else:
        print("\nOperands:")
        print(f"  Input:  {conv['operands']['input']}")
        print(f"  Weight: {conv['operands']['weight']}")
        print(f"  Output: {conv['operands']['output']}")

    print("\nTensor Shapes:")
    print(f"  Input:  {conv['tensor_types']['input']['full']}")
    print(f"  Weight: {conv['tensor_types']['weight']['full']}")
    print(f"  Output: {conv['tensor_types']['output']['full']}")

    # Show location info
    if "occurrences" in conv:
        print(f"\nLocation Reference: {conv['occurrences'][0]['location_ref']}")
        print(f"Location: {conv['occurrences'][0]['location']}")

        if show_occurrences and len(conv["occurrences"]) > 1:
            print(f"\nAll Occurrences ({len(conv['occurrences'])} total):")
            for i, occ in enumerate(conv["occurrences"]):
                print(
                    f"  {i+1}. {occ['result']} - {occ['location_ref']} - {occ['location']}"
                )
    else:
        print(f"\nLocation Reference: {conv['location_ref']}")
        print(f"Location: {conv['location']}")

    attrs = conv["attributes"]
    print("\nAttributes:")
    print(f"  Batch Group Count:   {attrs.get('batch_group_count', 'N/A')}")
    print(f"  Feature Group Count: {attrs.get('feature_group_count', 'N/A')}")
    print(f"  Input Dilation:      {attrs.get('input_dilation', 'N/A')}")
    print(f"  Padding:             {attrs.get('padding', 'N/A')}")
    print(f"  Weight Dilation:     {attrs.get('weight_dilation', 'N/A')}")
    print(f"  Window Reversal:     {attrs.get('window_reversal', 'N/A')}")
    print(f"  Window Strides:      {attrs.get('window_strides', 'N/A')}")

    if "convolution_layout" in attrs:
        print("\n  Convolution Layout:")
        layout = attrs["convolution_layout"]
        for key, value in layout.items():
            print(f"    {key:30s}: {value}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_convolutions.py <path_to_ttir_file> [options]")
        print("Options:")
        print("  --json: Output as JSON to stdout")
        print(
            "  --save-json <file>: Save output to JSON file (default: locations.json)"
        )
        print("  --unique: Keep only unique convolutions (deduplicate)")
        print("  --show-occurrences: Show all occurrences for unique convolutions")
        sys.exit(1)

    file_path = sys.argv[1]
    output_json = "--json" in sys.argv
    save_json = "--save-json" in sys.argv
    unique = "--unique" in sys.argv
    show_occurrences = "--show-occurrences" in sys.argv

    # Determine output file name
    output_file = "locations.json"
    if save_json:
        try:
            save_json_idx = sys.argv.index("--save-json")
            if save_json_idx + 1 < len(sys.argv) and not sys.argv[
                save_json_idx + 1
            ].startswith("--"):
                output_file = sys.argv[save_json_idx + 1]
        except (ValueError, IndexError):
            pass

    try:
        convolutions = extract_convolutions(file_path)
        total_convolutions = len(convolutions)

        # Deduplicate if requested
        if unique:
            convolutions = deduplicate_convolutions(convolutions)

        if output_json:
            # Output as JSON to stdout only
            print(json.dumps(convolutions, indent=2))
        else:
            # Output human-readable format
            if unique:
                print(
                    f"\nFound {len(convolutions)} unique convolution(s) from {total_convolutions} total in {file_path}"
                )
            else:
                print(
                    f"\nFound {len(convolutions)} convolution operation(s) in {file_path}"
                )

            for i, conv in enumerate(convolutions):
                print_convolution_summary(conv, i, show_occurrences)

            print(f"\n{'='*80}")
            if unique:
                print(
                    f"Total: {len(convolutions)} unique convolution(s) (from {total_convolutions} total)"
                )
            else:
                print(f"Total: {len(convolutions)} convolution(s)")
            print(f"{'='*80}\n")

        # Save to JSON file if requested
        if save_json:
            with open(output_file, "w") as f:
                json.dump(convolutions, f, indent=2)
            if unique:
                print(
                    f"\nSaved {len(convolutions)} unique convolution(s) (from {total_convolutions} total) to {output_file}"
                )
            else:
                print(f"\nSaved {len(convolutions)} convolution(s) to {output_file}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
